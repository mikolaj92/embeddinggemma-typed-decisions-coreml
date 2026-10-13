"""Native512/Q256 weight-free attention graph-reuse pilot; NOT an encoder.

Run in the repository's pinned Python environment, in TWO processes:
  python experiment_attention_reuse.py --output /tmp/attention-reuse-new
  python experiment_attention_reuse.py --output /tmp/attention-reuse-new --validate

Export never loads a package (skip_model_load=True). Validation loads one
package at a time and does not explicitly import torch or build an oracle.
Only deterministic synthetic attention inputs are used; no model is loaded.
Q/K/V are assumed already normalized and Q/K absolute-RoPE transformed.
This does NOT verify normalization, RoPE, full-encoder parity, ANE placement,
context extension, quality, production inference, or a decision head.
At native512, abs(query-key)<=512 permits every key: local/global geometry
cannot demonstrate the sliding-window boundary. Native1024 is a later test.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

SIZE = 512
TILE = 256
HEADS = 4
KINDS = {"local": (2, 256), "global": (1, 512)}
CASES = ("all_valid", "padding_holes", "sparse_empty", "all_masked")
COSINE_MIN = 0.998


def make_kernel(kv_heads):
    """Lazy Torch import: --validate never constructs this weight-free module."""
    import torch
    from torch import nn

    class TileAttention(nn.Module):
        def forward(self, q, k, v, allowed):
            permit = allowed != 0
            outputs = []
            # Explicit heads, same GQA mapping as experimental_encoder.py.
            for head in range(HEADS):
                kv = head // (HEADS // kv_heads)
                scores = q[:, head] @ k[:, kv].transpose(-1, -2)
                # Original scaling is 1.0, NOT 1/sqrt(d).
                scores = torch.where(permit, scores, torch.full_like(scores, -1e4))
                probs = torch.softmax(scores, dim=-1) * permit.to(scores.dtype)
                # Even a fully masked row remains finite and becomes exactly zero.
                outputs.append((probs @ v[:, kv]).unsqueeze(1))
            return torch.cat(outputs, dim=1)

    return TileAttention().eval()


def synthetic(shape, phase, amplitude, offset=0.0):
    """Analytic small-amplitude float32 values, no RNG or model inference."""
    index = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    values = offset + amplitude * (np.sin(index * 0.017 + phase)
                                  + 0.5 * np.cos(index * 0.031 + phase))
    return np.ascontiguousarray(values, dtype=np.float32)


def whole_mask(kind, case):
    positions = np.arange(SIZE)
    valid = np.ones(SIZE, dtype=bool)
    if case == "padding_holes":
        valid[positions % 11 == 0] = False
        valid[203:219] = False
        valid[487:] = False
    elif case == "sparse_empty":
        valid[:] = False
        valid[[1, 127, 255, 256, 383, 510]] = True
    elif case == "all_masked":
        valid[:] = False
    elif case != "all_valid":
        raise ValueError(case)
    # Absolute positions, inclusive original distance <=512, no tile-relative mask.
    geometry = (np.abs(positions[:, None] - positions[None, :]) <= 512
                if kind == "local" else np.ones((SIZE, SIZE), dtype=bool))
    # Key padding plus explicit query padding gives padded query rows exact zero.
    allowed = geometry & valid[:, None] & valid[None, :]
    return np.ascontiguousarray(allowed[None], dtype=np.int32)


def dense_oracle(q, k, v, allowed):
    """Independent whole512 PyTorch attention; no tile kernel or query splitting."""
    import torch
    with torch.inference_mode():
        queries = torch.from_numpy(q)
        keys = torch.from_numpy(k).repeat_interleave(HEADS // k.shape[1], dim=1)
        values = torch.from_numpy(v).repeat_interleave(HEADS // v.shape[1], dim=1)
        permit = torch.from_numpy(allowed).bool().unsqueeze(1)
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        scores.masked_fill_(~permit, -10000.0)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities.masked_fill_(~permit, 0.0)
        return torch.einsum("bhqk,bhkd->bhqd", probabilities, values).numpy().copy()


def metrics(actual, gold, allowed):
    if actual.shape != gold.shape:
        raise ValueError(f"Output shape {actual.shape} != {gold.shape}")
    finite = bool(np.isfinite(actual).all() and np.isfinite(gold).all())
    active = np.broadcast_to(allowed.any(axis=-1)[:, None, :], gold.shape[:-1])
    empty = ~active
    empty_zero = bool(np.all(actual[empty] == 0) and np.all(gold[empty] == 0))
    cosine = None
    if active.any() and finite:
        a, b = actual[active].astype(np.float64), gold[active].astype(np.float64)
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cosine = float(np.min(np.sum(a * b, axis=-1) / np.maximum(denominator, 1e-30)))
    max_abs = float(np.max(np.abs(actual.astype(np.float64) - gold))) if finite else None
    return {"finite": finite, "cosine_min_nonempty_head_rows": cosine,
            "max_abs": max_abs, "empty_padded_exact_zero": empty_zero,
            "nonempty_query_rows": int(allowed.any(axis=-1).sum()),
            "passed": finite and empty_zero and (cosine is None or cosine >= COSINE_MIN)}


def export(output):
    import torch
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    for kind, (kv_heads, width) in KINDS.items():
        q = synthetic((1, HEADS, SIZE, width), 0.1, 0.025)
        k = synthetic((1, kv_heads, SIZE, width), 0.7, 0.025)
        v = synthetic((1, kv_heads, SIZE, width), 1.3, 0.06, offset=0.02)
        fixture = {"q": q, "k": k, "v": v}
        module = make_kernel(kv_heads)
        cpu_checks = {}
        with torch.inference_mode():
            for case in CASES:
                allowed = whole_mask(kind, case)
                gold = dense_oracle(q, k, v, allowed)
                fixture[f"{case}_allowed"] = allowed
                fixture[f"{case}_gold"] = gold
                # Same module instance for both absolute query slices.
                tiles = [module(torch.from_numpy(q[:, :, start:start + TILE].copy()),
                                torch.from_numpy(k), torch.from_numpy(v),
                                torch.from_numpy(allowed[:, start:start + TILE].copy())).numpy()
                         for start in (0, TILE)]
                cpu_checks[case] = metrics(np.concatenate(tiles, axis=2), gold, allowed)
                if not cpu_checks[case]["passed"]:
                    raise RuntimeError(f"Dense-vs-tile Torch failure: {kind}/{case}")
            # Persist full host buffers and independent dense gold BEFORE conversion.
            np.savez(output / f"{kind}_inputs_gold.npz", **fixture)
            args = (torch.from_numpy(q[:, :, :TILE].copy()), torch.from_numpy(k),
                    torch.from_numpy(v), torch.from_numpy(fixture["all_valid_allowed"][:, :TILE].copy()))
            traced = torch.jit.trace(module, args, check_trace=False)
        assert not tuple(module.parameters()) and not tuple(module.buffers())
        del module, fixture, tiles, gold
        gc.collect()
        import coremltools as ct
        started = time.monotonic()
        converted = ct.convert(
            traced, convert_to="mlprogram",
            inputs=[ct.TensorType(name=name, shape=tensor.shape, dtype=dtype)
                    for name, tensor, dtype in zip(("q", "k", "v", "allowed"), args,
                                                   (np.float32, np.float32, np.float32, np.int32))],
            outputs=[ct.TensorType(name="attended", dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        converted.short_description = "Experimental weight-free native512 Q256 attention reuse only"
        converted.save(str(output / f"{kind}_attention.mlpackage"))
        print(json.dumps({"phase": "export", "kind": kind, "seconds": time.monotonic() - started,
                          "skip_model_load": True, "torch_checks": cpu_checks}), flush=True)
        del converted, traced, args, q, k, v, allowed
        gc.collect()
    print("Export ready. Run --validate in a NEW process; no native validation was performed.", flush=True)


def validate(output):
    import coremltools as ct
    report = {"experiment": "single_attention_graph_reuse", "native_size": SIZE,
              "query_tile": TILE, "compute_units": "CPU_AND_NE", "compute_precision": "FLOAT16",
              "cosine_min_threshold": COSINE_MIN, "ane_execution_verified": False,
              "absolute_rope_verified": False, "local_boundary_verified": False,
              "quality_approved": False, "production_changed": False, "kernels": {}, "passed": True}
    for kind in KINDS:
        with np.load(output / f"{kind}_inputs_gold.npz", allow_pickle=False) as archive:
            buffers = {name: archive[name].copy() for name in archive.files}
        snapshots = {name: array.copy() for name, array in buffers.items()}
        started = time.monotonic()
        # One package instance reused for both query tiles and every mask case.
        model = ct.models.MLModel(str(output / f"{kind}_attention.mlpackage"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
        results = {}
        for case in CASES:
            allowed = buffers[f"{case}_allowed"]
            pieces = []
            for start in (0, TILE):
                inputs = {"q": np.ascontiguousarray(buffers["q"][:, :, start:start + TILE]),
                          "k": buffers["k"], "v": buffers["v"],
                          "allowed": np.ascontiguousarray(allowed[:, start:start + TILE])}
                # Copy returned tile: a subsequent prediction must not overwrite it.
                pieces.append(np.array(model.predict(inputs)["attended"], copy=True))
            actual = np.concatenate(pieces, axis=2)
            results[case] = metrics(actual, buffers[f"{case}_gold"], allowed)
            report["passed"] &= results[case]["passed"]
        preserved = all(np.array_equal(array, snapshots[name]) for name, array in buffers.items())
        report["passed"] &= preserved
        report["kernels"][kind] = {"load_and_predict_seconds": time.monotonic() - started,
                                   "package_instances": 1, "query_starts": [0, TILE],
                                   "host_buffers_preserved": preserved, "cases": results}
        del model, buffers, snapshots, inputs, pieces, actual
        gc.collect()
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path,
                        help="New experiment directory outside production artifacts; existing for --validate")
    parser.add_argument("--validate", action="store_true", help="Load saved packages in a separate process")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    artifacts = Path(__file__).resolve().parent / "artifacts"
    if output == artifacts or output.is_relative_to(artifacts):
        parser.error("--output must be outside production artifacts")
    if args.validate:
        if not output.is_dir():
            parser.error("--validate requires an existing exported experiment directory")
        return validate(output)
    if output.exists():
        parser.error("Export requires a NEW --output directory; nothing will be overwritten")
    export(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
