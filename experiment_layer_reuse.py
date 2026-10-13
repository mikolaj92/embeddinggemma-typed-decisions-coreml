"""Native1024/Q256 trained single-layer graph reuse; NOT a full encoder/head.

Run in the pinned environment, in TWO separate processes:
  python experiment_layer_reuse.py --output /tmp/layer-reuse-new
  python experiment_layer_reuse.py --output /tmp/layer-reuse-new --validate

Reference CPU prepares original whole PLE(E0) and original pre-norm/QKV,
Q/K/V norms and absolute RoPE from an immutable synthetic x snapshot once.
Each original eager layer independently computes FP32 gold before stabilization.
One FP16 CPU_AND_NE package per trained layer (local0/global5) is reused for
all four tiles. No context extension, semantic quality approval, full-encoder
parity, production modification, or verified ANE placement is implied.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

SIZE = 1024
TILE = 256
LAYERS = {"local": 0, "global": 5}
NAMES = ("residual", "per_layer_input", "q", "k", "v", "allowed")
COSINE_MIN = 0.998


def make_block(layer):
    """Share only the trained post-QKV modules; do not register a full layer."""
    import torch
    from torch import nn

    class TileBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = layer.self_attn.o_proj
            self.post_attention_layernorm = layer.post_attention_layernorm
            self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
            self.mlp = layer.mlp
            self.post_feedforward_layernorm = layer.post_feedforward_layernorm
            self.ple_block = layer.ple_block
            self.register_buffer("layer_scalar", layer.layer_scalar)
            self.groups = layer.self_attn.num_key_value_groups
            self.heads = layer.self_attn.q_proj.out_features // layer.self_attn.head_dim

        def forward(self, residual, per_layer_input, q, k, v, allowed):
            permit = allowed != 0
            heads = []
            # Explicit GQA over ALL 1024 keys; original attention scaling is 1.0.
            for head in range(self.heads):
                kv = head // self.groups
                scores = q[:, head] @ k[:, kv].transpose(-1, -2)
                scores = torch.where(permit, scores, torch.full_like(scores, -1e4))
                probs = torch.softmax(scores, dim=-1) * permit.to(scores.dtype)
                heads.append(probs @ v[:, kv])
            attended = self.o_proj(torch.cat(heads, dim=-1))
            x = residual + self.post_attention_layernorm(attended)
            x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
            return self.ple_block(x, per_layer_input) * self.layer_scalar

    return TileBlock().eval()


def synthetic(shape, phase):
    i = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return np.ascontiguousarray(0.2 * (np.sin(i * 0.017 + phase)
                                     + 0.5 * np.cos(i * 0.031 + phase)), dtype=np.float32)


def allowed_mask(kind, valid):
    positions = np.arange(SIZE)
    geometry = (np.abs(positions[:, None] - positions[None, :]) <= 512
                if kind == "local" else np.ones((SIZE, SIZE), dtype=bool))
    # Original eager masking pads KEYS, not query residuals. No empty rows here.
    return np.ascontiguousarray((geometry & valid[None, :])[None], dtype=np.int32)


def tile_inputs(buffers, start):
    end = start + TILE
    return {"residual": buffers["residual"][:, start:end].copy(),
            "per_layer_input": buffers["per_layer_input"][:, start:end].copy(),
            "q": buffers["q"][:, :, start:end].copy(),
            "k": buffers["k"].copy(), "v": buffers["v"].copy(),
            "allowed": buffers["allowed"][:, start:end].copy()}


def metrics(actual, gold):
    if actual.shape != gold.shape:
        raise ValueError(f"Output shape {actual.shape} != {gold.shape}")
    finite = bool(np.isfinite(actual).all() and np.isfinite(gold).all())
    cosine = max_abs = None
    if finite:
        a, b = actual.astype(np.float64), gold.astype(np.float64)
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cosine = float(np.min(np.sum(a * b, axis=-1) / np.maximum(denominator, 1e-30)))
        max_abs = float(np.max(np.abs(a - b)))
    return {"finite": finite, "cosine_min_hidden_rows": cosine, "max_abs": max_abs,
            "passed": finite and cosine >= COSINE_MIN}


def export(output, upstream):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from convert import reference, stabilize
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import apply_rotary_pos_emb

    output.mkdir(parents=True, exist_ok=False)
    model = reference(upstream).cpu().float().eval()
    text = model.encoder.language_model
    text.config._attn_implementation = "eager"
    if (text.config.sliding_window != 512 or text.config.hidden_size_per_layer_input != 512
            or text.config.max_position_embeddings < SIZE
            or text.config.layer_types[0] != "sliding_attention"
            or text.config.layer_types[5] != "full_attention"):
        raise ValueError("Requires original native1024 local0/global5 architecture")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    shape = (1, SIZE, text.config.hidden_size)
    e0 = torch.from_numpy(synthetic(shape, 0.1))
    x = torch.from_numpy(synthetic(shape, 0.7))
    positions = torch.arange(SIZE).unsqueeze(0)
    valid = np.ones(SIZE, dtype=bool)
    valid[np.arange(SIZE) % 11 == 0] = False
    valid[203:219] = False
    valid[977:] = False
    padding = torch.from_numpy(valid[None].astype(np.int32))
    report = {"native_size": SIZE, "query_tile": TILE, "compute_units": "CPU_AND_NE",
              "compute_precision": "FLOAT16", "source_upstream": str(upstream),
              "oracle": "original eager layer.forward before stabilization",
              "quality_approved": False, "ane_execution_verified": False, "layers": {}}
    with torch.inference_mode():
        # Original whole prepared PLE; E0 and snapshot x intentionally differ.
        ple = text.ple(e0).detach().clone()
        for kind, index in LAYERS.items():
            layer = text.layers[index]
            layer.self_attn.config._attn_implementation = "eager"
            typ = text.config.layer_types[index]
            rotary = text.rotary_emb(x, positions, typ)
            per = ple[:, :, index, :].contiguous()
            mask_fn = create_bidirectional_sliding_window_mask if kind == "local" else create_bidirectional_mask
            oracle_mask = mask_fn(config=text.config, inputs_embeds=x, attention_mask=padding)
            allowed = allowed_mask(kind, valid)
            if oracle_mask is None or not np.array_equal((oracle_mask[:, 0] == 0).numpy(), allowed != 0):
                raise RuntimeError("Original eager mask differs from absolute <=512/key-padding geometry")
            # Independent full-layer oracle, NOT TileBlock or a tiled attention oracle.
            gold = layer(hidden_states=x.clone(), per_layer_input=per,
                         attention_mask=oracle_mask, position_embeddings=rotary).detach().clone()
            a = layer.self_attn
            h = layer.input_layernorm(x)
            def projected(proj, norm):
                return norm(proj(h).view(1, SIZE, -1, a.head_dim).transpose(1, 2))
            q = projected(a.q_proj, a.q_norm)
            k = projected(a.k_proj, a.k_norm)
            v = projected(a.v_proj, a.v_norm)
            q, k = apply_rotary_pos_emb(q, k, *rotary)
            buffers = {"e0": e0.numpy().copy(), "residual": x.numpy().copy(),
                       "per_layer_input": per.numpy().copy(), "q": q.numpy().copy(),
                       "k": k.numpy().copy(), "v": v.numpy().copy(), "allowed": allowed,
                       "padding": padding.numpy().copy(), "expected": gold.numpy().copy()}
            # Freeze host snapshots. Tile copies are the only execution inputs.
            for array in buffers.values():
                array.flags.writeable = False
            np.savez(output / f"{kind}_inputs_gold.npz", **buffers)
            block = make_block(layer)
            stabilize(block)  # Oracle and QKV are already saved, still original FP32.
            outputs = []
            for start in range(0, SIZE, TILE):
                inputs = tile_inputs(buffers, start)
                args = tuple(torch.from_numpy(inputs[name]) for name in NAMES)
                outputs.append(block(*args).numpy().copy())
            check = metrics(np.concatenate(outputs, axis=1), buffers["expected"])
            if not check["passed"] or check["max_abs"] > 1e-4:
                raise RuntimeError(f"Stable tiled CPU parity failed: {kind}: {check}")
            inputs = tile_inputs(buffers, 0)
            args = tuple(torch.from_numpy(inputs[name]) for name in NAMES)
            traced = torch.jit.trace(block, args, check_trace=False)
            import coremltools as ct
            converted = ct.convert(
                traced, convert_to="mlprogram",
                inputs=[ct.TensorType(name=name, shape=tensor.shape,
                                      dtype=np.int32 if name == "allowed" else np.float32)
                        for name, tensor in zip(NAMES, args)],
                outputs=[ct.TensorType(name="hidden", dtype=np.float32)],
                minimum_deployment_target=ct.target.macOS15,
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
            converted.short_description = f"Experimental trained {kind} layer{index} native1024 Q256 reuse only"
            converted.save(str(output / f"{kind}_layer.mlpackage"))
            report["layers"][kind] = {"index": index, "torch_check": check, "skip_model_load": True}
            print(json.dumps({"phase": "export", "kind": kind, "torch_check": check}), flush=True)
            del converted, traced, block, buffers, outputs, args, inputs, gold, q, k, v, h
            gc.collect()
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Export ready; run --validate in a NEW process. No native validation performed.", flush=True)


def validate(output):
    # No explicit Torch import, model load or oracle creation in this process.
    import coremltools as ct
    report = {"native_size": SIZE, "query_tile": TILE, "cosine_min_threshold": COSINE_MIN,
              "compute_units": "CPU_AND_NE", "compute_precision": "FLOAT16",
              "quality_approved": False, "ane_execution_verified": False,
              "full_encoder_verified": False, "production_changed": False, "layers": {}, "passed": True}
    for kind, index in LAYERS.items():
        with np.load(output / f"{kind}_inputs_gold.npz", allow_pickle=False) as archive:
            buffers = {name: archive[name].copy() for name in archive.files}
        snapshots = {name: array.copy() for name, array in buffers.items()}
        for array in buffers.values():
            array.flags.writeable = False
        model = ct.models.MLModel(str(output / f"{kind}_layer.mlpackage"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
        pieces = [np.array(model.predict(tile_inputs(buffers, start))["hidden"], copy=True)
                  for start in range(0, SIZE, TILE)]
        actual = np.concatenate(pieces, axis=1)
        check = metrics(actual, buffers["expected"])
        preserved = all(np.array_equal(buffers[name], old) for name, old in snapshots.items())
        check["passed"] = check["passed"] and preserved
        report["layers"][kind] = {"index": index, "package_instances": 1,
                                  "query_starts": list(range(0, SIZE, TILE)),
                                  "host_buffers_preserved": preserved, **check}
        report["passed"] = report["passed"] and check["passed"]
        del model, buffers, snapshots, pieces, actual
        gc.collect()
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "upstream")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    artifacts = (Path(__file__).resolve().parent / "artifacts").resolve()
    upstream = args.upstream.expanduser().resolve()
    if any(output == protected or output.is_relative_to(protected) for protected in (artifacts, upstream)):
        parser.error("--output must be outside resolved production artifacts and upstream")
    if args.validate:
        if not output.is_dir():
            parser.error("--validate requires an existing exported experiment directory")
        return validate(output)
    if args.output.expanduser().exists() or args.output.expanduser().is_symlink():
        parser.error("Export requires a NEW --output directory; nothing will be overwritten")
    if not upstream.is_dir():
        parser.error("--upstream must be an existing local trained checkpoint")
    export(output, upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
