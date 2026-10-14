"""Isolated native512-or-1024/Q256 full trained encoder reuse proof; no quality approval.

Two processes in the pinned environment (export never loads Core ML models):
  python experiment_encoder_reuse.py --output /tmp/encoder-reuse-NEW --upstream artifacts-8192/upstream
  python experiment_encoder_reuse.py --output /tmp/encoder-reuse-NEW --validate
Add --head at export for four original rendered state/question/option texts and
one synthetic noul head parity check. Add --size 1024 --long-state for the fixed
16-record museum extension; --validate derives size/scope from saved fixtures.
No production artifact/service is touched.
CPU FP32 prepares original PLE(E0), normalized QKV and absolute RoPE. Each of
24 distinct FP16 post-QKV packages is reused across size/256 query tiles, with all
512 or 1024 keys. Mixed hidden-state error accumulates across the entire encoder.
CPU_AND_NE is a request, not proof of ANE placement or semantic quality.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from experiment_context import FILLER, peak_rss_bytes
from experiment_layer_reuse import NAMES, TILE, make_block, metrics, tile_inputs

LAYER_COUNT = 24
PROBABILITY_MAX_ABS = 0.03


def load_original(upstream, size, checkpoint=None):
    import torch
    from convert import reference

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    model = reference(upstream).cpu().float().eval()
    if checkpoint is not None:
        # mmap avoids another resident copy of the saved original full state.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state, strict=True, assign=True)
        del state
    text = model.encoder.language_model
    text.config._attn_implementation = "eager"
    cfg = text.config
    if (len(text.layers) != LAYER_COUNT or len(cfg.layer_types) != LAYER_COUNT
            or cfg.hidden_size != 512 or cfg.hidden_size_per_layer_input != 512
            or cfg.sliding_window != 512 or cfg.max_position_embeddings < size
            or set(cfg.layer_types) != {"sliding_attention", "full_attention"}):
        raise ValueError("Requires original 24-layer native512/PLE512 architecture")
    for layer in text.layers:
        layer.self_attn.config._attn_implementation = "eager"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def buffers_for(text, index, x, per, padding, size):
    """Read one immutable whole-layer x; never derive K/V from partially updated tiles."""
    import torch
    from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import apply_rotary_pos_emb

    layer = text.layers[index]
    typ = text.config.layer_types[index]
    h = layer.input_layernorm(x)
    a = layer.self_attn

    def projected(projection, norm):
        return norm(projection(h).view(1, size, -1, a.head_dim).transpose(1, 2))

    q, k = projected(a.q_proj, a.q_norm), projected(a.k_proj, a.k_norm)
    v = projected(a.v_proj, a.v_norm)
    rotary = text.rotary_emb(x, torch.arange(size).unsqueeze(0), typ)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    positions = np.arange(size)
    geometry = (np.abs(positions[:, None] - positions[None, :]) <= text.config.sliding_window
                if typ == "sliding_attention" else np.ones((size, size), dtype=bool))
    allowed = np.ascontiguousarray((geometry & padding.numpy().astype(bool))[None], dtype=np.int32)
    mask_fn = create_bidirectional_sliding_window_mask if typ == "sliding_attention" else create_bidirectional_mask
    oracle_mask = mask_fn(config=text.config, inputs_embeds=x, attention_mask=padding)
    if oracle_mask is None or not np.array_equal((oracle_mask[:, 0] == 0).numpy(), allowed != 0):
        raise RuntimeError("Original eager mask differs from full-key tile geometry")
    result = {"residual": x.numpy().copy(), "per_layer_input": per.numpy().copy(),
              "q": q.numpy().copy(), "k": k.numpy().copy(), "v": v.numpy().copy(), "allowed": allowed}
    for value in result.values():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite CPU preparation at layer {index}")
        value.flags.writeable = False
    return result


def finish_embedding(model, x, padding, prompt_length):
    """Original norms/projection and decision-model pooling/projections/L2, CPU FP32."""
    import torch
    import torch.nn.functional as F

    text = model.encoder.language_model
    tokens = text.embedding_projection(text.norm(x))
    mask = padding.clone()
    if prompt_length and not model.config.pooling_include_prompt:
        # Match upstream's right-padded prompt exclusion, not query masking.
        mask[:, :prompt_length] = 0
    m = mask.unsqueeze(-1).to(tokens.dtype)
    pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
    for linear, activation in zip(model.projections, model.activations):
        pooled = activation(linear(pooled))
    return F.normalize(pooled[:, :model.config.embed_dim].float(), dim=-1)


def head_probabilities(model, embeddings, kind):
    import torch

    e = torch.from_numpy(np.concatenate(embeddings, axis=0)).float()
    kinds = torch.tensor([kind], dtype=torch.long)
    mask = torch.ones((1, 2), dtype=torch.bool)
    logits = model.head(e[0:1], e[1:2], e[2:4].unsqueeze(0), kinds, mask).float()
    return (logits / model.temperatures[kinds, None]).softmax(-1).numpy().copy()


def export(output, upstream, with_head, size, long_state):
    wall_start = time.perf_counter()
    inference_seconds = 0.0
    import torch
    from runtime import load_upstream
    from transformers import AutoTokenizer

    # Exclusive creation refuses production directories, prior runs and symlinks.
    if not upstream.is_dir():
        raise ValueError(f"Upstream directory is missing: {upstream}")
    output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    model = load_original(upstream, size)
    text = model.encoder.language_model
    original = load_upstream(upstream)
    spec = original.parse_question("safe", {"type": "noul", "instructions": "Is the proposed action safe?"})
    rendered = [original.render_state("synthetic_safety", {
        "action": "Read the public documentation", "external_changes": False,
        "approval_required": False, "notes": "Read only; do not send private data."})]
    original_rendered_state = rendered[0]
    if long_state:
        # Fixed unrelated museum records, appended after the unchanged read-only facts.
        records = [FILLER[0].format(i=i, v=(i * 7919 + 104729) % 1000003) for i in range(17)]
        rendered[0] += "\n" + "\n".join(records)
    if with_head:
        rendered += [original.render_question(spec), original.render_option(spec, 0), original.render_option(spec, 1)]
    tokenizer = AutoTokenizer.from_pretrained(upstream, local_files_only=True, trust_remote_code=False)
    load_seconds = time.perf_counter() - load_start
    report = {"native_size": size, "query_tile": TILE, "layer_count": LAYER_COUNT,
              "source_upstream": str(upstream), "compute_units": "CPU_AND_NE", "compute_precision": "FLOAT16",
              "oracle": "independent original full model.embed FP32 before any stabilization",
              "quality_approved": False, "semantic_pass_claimed": False,
              "ane_execution_verified": False, "head_enabled": with_head,
              "long_state": long_state, "long_state_records": 17 if long_state else 0,
              "original_rendered_state": original_rendered_state, "factual_gold_label": "true",
              "semantic_note": "Read-only action is safe; numerical parity is not semantic success (original short safe-question prediction was false).",
              "query_starts": list(range(0, size, TILE)), "head_kind": spec.kind_id, "head_options": list(spec.options), "texts": [], "layers": []}
    gold_embeddings = []
    with torch.inference_mode():
        # Save ORIGINAL full checkpoint before sharing/stabilizing post-QKV modules.
        torch.save(model.state_dict(), output / "original_checkpoint.pt")
        for case, raw in enumerate(rendered):
            features = model.tokenize([raw], tokenizer)
            expected_ids = tokenizer((model.config.prompt or "") + raw,
                                     add_special_tokens=True, truncation=False)["input_ids"]
            consumed_ids = features["input_ids"][0][features["attention_mask"][0].bool()].tolist()
            if consumed_ids != expected_ids or not bool((features["attention_mask"] == 1).all()):
                raise ValueError("Original tokenization lost or altered full input IDs; refusing truncation")
            length = len(expected_ids)
            if length > size:
                raise ValueError(f"Proof text exceeds native{size}; refusing truncation")
            if long_state and case == 0:
                assert 512 < length <= 1024, f"Long state must consume >512 and <=1024 actual tokens: {length}"
            prompt_length = features.get("prompt_length", 0)
            batch = tokenizer.pad({"input_ids": features["input_ids"].tolist(),
                                   "attention_mask": features["attention_mask"].tolist()},
                                  padding="max_length", max_length=size, return_tensors="pt")
            ids, padding = batch["input_ids"], batch["attention_mask"]
            taps = []
            handles = [layer.register_forward_hook(lambda _m, _a, out: taps.append(out.detach().numpy().copy()))
                       for layer in text.layers]
            inference_start = time.perf_counter()
            try:
                gold = model.embed(ids, padding, prompt_length=prompt_length).numpy().copy()
            finally:
                inference_seconds += time.perf_counter() - inference_start
                for handle in handles:
                    handle.remove()
            if len(taps) != LAYER_COUNT or not np.isfinite(gold).all() or not all(np.isfinite(t).all() for t in taps):
                raise RuntimeError("Original full FP32 oracle has missing/nonfinite taps")
            e0 = model.get_input_embeddings()(ids).detach().numpy().copy()
            np.savez(output / f"text_{case}_gold.npz", input_ids=ids.numpy(), padding=padding.numpy(),
                     e0=e0, hidden=np.stack(taps), expected=gold)
            report["texts"].append({"rendered": raw, "tokenized_text": (model.config.prompt or "") + raw,
                                    "valid_tokens": int(padding.sum()), "untruncated_tokens": length,
                                    "full_input_ids_equal": True, "prompt_length": int(prompt_length)})
            gold_embeddings.append(gold)
        if with_head:
            inference_start = time.perf_counter()
            probabilities = head_probabilities(model, gold_embeddings, spec.kind_id)
            inference_seconds += time.perf_counter() - inference_start
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Original head probabilities are nonfinite")
            np.save(output / "head_gold.npy", probabilities)
        # ALL independent full-encoder gold is now on disk. Stabilize blocks only.
        from convert import stabilize
        import coremltools as ct
        with np.load(output / "text_0_gold.npz", allow_pickle=False) as saved:
            e0 = torch.from_numpy(saved["e0"].copy())
            padding = torch.from_numpy(saved["padding"].copy())
            ple = text.ple(e0)
            hidden = saved["hidden"].copy()
        for index, layer in enumerate(text.layers):
            x = e0 if index == 0 else torch.from_numpy(hidden[index - 1].copy())
            buffers = buffers_for(text, index, x, ple[:, :, index, :].contiguous(), padding, size)
            block = make_block(layer)
            stabilize(block)
            actual = []
            for start in range(0, size, TILE):
                inputs = tile_inputs(buffers, start)
                args = tuple(torch.from_numpy(inputs[name]) for name in NAMES)
                actual.append(block(*args).numpy().copy())
            check = metrics(np.concatenate(actual, axis=1), hidden[index])
            if not check["passed"] or check["max_abs"] > 1e-4:
                raise RuntimeError(f"Stable CPU block parity failed at layer {index}: {check}")
            args = tuple(torch.from_numpy(tile_inputs(buffers, 0)[name]) for name in NAMES)
            traced = torch.jit.trace(block, args, check_trace=False)
            converted = ct.convert(traced, convert_to="mlprogram",
                inputs=[ct.TensorType(name=name, shape=arg.shape,
                                      dtype=np.int32 if name == "allowed" else np.float32)
                        for name, arg in zip(NAMES, args)],
                outputs=[ct.TensorType(name="hidden", dtype=np.float32)],
                minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
            converted.short_description = f"Experimental original trained layer{index} native{size} Q256 reuse"
            converted.save(str(output / f"layer_{index:02d}.mlpackage"))
            entry = {"index": index, "type": text.config.layer_types[index], "torch_check": check,
                     "input_shapes": {name: list(arg.shape) for name, arg in zip(NAMES, args)}, "skip_model_load": True}
            report["layers"].append(entry)
            print(json.dumps({"phase": "export", **entry}), flush=True)
            del converted, traced, block, buffers, actual, args, inputs
            gc.collect()
    report["timing"] = {"load_seconds": load_seconds, "inference_seconds": inference_seconds,
                        "wall_seconds": time.perf_counter() - wall_start, "process_peak_rss_bytes": peak_rss_bytes()}
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Export finished. Run --validate in a NEW process; no hardware parity claimed.", flush=True)


def validate(output):
    wall_start = time.perf_counter()
    # Claim a fresh validation report before loading models; never overwrite a run.
    report_path = output / "validation.json"
    with report_path.open("x", encoding="utf-8"):
        pass
    import torch
    import coremltools as ct

    report = json.loads((output / "export.json").read_text())
    size = report["native_size"]  # Saved export scope, never the CLI default.
    if (size not in (512, 1024) or report["query_tile"] != TILE
            or len(report["layers"]) != LAYER_COUNT or not 1 <= len(report["texts"]) <= 4):
        raise ValueError("Incomplete or incompatible encoder reuse export")
    load_start = time.perf_counter()
    model = load_original(Path(report["source_upstream"]), size, output / "original_checkpoint.pt")
    text = model.encoder.language_model  # Fresh ORIGINAL norms, never stabilized here.
    # Load each distinct trained block ONCE, outside all text/layer/tile loops.
    blocks = [ct.models.MLModel(str(output / f"layer_{i:02d}.mlpackage"),
                               compute_units=ct.ComputeUnit.CPU_AND_NE) for i in range(LAYER_COUNT)]
    load_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    result = {"native_size": size, "query_tile": TILE, "quality_approved": False,
              "ane_execution_verified": False, "semantic_pass_claimed": False,
              "query_starts": list(range(0, size, TILE)), "texts": [], "passed": True}
    embeddings = []
    # Stable host arrays retained for each loaded model; copies, not zero-copy inputs.
    host_inputs = [None] * LAYER_COUNT
    result["host_buffers_stable"] = True
    with torch.inference_mode():
        for case, metadata in enumerate(report["texts"]):
            with np.load(output / f"text_{case}_gold.npz", allow_pickle=False) as saved:
                if (saved["input_ids"].shape != (1, size) or saved["padding"].shape != (1, size)
                        or saved["e0"].shape != (1, size, text.config.hidden_size)
                        or saved["hidden"].shape != (LAYER_COUNT, 1, size, text.config.hidden_size)):
                    raise ValueError("Saved fixture dimensions differ from export size")
                x = torch.from_numpy(saved["e0"].copy())
                padding = torch.from_numpy(saved["padding"].copy())
                gold_hidden, gold_embedding = saved["hidden"].copy(), saved["expected"].copy()
            ple = text.ple(x)  # Original whole PLE(E0), once, never from mixed hidden.
            taps = []
            for index, block in enumerate(blocks):
                buffers = buffers_for(text, index, x, ple[:, :, index, :].contiguous(), padding, size)
                next_x = np.empty_like(buffers["residual"])
                for start in range(0, size, TILE):
                    inputs = tile_inputs(buffers, start)
                    if host_inputs[index] is None:
                        host_inputs[index] = inputs
                    else:
                        for name in NAMES:
                            np.copyto(host_inputs[index][name], inputs[name])
                    predicted = np.asarray(block.predict(host_inputs[index])["hidden"], dtype=np.float32)
                    if predicted.shape != (1, TILE, text.config.hidden_size):
                        raise ValueError(f"Bad hidden shape at layer {index}: {predicted.shape}")
                    next_x[:, start:start + TILE] = predicted
                check = metrics(next_x, gold_hidden[index])
                taps.append({"index": index, **check})
                print(json.dumps({"phase": "validate", "text": case, "layer": index, **check}), flush=True)
                if not check["finite"]:
                    raise RuntimeError(f"Nonfinite full-encoder tap at text {case}, layer {index}")
                x = torch.from_numpy(next_x)  # Swap only AFTER every tile consumed immutable x.
            embedding = finish_embedding(model, x, padding, metadata["prompt_length"]).numpy().copy()
            check = metrics(embedding, gold_embedding)
            result["texts"].append({"index": case, "layers": taps, "final_embedding": check})
            result["passed"] = result["passed"] and check["passed"]
            embeddings.append(embedding)
            with (output / f"text_{case}_actual.npy").open("xb") as actual_file:
                np.save(actual_file, embedding)
        if report["head_enabled"]:
            probabilities = head_probabilities(model, embeddings, report["head_kind"])
            gold = np.load(output / "head_gold.npy", allow_pickle=False)
            finite = bool(np.isfinite(probabilities).all() and np.isfinite(gold).all())
            diff = float(np.max(np.abs(probabilities - gold))) if finite else None
            agree = bool(finite and np.array_equal(probabilities.argmax(-1), gold.argmax(-1)))
            head = {"finite": finite, "max_abs_probability_diff": diff, "labels_agree": agree,
                    "original_probabilities": gold.tolist(), "actual_probabilities": probabilities.tolist(),
                    "passed": finite and diff <= PROBABILITY_MAX_ABS and agree}
            result["head"] = head
            result["passed"] = result["passed"] and head["passed"]
    result["timing"] = {"load_seconds": load_seconds,
                        "inference_seconds": time.perf_counter() - inference_start,
                        "wall_seconds": time.perf_counter() - wall_start, "process_peak_rss_bytes": peak_rss_bytes()}
    # Release native models while their persistent host arrays remain alive.
    del block, blocks
    gc.collect()
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Full encoder/head numerical parity gate failed; no quality approval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=Path("artifacts-8192/upstream"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--head", action="store_true", help="Export four rendered texts and fixed synthetic noul head gold")
    parser.add_argument("--size", type=int, choices=(512, 1024), default=512,
                        help="Export key width; validation derives size from saved export/fixtures")
    parser.add_argument("--long-state", action="store_true",
                        help="Append 16 fixed museum records after original read-only state (requires --size 1024)")
    args = parser.parse_args()
    if args.validate and (args.head or args.long_state):
        parser.error("--head and --long-state are export-only; validation uses the saved scope")
    if args.long_state and args.size != 1024:
        parser.error("--long-state requires --size 1024")
    if args.validate:
        validate(args.output.resolve())
    else:
        export(args.output.absolute(), args.upstream.resolve(), args.head, args.size, args.long_state)


if __name__ == "__main__":
    main()
