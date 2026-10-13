"""Standalone compiler-resource experiment; never a production/quality approval.

Example (run only with an explicit resource budget):
  uv run python experiment_export.py --artifacts artifacts-8192 --size 16384 \
      --query-tile 512 --output /tmp/tiled-16384-experiment

No manifest, tokenizer, decision head, service integration or production assets are
written. The repeated-token probe measures numerical parity, NOT decision quality.
The known synthetic quality failures remain unapproved. Core ML load/inference and
ANE placement are deliberately not attempted by this staged compiler measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

SIZES = (512, 1024, 2048, 4096, 8192, 16384)


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_provenance(root):
    """Bind relative filenames, resolved paths and content, including weight shards."""
    records = []
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            record = {"path": path.relative_to(root).as_posix(),
                      "resolved_path": str(path.resolve()),
                      "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            records.append(record)
            digest.update(json.dumps(record, sort_keys=True).encode() + b"\0")
    if not records:
        raise ValueError(f"Empty source tree: {root}")
    return {"root": str(root), "tree_sha256": digest.hexdigest(), "files": records}


def graph_estimates(config, size, tile):
    """Algebraic estimates, not measured compiler memory or whole-graph op counts."""
    types = tuple(config.layer_types)
    heads = config.num_attention_heads
    tiles = math.ceil(size / tile)
    local_elements = sum((min(size, s + tile) - s) *
                         (min(size, s + tile + 512) - max(0, s - 512))
                         for s in range(0, size, tile))
    full_layers = sum(t != "sliding_attention" for t in types)
    local_layers = len(types) - full_layers
    head_tiles = len(types) * heads * tiles
    return {"layers": len(types), "full_layers": full_layers,
            "local_layers": local_layers, "query_heads": heads,
            "query_tiles_per_head": tiles, "unrolled_head_tiles": head_tiles,
            "attention_matmul_count_estimate": 2 * head_tiles,
            "softmax_count_estimate": head_tiles,
            "attention_score_elements_all_layers_heads": heads *
                (full_layers * size * size + local_layers * local_elements),
            "largest_single_score_tile_fp32_bytes": min(tile, size) * size * 4,
            "ple_projection_fp32_bytes": size * len(types) * 512 * 4,
            "notes": "Attention matmul/softmax counts exclude projections and other ops. "
                     "Tiling reduces intermediates, not quadratic global FLOPs. "
                     "PLE and unrolled compiler graph can still dominate memory."}


def graph_node_count(graph):
    def count(block):
        return sum(1 + sum(count(child) for child in node.blocks())
                   for node in block.nodes())
    return count(graph)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts-8192"))
    parser.add_argument("--size", type=int, choices=SIZES, default=16384)
    parser.add_argument("--query-tile", type=int, choices=(256, 512), default=256)
    parser.add_argument("--output", type=Path, required=True,
                        help="NEW isolated directory; its parent must already exist")
    args = parser.parse_args()
    source = args.artifacts.expanduser().resolve()
    upstream = (source / "upstream").resolve()
    requested_output = args.output.expanduser()
    output = requested_output.resolve()
    if requested_output.exists() or requested_output.is_symlink() or output.exists():
        parser.error("--output must be a NEW directory; existing paths are refused")
    if output.is_relative_to(source) or output.is_relative_to(upstream):
        parser.error("--output must be outside both the source tree and upstream real path")
    if not source.is_dir() or not upstream.is_dir():
        parser.error("--artifacts must contain an existing upstream directory")
    if not output.parent.is_dir():
        parser.error("--output parent must already exist")
    args.artifacts, args.upstream, args.output = source, upstream, output
    return args


def main():
    args = parse_args()
    # Claim the directory atomically before heavy imports. Never reuse a prior run.
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = {"schema_version": 1, "experiment": "tiled-coreml-compiler-resource",
              "status": "incomplete", "quality_approved": False,
              "production_changed": False, "production_approved": False,
              "scope": "Repeated-token numerical probe and export resources only. "
                       "Known synthetic decision-quality gate failed; do not promote.",
              "source_artifacts": str(args.artifacts), "upstream_real_path": str(args.upstream),
              "output": str(args.output), "size": args.size, "query_tile": args.query_tile,
              "encoder_batch_size": 1, "compute_units": "CPU_AND_NE",
              "compute_precision": "FLOAT16", "skip_model_load": True,
              "coreml_inference_validated": False, "ane_placement_verified": False,
              "hardware": platform.platform(), "python": platform.python_version(),
              "argv": sys.argv, "timings_seconds": {}, "stage_peak_rss_bytes": {},
              "peak_rss_scope": "Process-lifetime high-water RSS; stages are cumulative, "
                                "not isolated tiled/compile peaks or service RAM.",
              "warning": "No production-compatible manifest or runtime assets are emitted."}

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = peak_rss_bytes()
        (args.output / "experiment-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")

    def timed(name, operation):
        report["current_stage"] = name
        save()
        start = time.perf_counter()
        try:
            return operation()
        finally:
            report["timings_seconds"][name] = time.perf_counter() - start
            report["stage_peak_rss_bytes"][name] = peak_rss_bytes()
            save()

    if args.size >= 16384:
        print("WARNING: >=16k experimental build: large unrolled graph, quadratic global "
              "attention, unchanged PLE, potentially high compiler RSS. Quality is NOT "
              "approved; export is NOT a production rollout.", file=sys.stderr, flush=True)
    try:
        # --help and every CLI/path rejection above require only the standard library.
        import numpy as np
        import torch
        import torch.nn.functional as F
        import transformers
        from convert import REVISION, SOURCE, reference, stabilize
        from experimental_encoder import TiledEncoder
        from validate_long import exact_text

        torch.set_num_threads(4)
        report.update(source=SOURCE, source_revision=REVISION, torch=torch.__version__,
                      transformers=transformers.__version__, torch_threads=4)
        report["export_code_sha256"] = {
            name: file_sha256(Path(__file__).resolve().parent / name)
            for name in ("experiment_export.py", "experimental_encoder.py", "convert.py",
                         "validate_long.py", "CONTEXT_EXPERIMENTS.md")}
        report["upstream_provenance"] = timed("hash_upstream", lambda: tree_provenance(args.upstream))
        report["source_metadata_sha256"] = {
            name: file_sha256(args.artifacts / name)
            for name in ("manifest.json", "head.safetensors")
            if (args.artifacts / name).is_file()}
        ref = timed("load_fp32_reference", lambda: reference(args.upstream).cpu().float().eval())
        ref.requires_grad_(False)
        # Do not change max_seq_length, RoPE, config, tokenizer, head or temperatures.
        tokenizer = ref.get_tokenizer()
        prompt = ref.config.prompt or ""
        text = timed("exact_text", lambda: exact_text(tokenizer, prompt, args.size))
        batch = tokenizer(prompt + text, add_special_tokens=True, truncation=False,
                          padding=False, return_tensors="pt")
        ids = batch["input_ids"]
        mask = batch["attention_mask"].to(torch.int32)
        if ids.shape != (1, args.size) or mask.shape != ids.shape or not bool((mask == 1).all()):
            raise RuntimeError("Exact full-length untruncated input/mask assertion failed")
        report["input_proof"] = {"truncation": False, "actual_tokens": ids.shape[1],
            "attended_tokens": int(mask.sum()), "input_ids_sha256":
                hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "rendered_text_sha256": hashlib.sha256((prompt + text).encode()).hexdigest(),
            "generator": "validate_long.exact_text", "quality_benchmark": False}
        report["graph_estimates"] = graph_estimates(ref.encoder.language_model.config,
                                                  args.size, args.query_tile)
        print(json.dumps(report["graph_estimates"], indent=2), flush=True)

        def parity(actual):
            cosine = float(F.cosine_similarity(expected, actual).item())
            error = float((expected - actual).abs().max().item())
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            passed = finite and math.isfinite(cosine) and math.isfinite(error) and cosine >= .99999 and error <= 1e-4
            return {"cosine": cosine if math.isfinite(cosine) else None,
                    "max_abs_error": error if math.isfinite(error) else None,
                    "finite": finite, "passed": passed,
                    "minimum_cosine": .99999, "maximum_abs_error": 1e-4}

        # Sequential original embed -> tiled -> stabilized: never batch long references.
        with torch.no_grad():
            expected = timed("reference_embed_fp32", lambda: ref.embed(ids, mask)).detach()
            embeds = ref.get_input_embeddings()(ids).detach()
            wrapper = TiledEncoder(ref.encoder.language_model, args.size, args.query_tile).eval()
            wrapper.requires_grad_(False)
            actual = timed("tiled_fp32", lambda: wrapper(embeds, mask)).detach()
            report["fp32_parity_before_stabilize"] = parity(actual)
            if not report["fp32_parity_before_stabilize"]["passed"]:
                raise RuntimeError("FP32 tiled parity failed; refusing trace/export")
            stabilize(wrapper)
            wrapper.requires_grad_(False)
            stable = timed("stabilized_tiled_fp32", lambda: wrapper(embeds, mask)).detach()
            report["fp32_parity_after_stabilize"] = parity(stable)
            if not report["fp32_parity_after_stabilize"]["passed"]:
                raise RuntimeError("Stabilized FP32 parity failed; refusing trace/export")
            # Store direct Core ML inputs plus IDs; small gold outputs are not truth labels.
            np.savez_compressed(args.output / "numerical-probe.npz",
                input_ids=ids.numpy(), embeddings=embeds.numpy(), attention_mask=mask.numpy(),
                reference_embedding=expected.numpy(), tiled_embedding=actual.numpy(),
                stabilized_embedding=stable.numpy())
            del ref  # Wrapper keeps shared encoder weights; no second model is loaded.
            traced = timed("trace", lambda: torch.jit.trace(wrapper, (embeds, mask), check_trace=False))
        report["traced_graph_node_count"] = graph_node_count(traced.inlined_graph)
        import coremltools as ct
        report["coremltools"] = ct.__version__
        converted = timed("coreml_convert", lambda: ct.convert(traced,
            inputs=[ct.TensorType(name="embeddings", shape=embeds.shape, dtype=np.float32),
                    ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.int32)],
            outputs=[ct.TensorType(name="sentence_embedding", dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True))
        converted.short_description = "EXPERIMENTAL tiled encoder: quality NOT approved; NOT production."
        converted.license = "Apache-2.0"
        converted.user_defined_metadata["quality_approved"] = "false"
        converted.user_defined_metadata["production_changed"] = "false"
        package = args.output / f"ExperimentalTiled{args.size}Q{args.query_tile}.mlpackage"
        timed("save_package", lambda: converted.save(str(package)))
        report["package_provenance"] = timed("hash_package", lambda: tree_provenance(package))
        report["probe_sha256"] = file_sha256(args.output / "numerical-probe.npz")
        report["status"] = "exported_not_loaded_not_quality_approved"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        save()
    print("Experimental export saved. No Core ML inference, quality approval, "
          "production manifest or service changes.", flush=True)


if __name__ == "__main__":
    main()
