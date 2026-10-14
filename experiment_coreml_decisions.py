"""Four isolated 16k Core ML decisions; synthetic diagnostic, never approval.

Explicit CLI: --package EXISTING.mlpackage --upstream LOCAL_CHECKPOINT --output NEW.json
No export, dense long-reference inference, config override, or production change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from diagnostic_structured_context import structured_state
from diagnostic_tiled_structured import original_head
from experiment_context import (
    CASES, QUESTIONS, WORKFLOW, checked_features, fit_state, ids_digest, peak_rss_bytes,
)

SIZE = 16384


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package, checkpoint = args.package.resolve(), args.upstream.resolve()
    if not package.is_dir() or package.suffix != ".mlpackage":
        parser.error("Package must be an existing isolated .mlpackage")
    if not checkpoint.is_dir():
        parser.error("Upstream must be an existing local checkpoint directory")
    output = args.output.resolve()
    protected = [package, checkpoint]
    for path in (args.package.absolute(), args.upstream.absolute(), package, checkpoint):
        protected.extend(p.resolve() for p in path.parents if p.name.startswith("artifacts"))
    root = Path(__file__).resolve().parent
    protected.extend(p.resolve() for p in root.glob("artifacts*"))
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be NEW; existing files and symlinks are forbidden")
    if any(output.is_relative_to(p) for p in protected):
        parser.error("Output must be outside package, artifacts, and real upstream")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Retain exclusive ownership of this handle for all incremental updates.
        stream = args.output.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error(f"Cannot exclusively claim output: {exc}")

    start = time.perf_counter()
    report = {
        "schema_version": 1, "experiment": "isolated-coreml-16k-original-decisions",
        "scope": "Four fixed synthetic cases; not a representative quality benchmark",
        "quality_approved": False, "production_approved": False,
        "production_changed": False, "quality_gate": None,
        "package": str(package), "upstream": str(checkpoint),
        "compute_units": "CPU_AND_NE", "torch_threads": 2,
        "dense_long_reference_inference": False, "workflow": WORKFLOW,
        "questions": QUESTIONS, "cases": [], "status": "incomplete",
        "decision_algorithm": "Original renderer/head/temperature/noul argmax; CoreML state encoder; cached original short embed",
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("experiment_coreml_decisions.py", "experiment_context.py",
                         "diagnostic_structured_context.py", "diagnostic_tiled_structured.py",
                         "convert.py", "runtime.py")
        },
    }

    def save(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - start,
                      process_peak_rss_bytes=peak_rss_bytes())
        stream.seek(0)
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stream.truncate()
        stream.flush()

    save("before_reference_load")
    try:
        import numpy as np
        import torch
        import coremltools as ct
        from convert import REVISION, SOURCE, reference

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        load_start = time.perf_counter()
        ref = reference(checkpoint).cpu().float().eval()
        report.update(source=SOURCE, source_revision=REVISION,
                      reference_load_seconds=time.perf_counter() - load_start)
        upstream = sys.modules[type(ref).__module__]
        tokenizer = ref.get_tokenizer()
        config = ref.config.to_dict()
        report.update(original_config=config, native_limit=ref.config.max_seq_length)
        if ref.config.max_seq_length != 8192:
            raise RuntimeError("Expected unchanged native 8192 configuration")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in ref.parameters()):
            raise RuntimeError("Original reference must be FP32 CPU")
        spec = upstream.parse_question("needs_review", QUESTIONS["needs_review"])
        cached, proofs = [], {}
        save("caching_native_question_options")
        with torch.inference_mode():
            texts = [upstream.render_question(spec),
                     *[upstream.render_option(spec, i) for i in range(len(spec.options))]]
            for name, text in zip(("question", "option_no", "option_yes"), texts):
                features, proof = checked_features(ref, tokenizer, text)
                if proof["untruncated_tokens"] > ref.config.max_seq_length:
                    raise RuntimeError("Question/option exceeds native limit")
                tick = time.perf_counter()
                cached.append(ref.embed(**features))
                proofs[name] = {**proof, "rendered_sha256": sha(text),
                                "embed_seconds": time.perf_counter() - tick}
        report["cached_token_proofs"] = proofs
        # Own these same NumPy objects for the entire model lifetime (native reset queues).
        buffers = {"embeddings": np.empty((1, SIZE, 512), dtype=np.float32),
                   "attention_mask": np.zeros((1, SIZE), dtype=np.int32)}
        report["input_shape"] = list(buffers["embeddings"].shape)
        save("before_coreml_load")
        tick = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        report["coreml_load_seconds"] = time.perf_counter() - tick
        save("coreml_loaded")
        with torch.inference_mode():
            for representation in ("raw", "structured"):
                for case in CASES:
                    tick = time.perf_counter()
                    if representation == "raw":
                        state, count = fit_state(tokenizer, ref.config.prompt or "", upstream,
                                                 case["evidence"], "beginning", SIZE)
                    else:
                        count = 538
                        state = structured_state(case, count)
                    rendered = upstream.render_state(WORKFLOW, state)
                    encoded = tokenizer((ref.config.prompt or "") + rendered,
                                        truncation=False, padding=False, add_special_tokens=True)
                    ids = encoded["input_ids"]
                    actual = len(ids)
                    if not 0 < actual <= SIZE or encoded["attention_mask"] != [1] * actual:
                        raise RuntimeError(f"Invalid untruncated state IDs/mask: {actual}")
                    if tokenizer.pad_token_id is None:
                        raise RuntimeError("Original tokenizer must define pad_token_id")
                    padded = [ids + [tokenizer.pad_token_id] * (SIZE - actual)]
                    # Original input-embedding module includes its original scaling.
                    original_inputs = ref.get_input_embeddings()(torch.tensor(padded, dtype=torch.long))
                    if tuple(original_inputs.shape) != (1, SIZE, 512):
                        raise RuntimeError("Unexpected original input-embedding shape")
                    np.copyto(buffers["embeddings"], original_inputs.numpy())
                    del original_inputs
                    buffers["attention_mask"].fill(0)
                    buffers["attention_mask"][0, :actual] = 1
                    if not np.isfinite(buffers["embeddings"]).all():
                        raise RuntimeError("Non-finite original input embeddings")
                    proof = {
                        "untruncated_tokens": actual, "consumed_tokens": actual,
                        "fixed_size": SIZE, "input_shape": [1, SIZE],
                        "attention_mask_sum": int(buffers["attention_mask"].sum()),
                        "mask_binary": True, "full_ids_equal": True,
                        "untruncated_ids_sha256": ids_digest(ids),
                        "consumed_ids_sha256": ids_digest(ids),
                        "padded_ids_sha256": ids_digest(padded),
                        "attention_mask_sha256": ids_digest(buffers["attention_mask"].tolist()),
                        "padding_tokens": SIZE - actual, "rendered_state_sha256": sha(rendered),
                        "embeddings_float32_sha256": hashlib.sha256(buffers["embeddings"].tobytes()).hexdigest(),
                    }
                    report["pending_case"] = {"id": case["id"], "representation": representation,
                                              "token_proof": proof}
                    save("before_state_prediction")
                    predict_start = time.perf_counter()
                    embedding = np.asarray(model.predict(buffers)["sentence_embedding"], dtype=np.float32)
                    encoder_seconds = time.perf_counter() - predict_start
                    if embedding.shape != (1, 768) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) <= 0:
                        raise RuntimeError("Invalid CoreML state embedding")
                    row = {**case, "representation": representation, "filler_records": count,
                           "position": "beginning" if representation == "raw" else None,
                           "state_sha256": sha(state if isinstance(state, str) else json.dumps(state, sort_keys=True)),
                           "rendered_state_sha256": sha(rendered), "rendered_state_tokens": actual,
                           "token_proofs": {"state": proof, **proofs},
                           "encoder_seconds": encoder_seconds,
                           **original_head(ref, upstream, spec, torch.from_numpy(embedding.copy()), cached)}
                    row.update(truth_correct=row["answer"]["label"] == case["gold_label"],
                               decision_seconds=time.perf_counter() - tick,
                               process_peak_rss_bytes=peak_rss_bytes())
                    report["cases"].append(row)
                    report.pop("pending_case", None)
                    report["configuration_unchanged"] = ref.config.to_dict() == config
                    if not report["configuration_unchanged"]:
                        raise RuntimeError("Original configuration changed")
                    save("decision_complete")
                    print(json.dumps(row, allow_nan=False), flush=True)
        report["status"] = "diagnostic_complete"
        save("complete")
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        save("error")
        raise
    finally:
        stream.close()


if __name__ == "__main__":
    main()
