"""Four isolated 16k Core ML decisions; synthetic diagnostic, never approval.

Explicit CLI: --package EXISTING.mlpackage --upstream LOCAL_CHECKPOINT --export-report EXPORT.json --output NEW.json
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


def file_sha256(path):
    """Bounded-memory hashing; never materialize checkpoint weights in Python."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_identity(package, checkpoint, export_path):
    """Bind this pair to the original export's path-sensitive provenance schema."""
    evidence = json.loads(export_path.read_text())
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
            or evidence.get("experiment") != "tiled-coreml-compiler-resource"
            or evidence.get("size") != SIZE):
        raise ValueError("Expected original 16384 tiled export report")
    expected_package = evidence["package_provenance"]
    records = []
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file():
            record = {"path": path.relative_to(package).as_posix(),
                      "resolved_path": str(path.resolve()),
                      "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            records.append(record)
            digest.update(json.dumps(record, sort_keys=True).encode() + b"\0")
    if (not records or records != expected_package["files"]
            or digest.hexdigest() != expected_package["tree_sha256"]
            or str(package) != expected_package["root"]):
        raise ValueError("Package content/path differs from original export provenance")
    expected_files = {record["path"]: record for record in evidence["upstream_provenance"]["files"]}
    names = ("model.safetensors", "config.json", "modeling_gemmadecision.py",
             "configuration_gemmadecision.py", "tokenizer.json", "tokenizer_config.json")
    actual = {}
    for name in names:
        path = checkpoint / name
        actual[name] = file_sha256(path)
        if actual[name] != expected_files[name]["sha256"]:
            raise ValueError(f"Upstream {name} differs from original export provenance")
    config = json.loads((checkpoint / "config.json").read_text())
    if (config.get("pooling_include_prompt") is not True
            or config.get("projections") != [] or config.get("embed_dim") != 768
            or config.get("max_seq_length") != 8192):
        raise ValueError("Checkpoint violates original prompt-pooling/projection/768/native8192 ABI")
    return {"verified_against_export_report": True, "export_report": str(export_path),
            "export_report_sha256": file_sha256(export_path),
            "package_tree_sha256": digest.hexdigest(), "upstream_file_sha256": actual,
            "head_temperatures_identity": "Bound by upstream model.safetensors content hash",
            "export_declared_source": evidence.get("source"),
            "export_declared_source_revision": evidence.get("source_revision"),
            "warning": "Export-relative identity, not independent Hugging Face revision authentication; files must remain immutable during execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
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
        identity = verified_identity(package, checkpoint, args.export_report.resolve())
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(f"Artifact identity validation failed: {exc}")

    import importlib.util
    def code_path(name):
        return Path(__file__) if name == "experiment_coreml_decisions.py" else Path(importlib.util.find_spec(name[:-3]).origin)
    start = time.perf_counter()
    report = {
        "schema_version": 1, "experiment": "isolated-coreml-16k-original-decisions",
        "scope": "Four fixed synthetic cases; not a representative quality benchmark",
        "quality_approved": False, "production_approved": False,
        "production_changed": False, "quality_gate": None,
        "package": str(package), "upstream": str(checkpoint),
        "artifact_identity": identity,
        "compute_units": "CPU_AND_NE", "torch_threads": 2,
        "dense_long_reference_inference": False, "workflow": WORKFLOW,
        "questions": QUESTIONS, "cases": [], "status": "incomplete",
        "decision_algorithm": "Original renderer/head/temperature/noul argmax; CoreML state encoder; cached original short embed",
        "source_sha256": {
            name: hashlib.sha256(code_path(name).read_bytes()).hexdigest()
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

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Retain the exclusively claimed handle; all following work is under finally.
        stream = args.output.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error(f"Cannot exclusively claim output: {exc}")
    try:
        save("before_reference_load")
        import numpy as np
        import torch
        import coremltools as ct
        from convert import REVISION, SOURCE, reference

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        load_start = time.perf_counter()
        ref = reference(checkpoint).cpu().float().eval()
        report.update(declared_source=SOURCE, declared_source_revision=REVISION,
                      reference_load_seconds=time.perf_counter() - load_start)
        upstream = sys.modules[type(ref).__module__]
        tokenizer = ref.get_tokenizer()
        config = ref.config.to_dict()
        report.update(original_config=config, native_limit=ref.config.max_seq_length)
        if (ref.config.max_seq_length != 8192 or ref.config.pooling_include_prompt is not True
                or ref.config.projections != [] or ref.config.embed_dim != 768):
            raise RuntimeError("Expected unchanged native8192/prompt-pooling/no-projections/768 ABI")
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
