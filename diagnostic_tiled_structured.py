"""Isolated original-state structured FP32 CPU tiled 16k pilot; never approval.

Run explicitly: uv run python diagnostic_tiled_structured.py --output /tmp/tiled-structured.json
No summary extraction, stabilization, config overrides, export, or production changes.
Raw-state and unseen-schema failures remain unresolved even if these fixed cases pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from diagnostic_structured_context import structured_state
from experiment_context import (
    CASES, FILLER, QUESTIONS, WORKFLOW, checked_features,
    claim_report_output, ids_digest, peak_rss_bytes,
)

COUNTS = (0, 266, 538)
SIZES = (512, 8192, 16384)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parent
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    revision = git_result.stdout.strip() if git_result.returncode == 0 else None
    names = ("diagnostic_tiled_structured.py", "diagnostic_structured_context.py",
             "experimental_encoder.py", "experiment_context.py", "convert.py")
    import importlib.util
    def source_file(name):
        if name == "diagnostic_tiled_structured.py":return Path(__file__)
        return Path(importlib.util.find_spec(name[:-3]).origin)
    return {"repository_revision": revision,
            "source_sha256": {name: hashlib.sha256(source_file(name).read_bytes()).hexdigest()
                              for name in names},
            "representation": "structured_state(CASES, fixed count); original known facts unchanged",
            "input_generation_revision": "fixed-structured-original-state-tiled-v1"}


def direct_state_features(ref, tok, upstream, state, count):
    """The untruncated direct IDs themselves, not ref.tokenize, feed the pilot."""
    import torch

    rendered = upstream.render_state(WORKFLOW, state)
    encoded = tok((ref.config.prompt or "") + rendered, add_special_tokens=True,
                  truncation=False, padding=False)
    expected = encoded["input_ids"]
    valid_mask = encoded["attention_mask"]
    actual = len(expected)
    size = next((size for size in SIZES if size >= actual), None)
    if size is None or not actual:
        raise RuntimeError(f"State cannot fit pilot fixed shapes: {actual}")
    assert actual <= 16384
    if count == 0 and size != 512:
        raise RuntimeError("Short baseline must fit 512")
    if count == 266 and size != 8192:
        raise RuntimeError("Fixed native case must choose 8192")
    if count == 538 and not (16000 <= actual <= 16384 and size == 16384):
        raise RuntimeError(f"Fixed long case is not near 16k: {actual}")
    if valid_mask != [1] * actual:
        raise RuntimeError("Unexpected original valid-token mask")
    if tok.pad_token_id is None:
        raise RuntimeError("Original tokenizer needs a padding token")
    ids = torch.tensor([expected + [tok.pad_token_id] * (size - actual)], dtype=torch.long)
    mask = torch.tensor([valid_mask + [0] * (size - actual)], dtype=torch.long)
    consumed = ids[0][mask[0].bool()].tolist()
    proof = {
        "untruncated_tokens": actual, "consumed_tokens": len(consumed),
        "fixed_size": size, "input_shape": list(ids.shape),
        "attention_mask_sum": int(mask.sum()), "mask_binary": True,
        "full_ids_equal": consumed == expected,
        "untruncated_ids_sha256": ids_digest(expected),
        "consumed_ids_sha256": ids_digest(consumed),
        "padded_ids_sha256": ids_digest(ids.tolist()),
        "attention_mask_sha256": ids_digest(mask.tolist()),
        "padding_tokens": size - actual,
        "tokenization": "direct original tokenizer(prompt + original render_state), truncation=False",
        "rendered_state_sha256": sha_text(rendered),
    }
    if consumed != expected or int(mask.sum()) != actual:
        raise RuntimeError("Exact IDs/mask proof failed")
    return {"input_ids": ids, "attention_mask": mask}, proof


def original_head(ref, upstream, spec, state_embedding, question_options):
    import torch

    start = time.perf_counter()
    emb = torch.cat([state_embedding, *question_options]).to(ref.head.state_proj.weight.dtype)
    kind = torch.tensor([spec.kind_id], device=ref.device)
    mask = torch.ones((1, len(spec.options)), dtype=torch.bool, device=ref.device)
    logits = ref.head(emb[0:1], emb[1:2], emb[2:].unsqueeze(0), kind, mask).float()
    temperatures = ref.temperatures.float()[kind].unsqueeze(-1)
    probabilities = (logits / temperatures).masked_fill(~mask, -1e4).softmax(-1)
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise RuntimeError("Non-finite original head output")
    p = probabilities[0].cpu()
    j = int(p.argmax())
    label = spec.options[j]
    answer = upstream.TypedAnswer(spec.kind, label, label == "true", float(p[j]),
                                  dict(zip(spec.options, p.tolist())), None)
    return {"answer": answer.to_dict(), "logits": logits[0].cpu().tolist(),
            "review_logit_margin": float(logits[0, 1] - logits[0, 0]),
            "review_probability_margin": float(p[1] - p[0]),
            "head_seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts-8192"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    output = claim_report_output(parser, artifacts, args.output)

    # --help and exclusive output validation require no heavy imports or model load.
    import torch
    from convert import REVISION, SOURCE, reference
    from experimental_encoder import TiledEncoder

    torch.set_num_threads(2)
    start = time.perf_counter()
    report = {
        "schema_version": 1, "experiment": "isolated-structured-original-state-tiled-16k",
        "status": "incomplete", "quality_approved": False, "production_approved": False,
        "quality_gate": None, "production_changed": False,
        "scope": "Fixed synthetic structured known-fact pilot only; no raw/unseen-schema approval",
        "unresolved_failures": ["raw state", "unseen schema"],
        "summary_extraction": False, "replacement_production_rule": False,
        "stabilization": False, "config_override": False, "encoder_batch_size": 1,
        "device": "cpu", "dtype": "float32", "torch_threads": 2,
        "source": SOURCE, "source_revision": REVISION, "artifacts": str(artifacts),
        "workflow": WORKFLOW, "questions": QUESTIONS, "filler_counts": COUNTS,
        "filler_templates": FILLER, "filler_formula": "(i*7919+104729)%1000003",
        "decision_algorithm": "Original state IDs/input embeddings, tiled state encoder; original sequential question/options, head/temperature/noul argmax",
        "output_exclusively_claimed": True, "cases": [],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - start
        report["process_peak_rss_bytes"] = peak_rss_bytes()
        report["peak_rss_scope"] = "Process lifetime high-water RSS, including model; not isolated inference RAM"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        report["input_provenance"] = provenance()
        load_start = time.perf_counter()
        ref = reference(artifacts / "upstream")
        report["reference_load_seconds"] = time.perf_counter() - load_start
        tok = ref.get_tokenizer()
        up = sys.modules[type(ref).__module__]
        text_model = ref.encoder.language_model
        config = ref.config.to_dict()
        text_config = text_model.config.to_dict()
        report["original_config"] = config
        report["native_limit"] = ref.config.max_seq_length
        if ref.config.max_seq_length != 8192:
            raise RuntimeError("Original max_seq_length must remain 8192")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in ref.parameters()):
            raise RuntimeError("Requires original FP32 CPU weights")
        spec = up.parse_question("needs_review", QUESTIONS["needs_review"])
        if spec.kind != "noul" or tuple(spec.options) != ("false", "true"):
            raise RuntimeError("Expected original noul false/true options")
        question_options, proofs = [], {}
        bank_start = time.perf_counter()
        with torch.inference_mode():
            # Embed the original question and each original option sequentially once.
            texts = [up.render_question(spec),
                     *[up.render_option(spec, i) for i in range(len(spec.options))]]
            for name, text in zip(("question", "option_no", "option_yes"), texts):
                features, proof = checked_features(ref, tok, text)
                if features["input_ids"].shape[1] > 8192:
                    raise RuntimeError("Question/option exceeds native capacity")
                proof["padded_ids_sha256"] = ids_digest(features["input_ids"].tolist())
                proof["attention_mask_sha256"] = ids_digest(features["attention_mask"].tolist())
                embed_start = time.perf_counter()
                question_options.append(ref.embed(**features))
                proof["embed_seconds"] = time.perf_counter() - embed_start
                proofs[name] = proof
            report["question_option_token_proofs"] = proofs
            report["question_option_embed_seconds"] = time.perf_counter() - bank_start

            for count in COUNTS:
                for case in CASES:
                    row_start = time.perf_counter()
                    state = structured_state(case, count)
                    features, proof = direct_state_features(ref, tok, up, state, count)
                    row = {**case, "filler_records": count, "state": state,
                           "state_sha256": sha_text(json.dumps(state, sort_keys=True)),
                           "token_proof": proof, "quality_approved": False,
                           "production_approved": False}
                    report["cases"].append(row)
                    build_start = time.perf_counter()
                    wrapper = TiledEncoder(text_model, proof["fixed_size"])
                    row["wrapper_build_seconds"] = time.perf_counter() - build_start
                    sharing = (wrapper.layers is text_model.layers and wrapper.ple is text_model.ple
                               and wrapper.norm is text_model.norm
                               and wrapper.projection is text_model.embedding_projection)
                    row["original_modules_shared"] = sharing
                    if not sharing:
                        raise RuntimeError("Wrapper must share original weights/modules")
                    row["registered_buffer_bytes"] = sum(b.numel() * b.element_size()
                                                         for b in wrapper.buffers())
                    row["rope_table_bytes"] = sum(p.numel() * p.element_size()
                                                  for p in wrapper.rope.parameters())
                    embed_start = time.perf_counter()
                    embeddings = ref.get_input_embeddings()(features["input_ids"])
                    row["input_embedding_seconds"] = time.perf_counter() - embed_start
                    row["input_embedding_buffer_bytes"] = embeddings.numel() * embeddings.element_size()
                    row["input_ids_mask_buffer_bytes"] = sum(t.numel() * t.element_size()
                                                               for t in features.values())
                    row["buffer_scope"] = "Resident explicit inputs/tables only; excludes transient attention/PLE and shared model weights"
                    tiled_start = time.perf_counter()
                    actual = wrapper(embeddings, features["attention_mask"])
                    row["tiled_state_seconds"] = time.perf_counter() - tiled_start
                    if not torch.isfinite(actual).all():
                        raise RuntimeError("Non-finite tiled embedding")
                    if count != 266:
                        row.update(original_head(ref, up, spec, actual, question_options))
                    passed = True
                    if count == 0:
                        compare_start = time.perf_counter()
                        original = ref.decide(state, QUESTIONS, workflow=WORKFLOW,
                                              tokenizer=tok)["needs_review"].to_dict()
                        error = max(abs(row["answer"]["probabilities"][k] - original["probabilities"][k])
                                    for k in original["probabilities"])
                        passed = row["answer"]["label"] == original["label"] and error <= 1e-5
                        row["original_decide_equivalence"] = {
                            "answer": original, "max_probability_error": error,
                            "tolerance": 1e-5, "passed": passed,
                            "seconds": time.perf_counter() - compare_start,
                            "scope": "Required original decide baseline; its internal text bank is not the cached sequential bank"}
                    elif count == 266:
                        compare_start = time.perf_counter()
                        expected = ref.embed(**features)  # SAME exact padded IDs and mask.
                        cosine = float(torch.nn.functional.cosine_similarity(expected, actual).item())
                        error = float((expected - actual).abs().max())
                        passed = bool(torch.isfinite(expected).all()) and cosine >= .99999 and error <= 1e-4
                        row["native_embedding_parity"] = {
                            "cosine": cosine, "max_abs_error": error,
                            "minimum_cosine": .99999, "maximum_abs_error": 1e-4,
                            "same_ids_and_mask": True, "passed": passed,
                            "seconds": time.perf_counter() - compare_start,
                            "padded_ids_sha256": proof["padded_ids_sha256"],
                            "attention_mask_sha256": proof["attention_mask_sha256"]}
                        # Establish identical-input native parity before evaluating either head.
                        if not passed:
                            save()
                            raise RuntimeError("Native embedding parity failed; head/long cases skipped")
                        row.update(original_head(ref, up, spec, actual, question_options))
                        row["original_native_head"] = original_head(ref, up, spec, expected, question_options)
                        del expected
                    else:
                        row["long_scope"] = "Experimental tiled state only; no original long ref.embed or decide"
                        row["experimental_16k_exact_consumption"] = (
                            proof["full_ids_equal"] and 16000 <= proof["consumed_tokens"] <= 16384)
                    row["truth_correct"] = row["answer"]["label"] == case["gold_label"]
                    row["configuration_unchanged"] = (
                        ref.config.to_dict() == config and text_model.config.to_dict() == text_config
                        and ref.config.max_seq_length == 8192)
                    row["decision_seconds"] = time.perf_counter() - row_start
                    row["process_peak_rss_bytes"] = peak_rss_bytes()
                    save()
                    print(json.dumps({"id": case["id"], "count": count,
                                      "tokens": proof["consumed_tokens"], "label": row["answer"]["label"],
                                      "margin": row["review_logit_margin"], "approved": False}), flush=True)
                    del embeddings, actual, wrapper
                    if not row["configuration_unchanged"]:
                        raise RuntimeError("Original configuration mutated")
                    if not passed:
                        raise RuntimeError("Baseline/native parity failed; larger counts skipped")
        report["configuration_unchanged"] = all(row["configuration_unchanged"] for row in report["cases"])
        report["all_fixed_cases_correct"] = all(row["truth_correct"] for row in report["cases"])
        report["status"] = "diagnostic_complete_unapproved"
        save()
        return 0
    except Exception as exc:
        report["status"] = "error_unapproved"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
