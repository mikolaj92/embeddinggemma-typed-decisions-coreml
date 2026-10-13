"""Fixed OOD synthetic structured-state diagnostic, not a quality approval.

Run explicitly: uv run python diagnostic_structured_context.py --output /tmp/structured.json
Native configuration only; no summary extraction, tuning, rewording, or algorithm change.
Known facts are explicitly represented, not extracted from a trace by a new mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from experiment_context import (
    CASES, FILLER, QUESTIONS, WORKFLOW, checked_features,
    claim_report_output, peak_rss_bytes,
)

COUNTS = (0, 1, 16, 128, 266)


def structured_state(case, count):
    """Fixed original evidence plus explicit known facts and a matched filler prefix."""
    unsafe = case["id"] == "unsafe"
    state = {
        "action_evidence": case["evidence"],
        "agent": {"autonomy": "unsupervised"},
        "trace_summary": {
            "constraint_violations": int(unsafe),
            "irreversible_actions": int(unsafe),
            "authorization_obtained": not unsafe,
            "last_recovery_copy_intact": not unsafe,
        },
    }
    if count:
        state["unrelated_records"] = "\n".join(
            FILLER[i % len(FILLER)].format(i=i, v=(i * 7919 + 104729) % 1000003)
            for i in range(count)
        )
    return state


def predict(ref, tokenizer, upstream, state):
    """Dictionary-capable equivalent of experiment_context.predict.

    Its state.encode() report hash assumes strings. Only that report hash differs;
    rendering, checked single-text embedding, head, calibration and answer do not.
    """
    import torch

    spec = upstream.parse_question("needs_review", QUESTIONS["needs_review"])
    rendered = upstream.render_state(WORKFLOW, state)
    texts = [rendered, upstream.render_question(spec),
             *[upstream.render_option(spec, i) for i in range(len(spec.options))]]
    proofs, embeddings = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for text in texts:
            features, proof = checked_features(ref, tokenizer, text)
            features = {k: v.to(ref.device) if torch.is_tensor(v) else v
                        for k, v in features.items()}
            embeddings.append(ref.embed(**features))
            proofs.append(proof)
        emb = torch.cat(embeddings).to(ref.head.state_proj.weight.dtype)
        kind = torch.tensor([spec.kind_id], device=ref.device)
        mask = torch.ones((1, len(spec.options)), dtype=torch.bool, device=ref.device)
        logits = ref.head(emb[0:1], emb[1:2], emb[2:].unsqueeze(0), kind, mask).float()
        temps = ref.temperatures.float()[kind].unsqueeze(-1)
        probabilities = (logits / temps).masked_fill(~mask, -1e4).softmax(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite original-head output")
        p = probabilities[0].cpu()
        j = int(p.argmax())
        label = spec.options[j]
        answer = upstream.TypedAnswer(spec.kind, label, label == "true", float(p[j]),
                                      dict(zip(spec.options, p.tolist())), None)
    return {
        "answer": answer.to_dict(), "logits": logits[0].cpu().tolist(),
        "rendered_state_tokens": proofs[0]["consumed_tokens"],
        "token_proofs": dict(zip(("state", "question", "option_no", "option_yes"), proofs)),
        "state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
        "rendered_state_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "decision_seconds": time.perf_counter() - start,
        "process_peak_rss_bytes": peak_rss_bytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts-8192"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    output = claim_report_output(parser, artifacts, args.output)

    # --help and output validation do not load torch or the reference model.
    import torch
    from convert import REVISION, SOURCE, reference

    torch.set_num_threads(4)
    start = time.perf_counter()
    report = {
        "schema_version": 1, "experiment": "fixed-structured-context-diagnostic",
        "scope": "OOD synthetic known-fact representation; not a representative benchmark or quality approval",
        "summary_extraction": False, "rewording_after_output": False,
        "quality_approved": False, "production_approved": False, "quality_gate": None,
        "production_changed": False, "replacement_production_rule": False,
        "decision_algorithm": "Original renderer, sequential original embed, original head/calibration/noul argmax",
        "encoder_batch_size": 1, "source": SOURCE, "source_revision": REVISION,
        "artifacts": str(artifacts), "workflow": WORKFLOW, "questions": QUESTIONS,
        "filler_counts": COUNTS, "filler_templates": FILLER,
        "filler_formula": "(i*7919+104729)%1000003", "cases": [], "status": "incomplete",
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - start
        report["process_peak_rss_bytes"] = peak_rss_bytes()
        report["peak_rss_scope"] = "Process lifetime high-water RSS including model, not isolated inference RAM"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        ref = reference(artifacts / "upstream")
        tokenizer = ref.get_tokenizer()
        upstream = sys.modules[type(ref).__module__]
        config = ref.config.to_dict()
        report["original_config"] = config
        report["native_limit"] = ref.config.max_seq_length
        if ref.config.max_seq_length != 8192:
            raise RuntimeError("Expected native limit exactly 8192; no overrides allowed")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in ref.parameters()):
            raise RuntimeError("Expected unchanged original FP32 CPU reference")
        for count in COUNTS:
            for case in CASES:
                state = structured_state(case, count)
                rendered = upstream.render_state(WORKFLOW, state)
                tokens = len(tokenizer((ref.config.prompt or "") + rendered,
                                       add_special_tokens=True, truncation=False)["input_ids"])
                if tokens > ref.config.max_seq_length:
                    raise RuntimeError(f"Fixed state exceeds native capacity: {tokens}")
                row = {**case, "filler_records": count, "state": state,
                       **predict(ref, tokenizer, upstream, state)}
                row["truth_correct"] = row["answer"]["label"] == case["gold_label"]
                row["review_logit_margin"] = row["logits"][1] - row["logits"][0]
                if count == 0:
                    with torch.inference_mode():
                        original = ref.decide(state, QUESTIONS, workflow=WORKFLOW,
                                              tokenizer=tokenizer)["needs_review"].to_dict()
                    error = max(abs(row["answer"]["probabilities"][k] - original["probabilities"][k])
                                for k in original["probabilities"])
                    equivalent = row["answer"]["label"] == original["label"] and error <= 1e-5
                    row["original_decide_equivalence"] = {
                        "answer": original, "max_probability_error": error,
                        "tolerance": 1e-5, "passed": equivalent,
                    }
                report["cases"].append(row)
                report["configuration_unchanged"] = ref.config.to_dict() == config
                save()
                print(json.dumps({"id": case["id"], "count": count, "tokens": tokens,
                                  "correct": row["truth_correct"], "margin": row["review_logit_margin"]}),
                      flush=True)
                if not report["configuration_unchanged"]:
                    raise RuntimeError("Original configuration changed")
                if count == 0 and not equivalent:
                    raise RuntimeError("Sequential/original decide equivalence failed; larger counts skipped")
        report["all_fixed_cases_correct"] = all(row["truth_correct"] for row in report["cases"])
        report["status"] = "diagnostic_complete"
        save()
        return 0
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
