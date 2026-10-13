"""OOD diagnostic only, never a production rule or quality gate.
Explicitly run on mini: uv run python diagnostic_pooling.py --output /tmp/pooling.json
Fixed native-4k beginning pair; native 8192 limit, head, calibration stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from experiment_context import CASES, FILLER, QUESTIONS, WORKFLOW, checked_features, claim_report_output, fit_state, ids_digest, peak_rss_bytes


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts-8192"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    if args.output.resolve().suffix.lower() != ".json":
        parser.error("Output must be a JSON path outside artifacts")
    output = claim_report_output(parser, artifacts, args.output)
    import torch
    import torch.nn.functional as F
    import transformers
    from convert import SOURCE, REVISION, reference
    torch.set_num_threads(4)
    start_time = time.perf_counter()
    report = {"schema_version": 1, "experiment": "same-contextual-tokens-pooling-diagnostic",
              "scope": "Fixed synthetic native4k beginning safe/unsafe pair, not a benchmark",
              "evidence_only_pooling_is_ood_diagnostic": True, "quality_gate": None,
              "production_changed": False, "production_approved": False,
              "replacement_production_rule": False, "native_limit": 8192,
              "rendered_token_budget": 4096, "position": "beginning",
              "source": SOURCE, "source_revision": REVISION, "artifacts": str(artifacts),
              "hardware": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "questions": QUESTIONS, "workflow": WORKFLOW, "filler_templates": FILLER,
              "filler_formula": "(i*7919+104729)%1000003", "cases": [], "status": "incomplete"}

    def save():
        report["elapsed_seconds"] = time.perf_counter() - start_time
        report["process_peak_rss_bytes"] = peak_rss_bytes()
        report["peak_rss_scope"] = "Process lifetime high-water RSS including model, not isolated inference RAM"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        ref = reference(artifacts / "upstream")
        upstream = sys.modules[type(ref).__module__]
        tokenizer = ref.get_tokenizer()
        config = ref.config.to_dict()
        if ref.config.max_seq_length != 8192:
            raise RuntimeError("Expected native limit exactly 8192; no configuration overrides allowed")
        if (ref.config.encoder_method != "forward" or list(ref.config.encoder_output) != ["last_hidden_state"]
                or ref.config.projections or len(ref.projections) or len(ref.activations)
                or ref.config.embed_dim != 768):
            raise RuntimeError("Expected forward/last_hidden_state projected768, no additional projections")
        if not tokenizer.is_fast:
            raise RuntimeError("Fast tokenizer offsets required")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in ref.parameters()):
            raise RuntimeError("Expected original FP32 CPU reference")
        report["original_config"] = config
        report["config_sha256"] = digest(json.dumps(config, sort_keys=True, default=str))
        paths = [Path(__file__), Path(__file__).with_name("experiment_context.py"),
                 Path(__file__).with_name("convert.py")]
        paths += [artifacts / "upstream" / name for name in (
            "config.json", "modeling_gemmadecision.py", "configuration_gemmadecision.py",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
        report["provenance_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in paths if p.is_file()}
        report["weight_files"] = [{"name": p.name, "bytes": p.stat().st_size}
                                  for p in sorted((artifacts / "upstream").glob("*.safetensors"))]
        report["weight_provenance_note"] = "Unchanged local checkpoint; weight file sizes recorded, not rehashed"
        text_config = ref.encoder.language_model.config
        report["attention_implementation"] = getattr(text_config, "_attn_implementation", None)
        report["positional_limit"] = text_config.max_position_embeddings
        spec = upstream.parse_question("needs_review", QUESTIONS["needs_review"])

        def device_features(features):
            return {k: v.to(ref.device) if torch.is_tensor(v) else v for k, v in features.items()}

        def original_embed(text):
            features, proof = checked_features(ref, tokenizer, text)
            return ref.embed(**device_features(features)), proof

        with torch.inference_mode():
            bank, bank_proofs = [], []
            for text in [upstream.render_question(spec),
                         *[upstream.render_option(spec, i) for i in range(len(spec.options))]]:
                vector, proof = original_embed(text)
                bank.append(vector)
                bank_proofs.append(proof)
            question, options = bank[0], torch.cat(bank[1:])
            report["fixed_question_options_token_proofs"] = bank_proofs
            report["question_options_embedding"] = "Sequential original ref.embed once each, fixed for all scores"
            kind = torch.tensor([spec.kind_id], device=ref.device)
            option_mask = torch.ones((1, len(spec.options)), dtype=torch.bool, device=ref.device)
            temperature = ref.temperatures.float()[kind].unsqueeze(-1)
            if not torch.isfinite(temperature).all() or not (temperature > 0).all():
                raise RuntimeError("Invalid original temperature")

            def score(vector, gold):
                dtype = ref.head.state_proj.weight.dtype
                logits = ref.head(vector.to(dtype), question.to(dtype), options.to(dtype).unsqueeze(0),
                                  kind, option_mask).float()
                calibrated = logits / temperature
                probs = calibrated.masked_fill(~option_mask, -1e4).softmax(-1)
                if not torch.isfinite(logits).all() or not torch.isfinite(probs).all():
                    raise RuntimeError("Nonfinite head output")
                ti, fi = spec.options.index("true"), spec.options.index("false")
                gi = spec.options.index(gold)
                label = spec.options[int(probs[0].argmax())]
                return {"options_order": list(spec.options), "logits": logits[0].cpu().tolist(),
                        "calibrated_logits": calibrated[0].cpu().tolist(),
                        "temperature": float(temperature.item()),
                        "probabilities": dict(zip(spec.options, probs[0].cpu().tolist())),
                        "label": label, "matches_synthetic_fact": label == gold,
                        "review_margin_true_minus_false": float((logits[0, ti] - logits[0, fi]).item()),
                        "calibrated_review_margin": float((calibrated[0, ti] - calibrated[0, fi]).item()),
                        "gold_margin": float((logits[0, gi] - logits[0, 1 - gi]).item())}

            vectors = {}
            for case in CASES:
                state, n = fit_state(tokenizer, ref.config.prompt or "", upstream,
                                     case["evidence"], "beginning", 4096)
                rendered = upstream.render_state(WORKFLOW, state)
                original = (ref.config.prompt or "") + rendered
                evidence = case["evidence"]
                if original.count(evidence) != 1:
                    raise RuntimeError("Evidence must occur exactly once in original prompted rendered state")
                a = original.index(evidence)
                b = a + len(evidence)
                enc = tokenizer(original, add_special_tokens=True, truncation=False,
                                return_offsets_mapping=True, return_special_tokens_mask=True)
                features, proof = checked_features(ref, tokenizer, rendered)
                positions = features["attention_mask"][0].bool().nonzero().flatten().tolist()
                consumed = features["input_ids"][0, positions].tolist()
                if consumed != enc["input_ids"] or len(enc["offset_mapping"]) != len(consumed):
                    raise RuntimeError("Offsets token IDs differ from ref.tokenize")
                selected = [i for i, ((left, right), special) in enumerate(zip(
                    enc["offset_mapping"], enc["special_tokens_mask"]))
                    if not special and right > left and left < b and right > a]
                covered = set()
                for i in selected:
                    left, right = enc["offset_mapping"][i]
                    covered.update(range(max(a, left), min(b, right)))
                if not selected or any(a + i not in covered for i, c in enumerate(evidence) if not c.isspace()):
                    raise RuntimeError("Offsets do not cover exact evidence non-whitespace characters")
                span_mask = torch.zeros_like(features["attention_mask"])
                span_mask[0, [positions[i] for i in selected]] = 1
                features = device_features(features)
                kwargs = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"],
                          "return_dict": True}
                if ref._encoder_kwargs is not None:
                    kwargs = {k: v for k, v in kwargs.items() if k in ref._encoder_kwargs}
                tokens = ref.encoder.forward(**kwargs).last_hidden_state
                if tokens.ndim != 3 or tokens.shape[-1] != 768 or not torch.isfinite(tokens).all():
                    raise RuntimeError("Expected finite already-projected contextual768 token outputs")
                full_mask = features["attention_mask"]
                if features.get("prompt_length") and not ref.config.pooling_include_prompt:
                    full_mask = upstream._exclude_prompt_from_mask(full_mask, int(features["prompt_length"]))
                span_mask = span_mask.to(ref.device) * full_mask

                def pool(mask):
                    if not mask.sum():
                        raise RuntimeError("Empty pooling mask")
                    m = mask.unsqueeze(-1).to(tokens.dtype)
                    mean = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
                    vector = F.normalize(mean[:, :ref.config.embed_dim].float(), dim=-1)
                    if not torch.isfinite(vector).all() or not vector.norm() > 0:
                        raise RuntimeError("Invalid pooled vector")
                    return vector

                full, span = pool(full_mask), pool(span_mask)
                short, short_proof = original_embed(upstream.render_state(WORKFLOW, evidence))
                variants = {"full_mean": full, "evidence_span_mean_ood": span, "standalone_short_original": short}
                vectors[case["id"]] = variants
                scores = {name: score(vector, case["gold_label"]) for name, vector in variants.items()}
                report["cases"].append({**case, "filler_records": n, "state_sha256": digest(state),
                    "token_span_proof": {**proof, "character_span": [a, b], "exact_evidence_text": original[a:b],
                        "original_prompted_text_sha256": digest(original), "offset_ids_equal_ref_tokenize": True,
                        "offset_ids_sha256": ids_digest(enc["input_ids"]), "no_truncation": True,
                        "valid_evidence_tokens": len(selected), "span_pool_valid_tokens": int(span_mask.sum()),
                        "selected_input_positions": [positions[i] for i in selected],
                        "selected_token_ids": [consumed[i] for i in selected],
                        "selected_offsets": [list(enc["offset_mapping"][i]) for i in selected],
                        "boundary_overlap_offsets": [list(enc["offset_mapping"][i]) for i in selected
                            if enc["offset_mapping"][i][0] < a or enc["offset_mapping"][i][1] > b],
                        "selection_rule": "Non-special attention-valid tokens overlapping exact evidence character span",
                        "all_non_whitespace_evidence_characters_covered": True},
                    "pooling_proof": {"same_contextual_token_outputs": True, "encoder_forward_calls": 1,
                        "token_shape": list(tokens.shape), "token_dtype": str(tokens.dtype),
                        "full_pool_valid_tokens": int(full_mask.sum()), "span_pool_valid_tokens": int(span_mask.sum()),
                        "formula": "Original token-dtype masked mean, float32 L2; no extra projection",
                        "vector_l2": {name: float(v.norm()) for name, v in variants.items()}},
                    "short_token_proof": short_proof, "scores": scores,
                    "full_span_cosine": float(F.cosine_similarity(full, span).item()),
                    "cosine_to_standalone_short": {name: float(F.cosine_similarity(v, short).item())
                                                   for name, v in variants.items()},
                    "span_minus_full_review_margin": scores["evidence_span_mean_ood"]["review_margin_true_minus_false"]
                        - scores["full_mean"]["review_margin_true_minus_false"]})
                del tokens
                save()
            report["safe_unsafe_separation"] = {
                name: {"cosine_similarity": float(F.cosine_similarity(vectors["safe"][name], vectors["unsafe"][name]).item()),
                       "cosine_distance": float(1 - F.cosine_similarity(vectors["safe"][name], vectors["unsafe"][name]).item())}
                for name in vectors["safe"]}
        unsafe = next(row for row in report["cases"] if row["id"] == "unsafe")["scores"]
        report["interpretation"] = {
            "unsafe_full_false_safe_observed": unsafe["full_mean"]["label"] == "false",
            "unsafe_span_matches_fact": unsafe["evidence_span_mean_ood"]["matches_synthetic_fact"],
            "unsafe_short_matches_fact": unsafe["standalone_short_original"]["matches_synthetic_fact"],
            "reading": "Full/span compares only aggregation on identical contextual outputs. Recovery is consistent with an aggregation contribution on this pair, not a general causal verdict. Non-recovery neither rules out dilution nor proves contextual encoding loss: the trained head sees OOD span vectors.",
            "limitations": "Oracle evidence boundaries; span removes trained prompt/state/filler pooling contributions. Evidence-only is OOD DIAGNOSTIC, never replacement production rule or quality gate. Short has different encoder context. Cosine separation alone does not prove decision quality. No 8k/16k inference or representative benchmark."}
        report["configuration_unchanged"] = ref.config.to_dict() == config
        if not report["configuration_unchanged"]:
            raise RuntimeError("Reference config unexpectedly changed")
        report["status"] = "diagnostic_complete"
        save()
        print(json.dumps({"output": str(output), "status": report["status"], "interpretation": report["interpretation"]}), flush=True)
        return 0  # Execution success only, not a quality pass/fail.
    except Exception as exc:
        report["status"] = "execution_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
