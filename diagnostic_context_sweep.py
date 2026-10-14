"""Isolated FP32 CPU diagnostic. Default: native512 verification ONLY.
No export, stabilization, config override, training, or production mutation.
Use --execute --tokens 16384 only after reading the native verification report.
Original Transformers/EmbeddingGemma2 architecture, Apache-2.0; see NOTICE.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from experiment_context import (CASES, QUESTIONS, WORKFLOW, fit_state,
    checked_features, claim_report_output, ids_digest, peak_rss_bytes)
from diagnostic_structured_context import structured_state
from diagnostic_tiled_structured import original_head

GIB = 1024 ** 3


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    # Filesystem-only checkpoint/package guard. No Core ML import/load.
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in files}


def make_encoder(text_model, size, stage):
    import torch
    import torch.nn.functional as F
    from experimental_encoder import TiledEncoder, rotate

    class BoundedEncoder(TiledEncoder):
        """Original modules; only memory layout/GEMM row batching change.

        E0 and each layer's input remain immutable until its attention completes.
        KV are complete; Q and all head outputs exist only for one query tile.
        Pointwise original operations and per-layer PLE are token-tiled.
        """
        def forward(self, embeddings, attention_mask):
            if embeddings.shape != (1, self.size, 512):
                raise ValueError("Requires original batch-one 512-wide embeddings")
            e0 = embeddings
            x = e0
            bias = (1 - attention_mask.float()).unsqueeze(1) * -10000.0
            projection = self.ple.per_layer_model_projection
            for i, layer in enumerate(self.layers):
                stage("layer_start", layer=i, size=self.size)
                a = layer.self_attn
                d = a.head_dim
                # Derive dimensions from actual layer weights, including global overrides.
                kw = a.k_proj.weight.shape[0]
                vw = a.v_proj.weight.shape[0]
                keys = x.new_empty((1, self.size, kw))
                values = x.new_empty((1, self.size, vw))
                cos = self.rope[self.types[i] + "_cos"]
                sin = self.rope[self.types[i] + "_sin"]
                for s in range(0, self.size, 512):
                    e = min(self.size, s + 512)
                    h = layer.input_layernorm(x[:, s:e])
                    kp = a.k_proj(h).split(d, dim=-1)
                    vp = a.v_proj(h).split(d, dim=-1)
                    for j, (k, v) in enumerate(zip(kp, vp)):
                        k = a.k_norm(k)
                        k = k * cos[s:e] + rotate(k) * sin[s:e]
                        keys[:, s:e, j*d:(j+1)*d] = k
                        values[:, s:e, j*d:(j+1)*d] = a.v_norm(v)
                    del h, kp, vp, k, v
                ks = keys.split(d, dim=-1)
                vs = values.split(d, dim=-1)
                attended = x.new_empty(x.shape)
                local = self.types[i] == "sliding_attention"
                for s in range(0, self.size, 256):
                    e = min(self.size, s + 256)
                    lo = max(0, s - 512) if local else 0
                    hi = min(self.size, e + 512) if local else self.size
                    h = layer.input_layernorm(x[:, s:e])
                    qs = a.q_proj(h).split(d, dim=-1)
                    layer_bias = bias[:, :, lo:hi]
                    allowed = attention_mask[:, None, lo:hi] != 0
                    if local:
                        distance = (self.positions[s:e, None] - self.positions[None, lo:hi]).abs()
                        window = distance <= 512
                        layer_bias = layer_bias + (~window).float().unsqueeze(0) * -10000.0
                        allowed = allowed & window.unsqueeze(0)
                    heads = []
                    for j, q in enumerate(qs):
                        q = a.q_norm(q)
                        q = q * cos[s:e] + rotate(q) * sin[s:e]
                        k = ks[j // a.num_key_value_groups][:, lo:hi]
                        v = vs[j // a.num_key_value_groups][:, lo:hi]
                        scores = q @ k.transpose(-1, -2) + layer_bias
                        scores = torch.where(allowed.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
                        probs = F.softmax(scores, dim=-1) * allowed.to(scores.dtype)
                        heads.append(probs @ v)
                        del scores, probs, q, k, v
                    attended[:, s:e] = a.o_proj(torch.cat(heads, dim=-1))
                    del h, qs, heads, allowed, layer_bias
                del keys, values, ks, vs
                out = x.new_empty(x.shape)
                rows = slice(i * 512, (i + 1) * 512)
                pbias = None if projection.bias is None else projection.bias[rows]
                for s in range(0, self.size, 512):
                    e = min(self.size, s + 512)
                    y = x[:, s:e] + layer.post_attention_layernorm(attended[:, s:e])
                    y = y + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(y)))
                    per = F.linear(e0[:, s:e], projection.weight[rows], pbias)
                    per = per * self.ple.per_layer_model_projection_scale
                    per = self.ple.per_layer_projection_norm(per)
                    out[:, s:e] = layer.ple_block(y, per) * layer.layer_scalar
                    del y, per
                x = out
                del out, attended
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"Non-finite hidden state at layer{i}")
                stage("layer_end", layer=i, size=self.size)
            # Preserve original pool reduction order rather than incremental sums.
            normalized = x.new_empty(x.shape)
            for s in range(0, self.size, 512):
                normalized[:, s:s+512] = self.norm(x[:, s:s+512])
            mask = attention_mask.float().unsqueeze(-1)
            pool_weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (normalized * pool_weights).sum(dim=1)
            out = self.projection(pooled)
            scaled = out / out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
            return scaled / torch.sqrt((scaled * scaled).sum(dim=-1, keepdim=True)).clamp(min=1e-4)

    return BoundedEncoder(text_model, size, query_tile=256).eval()


def direct_features(ref, tok, up, state, size):
    import torch
    rendered = up.render_state(WORKFLOW, state)
    encoded = tok((ref.config.prompt or "") + rendered, add_special_tokens=True,
                  truncation=False, padding=False)
    expected = encoded["input_ids"]
    actual = len(expected)
    if not 0 < actual <= size or encoded["attention_mask"] != [1] * actual:
        raise RuntimeError("Invalid untruncated IDs or capacity")
    if tok.pad_token_id is None:
        raise RuntimeError("Original padding token missing")
    ids = torch.tensor([expected + [tok.pad_token_id] * (size-actual)], dtype=torch.long)
    mask = torch.tensor([[1] * actual + [0] * (size-actual)], dtype=torch.long)
    consumed = ids[0][mask[0].bool()].tolist()
    if consumed != expected or int(mask.sum()) != actual:
        raise RuntimeError("Consumed IDs/mask mismatch")
    proof = dict(untruncated_tokens=actual, consumed_tokens=len(consumed), fixed_size=size,
                 full_ids_equal=True, attention_mask_sum=int(mask.sum()), mask_binary=True,
                 untruncated_ids_sha256=ids_digest(expected), consumed_ids_sha256=ids_digest(consumed),
                 padded_ids_sha256=ids_digest(ids.tolist()), attention_mask_sha256=ids_digest(mask.tolist()),
                 rendered_state_sha256=hashlib.sha256(rendered.encode()).hexdigest())
    return {"input_ids": ids, "attention_mask": mask}, proof


def structured_count(ref, tok, up, size):
    # Shared count chosen from BOTH fixed facts, no predictions or gold-label feedback.
    def lengths(count):
        return [len(tok((ref.config.prompt or "") + up.render_state(WORKFLOW, structured_state(c, count)),
                        truncation=False, add_special_tokens=True)["input_ids"]) for c in CASES]
    lo, hi = 0, 1
    while max(lengths(hi)) <= size:
        hi *= 2
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if max(lengths(middle)) <= size:
            lo = middle
        else:
            hi = middle - 1
    if not all(size - 128 <= n <= size for n in lengths(lo)):
        raise RuntimeError("Fixed structured pair cannot meet near-capacity proof")
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts-8192"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(16384, 32768, 65536, 131072), default=16384)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run one requested long level after native gates")
    mode.add_argument("--verify-only", action="store_true", help="Native512 verification only (default)")
    parser.add_argument("--representations", nargs="+", choices=("raw", "structured"), default=["raw", "structured"])
    parser.add_argument("--prior-report", type=Path, help="Completed preceding level report, mandatory above16k")
    parser.add_argument("--rss-budget-gib", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.rss_budget_gib <= 8:
        parser.error("RSS budget must be positive and at most8GiB")
    if args.execute and args.tokens > 16384:
        if args.prior_report is None:
            parser.error("Above16k requires --prior-report for the preceding level")
        prior = json.loads(args.prior_report.read_text())
        expected_cases = (6 if "raw" in args.representations else 0) + (2 if "structured" in args.representations else 0)
        if (prior.get("requested_tokens") != args.tokens // 2
                or prior.get("status") != "diagnostic_complete_unapproved"
                or not prior.get("artifact_inventory_unchanged")
                or not prior.get("configuration_unchanged")
                or not prior.get("native_gate_passed")
                or not prior.get("long_executed")
                or not prior.get("manifest_frozen_before_long_predictions")
                or prior.get("representations") != args.representations
                or len(prior.get("cases", [])) != expected_cases
                or len(prior.get("manifest", [])) != expected_cases
                or not prior.get("parity")
                or not all(p.get("passed") for p in prior["parity"])
                or prior.get("source_sha256") != file_hash(Path(__file__))
                or (args.tokens == 32768 and (
                    not any(p.get("scope") == "same_IDs_existing_tiled16k" for p in prior["parity"])
                    or not 0 <= prior.get("long_head_parity_max_probability_error", float("inf")) <= 1e-5))):
            parser.error("Prior level numeric/resource completion gate failed (quality may be negative)")
    artifacts = args.artifacts.resolve()
    output = claim_report_output(parser, artifacts, args.output)
    started = time.perf_counter()
    report = dict(status="incomplete", production_approved=False, quality_approved=False,
                  production_changed=False, config_override=False, stabilization=False,
                  device="cpu", dtype="float32", threads=2, requested_tokens=args.tokens,
                  long_executed=False, representations=args.representations, stages=[], cases=[], parity=[], manifest=[])
    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = peak_rss_bytes()
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def stage(name, **details):
        rss = peak_rss_bytes()
        report["stages"].append(dict(stage=name, peak_rss_bytes=rss, **details))
        if name == "layer_end":
            save()
        if rss > args.rss_budget_gib * GIB:
            raise RuntimeError("Process lifetime RSS high-water exceeded diagnostic budget")
    before = inventory(artifacts)
    try:
        import torch
        from convert import reference, SOURCE, REVISION
        from experimental_encoder import TiledEncoder
        torch.set_num_threads(2)
        report.update(source=SOURCE, source_revision=REVISION,
                      source_sha256=file_hash(Path(__file__)),
                      packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors")},
                      artifact_inventory_sha256=ids_digest(before))
        if args.execute and args.tokens > 16384:
            if prior.get("artifact_inventory_sha256") != report["artifact_inventory_sha256"]:
                raise RuntimeError("Prior report artifacts differ")
            report["prior_report_sha256"] = file_hash(args.prior_report)
        checkpoints = sorted((artifacts / "upstream").rglob("*.safetensors"))
        if not checkpoints:
            raise RuntimeError("Original checkpoint safetensors missing")
        report["checkpoint_data_sha256"] = {str(p.relative_to(artifacts)): file_hash(p) for p in checkpoints}
        if args.execute and args.tokens > 16384 and prior.get("checkpoint_data_sha256") != report["checkpoint_data_sha256"]:
            raise RuntimeError("Prior checkpoint data hashes differ")
        report["upstream_source_sha256"] = {str(p.relative_to(artifacts)): file_hash(p)
            for p in (artifacts / "upstream").rglob("*") if p.is_file() and p.suffix in (".json", ".py", ".model")}
        ref = reference(artifacts / "upstream")
        stage("reference_loaded")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in ref.parameters()):
            raise RuntimeError("Original FP32 CPU weights required")
        up = sys.modules[type(ref).__module__]
        tok = ref.get_tokenizer()
        text = ref.encoder.language_model
        config, text_config = ref.config.to_dict(), text.config.to_dict()
        if ref.config.max_seq_length != 8192:
            raise RuntimeError("Native limit must remain8192")
        if not getattr(ref.config, "pooling_include_prompt", False):
            raise RuntimeError("Requires original prompt-inclusive pooling")
        if getattr(ref.config, "projections", []) != []:
            raise RuntimeError("Requires original empty external projections")
        if text.config.embedding_dim != 768:
            raise RuntimeError("Requires original768-wide embedding ABI")
        for parameter in ref.parameters():
            parameter.requires_grad_(False)
        report["original_config"] = config
        if args.execute and args.tokens > 16384 and prior.get("original_config") != config:
            raise RuntimeError("Prior original configuration differs")
        spec = up.parse_question("needs_review", QUESTIONS["needs_review"])
        if spec.kind != "noul" or tuple(spec.options) != ("false", "true"):
            raise RuntimeError("Original false/true noul head required")
        def compare(expected, actual, scope):
            if expected.shape != (1, 768) or actual.shape != (1, 768):
                raise RuntimeError("Embedding ABI shape mismatch")
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                raise RuntimeError("Non-finite embedding parity input")
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            cosine = float(torch.nn.functional.cosine_similarity(expected, actual).item())
            error = float((expected-actual).abs().max())
            passed = finite and cosine >= .99999 and error <= 1e-4
            report["parity"].append(dict(scope=scope, finite=finite, cosine=cosine, max_abs_error=error, passed=passed))
            save()
            if not passed:
                raise RuntimeError("Embedding parity gate failed: " + scope)
        with torch.inference_mode():
            bank = []
            for rendered in [up.render_question(spec), *[up.render_option(spec, i) for i in range(2)]]:
                f, proof = checked_features(ref, tok, rendered)
                bank.append(ref.embed(**f))
                report.setdefault("question_option_proofs", []).append(proof)
            # Independent original native head/gold diagnostic, not parity-derived labels.
            for case in CASES:
                state = structured_state(case, 0)
                f, proof = direct_features(ref, tok, up, state, 512)
                e0 = ref.get_input_embeddings()(f["input_ids"])
                expected = ref.embed(**f)
                pilot = TiledEncoder(text, 512).eval()(e0, f["attention_mask"])
                actual = make_encoder(text, 512, stage)(e0, f["attention_mask"])
                compare(expected, actual, case["id"] + ":upstream_native512")
                compare(pilot, actual, case["id"] + ":existing_tiled512")
                answer = original_head(ref, up, spec, actual, bank)
                expected_answer = original_head(ref, up, spec, expected, bank)
                head_error = max(abs(answer["answer"]["probabilities"][k] - expected_answer["answer"]["probabilities"][k]) for k in spec.options)
                if head_error > 1e-5 or answer["answer"]["label"] != expected_answer["answer"]["label"]:
                    raise RuntimeError("Identical padded native head parity failed")
                native = ref.decide(state, QUESTIONS, workflow=WORKFLOW, tokenizer=tok)["needs_review"].to_dict()
                err = max(abs(answer["answer"]["probabilities"][k]-native["probabilities"][k]) for k in native["probabilities"])
                if err > 1e-5 or answer["answer"]["label"] != native["label"]:
                    raise RuntimeError("Original native decide/head parity failed")
                report.setdefault("native_controls", []).append(dict(id=case["id"], token_proof=proof,
                    answer=answer, original_answer=native, max_probability_error=err,
                    truth_correct=native["label"] == case["gold_label"]))
                del f, e0, expected, pilot, actual
            # Fixed original-tokenizer IDs, with holes spanning both local-window edges.
            seam_ids = tok("catalog museum weather library astronomy garden " * 256,
                           add_special_tokens=True, truncation=False)["input_ids"][:1024]
            if len(seam_ids) != 1024:
                raise RuntimeError("Native seam fixture is not1024tokens")
            seam_ids = torch.tensor([seam_ids], dtype=torch.long)
            seam_e0 = ref.get_input_embeddings()(seam_ids)
            for variant in ("full", "holes", "empty"):
                seam_mask = torch.ones_like(seam_ids)
                if variant == "holes":
                    seam_mask[:, 510:515] = 0
                    seam_mask[:, 767:770] = 0
                    seam_mask[:, 1000:] = 0
                elif variant == "empty":
                    seam_mask.zero_()
                bounded = make_encoder(text, 1024, stage)(seam_e0, seam_mask)
                if variant == "empty":
                    if bounded.shape != (1, 768) or not torch.isfinite(bounded).all() or torch.count_nonzero(bounded):
                        raise RuntimeError("All-empty mask must yield finite zero embedding")
                    report["parity"].append(dict(scope="empty1024_finite_zero", passed=True))
                else:
                    native = ref.embed(seam_ids, seam_mask)
                    compare(native, bounded, "upstream1024_seam_" + variant)
                    pilot = TiledEncoder(text, 1024).eval()(seam_e0, seam_mask)
                    compare(pilot, bounded, "existing1024_seam_" + variant)
                    del native, pilot
                del bounded, seam_mask
            del seam_ids, seam_e0
            report["native_gate_passed"] = True
            save()
            if args.execute:
                # Conservative layout forecast, not a measured or hard allocator bound.
                n = args.tokens
                activation_estimate = 12 * n * 512 * 4 + 256 * n * 4 * 6 + 64 * 1024**2
                report["memory_preflight"] = dict(heuristic_layout_estimate_bytes=activation_estimate,
                    process_peak_after_native=peak_rss_bytes(), safety_reserve_bytes=GIB,
                    projected_process_bytes=peak_rss_bytes()+activation_estimate+GIB,
                    budget_bytes=int(args.rss_budget_gib*GIB),
                    note="Conservative tensor-layout estimate; RSS sampled between layers, not hard peak enforcement")
                if peak_rss_bytes()+activation_estimate+GIB > args.rss_budget_gib*GIB:
                    raise RuntimeError("Memory preflight exceeds budget; long allocation refused")
                rows = []
                if "raw" in args.representations:
                    for case in CASES:
                        for position in ("beginning", "middle", "end"):
                            state, count = fit_state(tok, ref.config.prompt or "", up, case["evidence"], position, n)
                            rows.append((case, "raw", position, count, state))
                if "structured" in args.representations:
                    count = structured_count(ref, tok, up, n)
                    rows.extend((case, "structured", "original_schema", count, structured_state(case, count)) for case in CASES)
                for case, representation, position, count, state in rows:
                    f, proof = direct_features(ref, tok, up, state, n)
                    if not n-128 <= proof["consumed_tokens"] <= n:
                        raise RuntimeError("Frozen manifest near-capacity proof failed")
                    report["manifest"].append(dict(id=case["id"], representation=representation,
                        position=position, filler_records=count, token_proof=proof))
                    del f
                report["manifest_frozen_before_long_predictions"] = True
                save()
                report["long_executed"] = True
                for index, (case, representation, position, count, state) in enumerate(rows):
                    f, proof = direct_features(ref, tok, up, state, n)
                    if proof != report["manifest"][index]["token_proof"]:
                        raise RuntimeError("Manifest regeneration changed IDs")
                    e0 = ref.get_input_embeddings()(f["input_ids"])
                    encoder = make_encoder(text, n, stage)
                    t = time.perf_counter()
                    actual = encoder(e0, f["attention_mask"])
                    encoder_seconds = time.perf_counter() - t
                    if not torch.isfinite(actual).all():
                        raise RuntimeError("Non-finite bounded embedding")
                    # Mandatory16k same-input old-pilot rounding proof before larger levels.
                    if index == 0 and n == 16384:
                        del encoder
                        pilot = TiledEncoder(text, n).eval()(e0, f["attention_mask"])
                        compare(pilot, actual, "same_IDs_existing_tiled16k")
                        pilot_head = original_head(ref, up, spec, pilot, bank)
                        bounded_head = original_head(ref, up, spec, actual, bank)
                        head_error = max(abs(pilot_head["answer"]["probabilities"][k] - bounded_head["answer"]["probabilities"][k]) for k in spec.options)
                        if head_error > 1e-5 or pilot_head["answer"]["label"] != bounded_head["answer"]["label"]:
                            raise RuntimeError("16k original head rounding parity failed")
                        report["long_head_parity_max_probability_error"] = head_error
                        stage("existing_16k_parity_completed")
                        del pilot
                    elif index == 0:
                        report["preceding_level_report_verified"] = True
                    answer = original_head(ref, up, spec, actual, bank)
                    report["cases"].append(dict(**report["manifest"][index], gold_label=case["gold_label"],
                        **answer, truth_correct=answer["answer"]["label"] == case["gold_label"],
                        encoder_seconds=encoder_seconds, decision_and_parity_seconds=time.perf_counter()-t,
                        process_peak_rss_bytes=peak_rss_bytes()))
                    if index == len(rows) - 1:
                        repeat_encoder = make_encoder(text, n, stage)
                        repeated = repeat_encoder(e0, f["attention_mask"])
                        compare(actual, repeated, "last_case_repeat")
                        repeat_answer = original_head(ref, up, spec, repeated, bank)
                        repeat_error = max(abs(answer["answer"]["probabilities"][k] - repeat_answer["answer"]["probabilities"][k]) for k in spec.options)
                        if repeat_error > 1e-5 or answer["answer"]["label"] != repeat_answer["answer"]["label"]:
                            raise RuntimeError("Repeat head stability failed")
                        report["repeat_head_max_probability_error"] = repeat_error
                        del repeated, repeat_encoder
                    stage("case_end", index=index)
                    save()
                    del f, e0, actual
                    if n != 16384 or index != 0:
                        del encoder
            if ref.config.to_dict() != config or text.config.to_dict() != text_config:
                raise RuntimeError("Original configuration changed")
        report["configuration_unchanged"] = True
        report["quality_gate"] = dict(passed=bool(report["cases"]) and all(r["truth_correct"] for r in report["cases"]),
            scope="Fixed synthetic facts only; raw and structured reported separately; never production approval")
        report["status"] = "diagnostic_complete_unapproved" if args.execute else "native_verified_long_not_executed"
    except Exception as exc:
        report["status"] = "error_unapproved"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["artifact_inventory_unchanged"] = inventory(artifacts) == before
        if not report["artifact_inventory_unchanged"]:
            report["status"] = "artifact_guard_failed_unapproved"
        save()
    print(json.dumps(dict(output=str(output), status=report["status"], quality_gate=report.get("quality_gate"))))
    return 0 if report["artifact_inventory_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
