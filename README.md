# EmbeddingGemma 2 typed decisions — Core ML adaptation

**Original model and training by [mrm8488 (Manuel Romero)](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision). Shoutout and full credit to the author.** Google created the base [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

We only adapt the existing fine-tuned checkpoint for fixed-shape **Core ML CPU + Apple Neural Engine** inference and add local serving. **No retraining, new dataset, new decision algorithm or claim of model authorship.** Original schema rendering, trained head and calibration temperatures are retained. The head runs in FP32 PyTorch on CPU; the heavy encoder runs in Core ML with GPU excluded.

Thanks to [FluidInference](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) for demonstrating fixed-shape inference and overflow-aware normalization. Their base encoder weights are **not** used under the fine-tuned head. Our exporter adapts the Hugging Face Transformers architecture; we did not locate or copy a Fluid exporter.

## Scope and limits

- Text only; typed choice / score / yes-no answers with probabilities, not generated text.
- Token capacity is set by the validated artifact manifest, per rendered text including task prefix and special tokens. The exporter supports fixed buckets **512 / 1024 / 2048 / 4096 / 8192**; the runtime selects the smallest fitting model independently for each state/question/option. **4096 and 8192 pass real-weight hardware parity; production serves8192 after candidate and production HTTP stability gates passed.** Original checkpoint limit is 8192. Longer inputs are rejected, never silently truncated or chunk-averaged.
- Local layers use bidirectional distance **<=512** (distance512 included,513 excluded); full attention stays global in layers5/11/17/23. Dense masks preserve semantics but do not provide sparse-attention speedups. Fixed models duplicate encoder weights, trading disk/memory for predictable shapes.
- FP16 Core ML encoder with max-abs RMSNorm/L2 rescaling and mean weights applied before reduction. Native PyTorch FP16 is unsafe; this separate graph adaptation requires parity tests.
- Runtime retains and reuses one input-buffer pair per bucket. coremltools9 wraps NumPy input arrays in Objective-C objects; on this macOS build delayed Core ML reset triggered a native `_PyObject_Free` SIGSEGV when short-lived input arrays lost their final reference. Persistent buffers avoid that deallocation path; `verify_idle_api.py` exercises mixed-bucket HTTP requests,20-second idle cleanup and PID stability. This is a targeted workaround, not an upstream library fix.
- CPU_AND_NE allows Neural Engine **and CPU fallback**. Compute-plan reports describe anticipated placement, not execution traces or wall-clock percentages.
- Approximate numeric parity; near-tied decisions can change. Not a substitute for hard safety rules or human authorization.
- Multilingual model, English-trained decision schemas. Validate Polish workflows separately.
- macOS 15+ Apple Silicon, Python 3.12. Stack pinned in uv.lock. coremltools warns this newer PyTorch is outside its tested-version range; real conversion tests are required.

## Reproduce

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --frozen
uv run python download.py
for size in 512 1024 2048; do uv run python convert.py --size "$size"; done
uv run python validate.py
uv run python validate_long.py
uv run python placement.py
uv run pytest -q
```

Checkpoint pinned to **09fc4497a95529ec66842caac291e21c62490fab**. Report hashes bind the selected Core ML encoder packages and bucket mapping only, not the entire head/tokenizer/table/custom-code pipeline; all artifacts must remain trusted local build products. Downloaded custom Python is executed locally; review it first. Artifacts are trusted local build products, not an untrusted-upload format. No latest-revision downloads at startup. Large weights/packages are excluded from Git.

Validation compares real fine-tuned PyTorch FP32 weights with Core ML encoder + CPU head, checks all three types, multilingual/repeated-token/exact-512 examples and overlength rejection. artifacts/validation.json is bound to the encoder package SHA256. This is a small parity suite, **not a task-quality or safety benchmark**.

## Run locally

```sh
export DECISIONS_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run python -m uvicorn server:app --host 127.0.0.1 --port 8005
```

POST /v1/decisions accepts {workflow, state, questions}; see examples.json. Authorization: Bearer $DECISIONS_API_KEY. GET /health returns readiness only. Missing key disables inference. Unauthorized requests return 401. Schema, body, token and finite-value checks are enforced; encoder calls are serialized. Keep loopback by default. Do not expose to the Internet without TLS, rate/admission limits and request deadlines.

## macOS login autostart

After successful validation of the exact package:

```sh
uv run python install_service.py --port 8005
uv run python smoke.py
```

Creates a per-user LaunchAgent and a random credential in a private user config directory, not Git or the plist. Refuses to overwrite an installation. Starts at user login, not before login. Trusted LAN deployment can pass --host 0.0.0.0 deliberately; the service does not supply TLS. Existing embeddings need not be modified or stopped.

## Upgrade an existing service safely

Build new packages into a separate artifact directory; keep existing packages unchanged for rollback. Run validate.py and validate_long.py with --artifacts pointing there. Optional placement.py describes anticipated device placement, not execution utilization; on the 4096 graph this inspection took over eight minutes in the system ANE compiler and was stopped. During the subsequent 8192 load, a privileged stack sample showed active ANE buffer/performance optimization; long compilation is not by itself proof of a deadlock. Both parity reports must bind to selected encoder package hashes; the long report must cover every served bucket, match max_tokens and prove rejection of max_tokens+1. Use `uv run python switch_artifacts.py artifacts-8192 --startup-timeout 1800` only after those gates pass. The switch retains the old run script and restores it on failure. Health checks match buckets/max_tokens and the loaded encoder hash. Port-aware smoke checks exercise bucket boundaries and inspect the overlength error owner; six mixed-bucket cycles include 20-second idle periods with the exact listening PID stable. Keep upstream config, head, tokenizer and token table unchanged after validation (the package hash binds encoder packages, not those assets).

## License and attribution

Apache-2.0; see LICENSE and NOTICE. Architecture adaptation based on Transformers, Copyright 2026 HuggingFace Team. Not endorsed by Google, mrm8488, Hugging Face or FluidInference. Preserve attribution in derived packages.

## Measurements

8192 real-weight validation (2026-10-13, Mac mini M4) passes all13 boundaries including4097/6144/8192, minimum cosine0.9997220. Six varied audit embeddings cover beginning/middle/end at4049 and8153 rendered tokens; reference consumed counts match every input. All33 long-state typed decisions match reference labels, maximum probability difference0.0114684 (1.1468 percentage points). Warm per-text medians:4096~925ms,8192~3.511s (three repetitions); these are encoder timings, not complete API latency. Combined FP32-reference+CoreML peak RSS10.56GiB, not service-only RAM.8193 is rejected. See reports/mini-m4-8192-long-validation.json and reports/mini-m4-8192-baseline-validation.json.37 behavioral tests pass. Configuration excludes GPU (CPU_AND_NE); no ANE execution utilization trace is claimed. Candidate port8006 and production port8005 each passed six mixed-bucket HTTP cycles with20-second idle periods and unchanged listener PID. Production health matches the validated encoder digest. The old2048 artifact directory and timestamped runner backup remain available; see reports/mini-m4-8192-candidate-stability.json and reports/mini-m4-8192-production-rollout.json. Cold CPU_AND_NE loading takes minutes because large graphs are compiled again across processes; allow30minutes startup and expect an outage during restart. This is not warm request latency.

4096 upgrade (2026-10-13, Mac mini M4): all ten embedding boundaries pass through4096, minimum cosine0.9998422; 12/12 long-state labels agree and max probability difference0.0068223 (0.6822 percentage points). Warm median per-text latency for bucket4096 is~924ms. Combined reference+CoreML validation peak RSS~8.30GiB; not service-only RAM. Evidence: reports/mini-m4-4096-validation.json. Thirty behavioral/construction tests pass. Enhanced varied-state and reference-no-truncation validation was added after this first report; the later8192 report adds that evidence for both4096 and8192.

Long-context adaptation on Mac mini M4: all seven embedding boundary cases pass (32/512/513/1024/1025/1536/2048), minimum cosine0.999897. All9 long-state question labels agree, maximum probability difference0.002599 (0.260 percentage points). Median warm per-text encode latency,3 repetitions: bucket512 ~29ms,1024 ~82ms,2048 ~265ms. These timings include tokenizer/table lookup/Core ML; not whole-decision request time. Combined validation process peak RSS ~6.05GiB includes the full PyTorch reference encoder and all Core ML models, **not service-only memory**. For **each bucket** the anticipated Core ML plan assigns6758 non-constant operations to Neural Engine and one input cast to CPU (not a hardware execution trace). First load/compile measurements were approximately28/38/56 seconds; startup is intentionally slower than warm inference. 24 unit/behavioral tests pass. The deployed service passed six HTTP boundary/idle cycles,20seconds each, with a stable PID after the persistent-buffer workaround. One observed service RSS snapshot was ~513MiB; this excludes memory in Core ML system services. Reports under reports/mini-m4-long-*.json and reports/mini-m4-idle-api.json.

Previous single-bucket baseline (retained for comparison):

Validated on Mac mini M4 (16 GB), macOS 27.0.1: 121 encoder texts including exact 512-token inputs, minimum cosine 0.9998648, mean cosine 0.9999855; 27/27 question labels agree; maximum probability difference 0.006463 (0.6463 percentage points). The anticipated compute plan assigns 6757 non-constant operations to Neural Engine and one cast to CPU; this is not measured runtime utilization. 21 behavioral/numerical tests pass. Target-device reports in reports/. Unit tests substitute only the hardware encoder; real-weight comparisons use actual Core ML. Placement is the anticipated Core ML plan, not execution profiling. No new-model accuracy claims.
