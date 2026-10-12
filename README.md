# EmbeddingGemma 2 typed decisions — Core ML adaptation

**Original model and training by [mrm8488 (Manuel Romero)](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision). Shoutout and full credit to the author.** Google created the base [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

We only adapt the existing fine-tuned checkpoint for fixed-shape **Core ML CPU + Apple Neural Engine** inference and add local serving. **No retraining, new dataset, new decision algorithm or claim of model authorship.** Original schema rendering, trained head and calibration temperatures are retained. The head runs in FP32 PyTorch on CPU; the heavy encoder runs in Core ML with GPU excluded.

Thanks to [FluidInference](https://huggingface.co/FluidInference/embeddinggemma-2-coreml) for demonstrating fixed-shape inference and overflow-aware normalization. Their base encoder weights are **not** used under the fine-tuned head. Our exporter adapts the Hugging Face Transformers architecture; we did not locate or copy a Fluid exporter.

## Scope and limits

- Text only; typed choice / score / yes-no answers with probabilities, not generated text.
- **512 tokens per rendered text**, including task prefix and special tokens. Longer inputs are rejected, never silently truncated or chunk-averaged. Original model supports 8192; this adaptation does not.
- FP16 Core ML encoder with max-abs RMSNorm/L2 rescaling and mean weights applied before reduction. Native PyTorch FP16 is unsafe; this separate graph adaptation requires parity tests.
- CPU_AND_NE allows Neural Engine **and CPU fallback**. Compute-plan reports describe anticipated placement, not execution traces or wall-clock percentages.
- Approximate numeric parity; near-tied decisions can change. Not a substitute for hard safety rules or human authorization.
- Multilingual model, English-trained decision schemas. Validate Polish workflows separately.
- macOS 15+ Apple Silicon, Python 3.12. Stack pinned in uv.lock. coremltools warns this newer PyTorch is outside its tested-version range; real conversion tests are required.

## Reproduce

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync --frozen
uv run python download.py
uv run python convert.py
uv run python validate.py
uv run python placement.py
uv run pytest -q
```

Checkpoint pinned to **09fc4497a95529ec66842caac291e21c62490fab**. Downloaded custom Python is executed locally; review it first. Artifacts are trusted local build products, not an untrusted-upload format. No latest-revision downloads at startup. Large weights/packages are excluded from Git.

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

## License and attribution

Apache-2.0; see LICENSE and NOTICE. Architecture adaptation based on Transformers, Copyright 2026 HuggingFace Team. Not endorsed by Google, mrm8488, Hugging Face or FluidInference. Preserve attribution in derived packages.

## Measurements

Validated on Mac mini M4 (16 GB), macOS 27.0.1: 121 encoder texts including exact 512-token inputs, minimum cosine 0.9998648, mean cosine 0.9999855; 27/27 question labels agree; maximum probability difference 0.006463 (0.6463 percentage points). The anticipated compute plan assigns 6757 non-constant operations to Neural Engine and one cast to CPU; this is not measured runtime utilization. 21 behavioral/numerical tests pass. Target-device reports in reports/. Unit tests substitute only the hardware encoder; real-weight comparisons use actual Core ML. Placement is the anticipated Core ML plan, not execution profiling. No new-model accuracy claims.
