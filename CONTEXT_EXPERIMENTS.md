# Context beyond8192: experimental, not production support

Production remains validated8192. No upstream config, tokenizer, packages or service runner may be changed by experiments.

## Established limits

Pinned decision checkpoint revision09fc4497a95529ec66842caac291e21c62490fab has outer max_seq_length8192. Its encoder text config sets max_position_embeddings262144, default RoPE theta1000000 for full attention and10000 for sliding attention. Thus16k-128k is inside the configured positional ceiling, but outside the advertised embedding context. This is not evidence of trained quality beyond8192. Initial16k probe uses original positions/frequencies, no scaling.

The exporter constructor also transiently creates an int64 NxN distance tensor (2GiB at16k). The exporter materializes a square local mask and square attention scores/probabilities even for local layers.24 layers include four full-attention layers; local window distance512. Mini has16GiB unified RAM.

| Tokens | Single FP16 NxN tensor | Single FP32 NxN tensor | Quadratic work vs8192 |
|---|---:|---:|---:|
|8192|0.125GiB|0.25GiB|1x|
|16384|0.5GiB|1GiB|4x|
|32768|2GiB|4GiB|16x|
|65536|8GiB|16GiB|64x|
|131072|32GiB|64GiB|256x|

These are algebraic single-tensor sizes, not measured peak memory or latency. Multiple temporaries, heads and system compiler allocations increase memory. Exact tiled local/global attention can reduce materialized memory, but full-attention FLOPs remain quadratic. Independently encoding chunks and averaging embeddings is a different algorithm, not full-context support.

Primary sources: [Google model card](https://huggingface.co/google/embeddinggemma-2/raw/main/README.md) explicitly advertises8192; [Google config](https://huggingface.co/google/embeddinggemma-2/raw/main/config.json) has positional ceiling262144. The pinned decision config was independently inspected; a larger positional ceiling is not a claim of long-context training or quality.

## First16k reference-only result (2026-10-13)

FAILED independent synthetic quality gate: short safe/unsafe evidence2/2 correct and sequential original-head scoring matches original decide within1e-5 probability tolerance. At16k, safe beginning/middle/end3/3 correct; unsafe beginning/middle/end0/3, all predicted false (no review) despite unauthorized irreversible deletion of last backup. Long truth accuracy50%. Exact untruncated token-ID and mask checks passed; this is not a truncation failure. Reports/mini-m4-context-16k-quality.json records configurations and provenance. These six synthetic inputs do not establish general accuracy; however false-safe errors are a concrete reason NOT to promote16k. Native4k/8k controls completed: safe cases3/3 correct at each length, unsafe cases0/3 at each length. Thus this failure exists inside native context and cannot be attributed solely to16k extrapolation. Evidence dilution/mean pooling or head distribution shift are hypotheses, not established root cause. See reports/mini-m4-context-native-quality-control.json. Prior production parity establishes conversion correctness, not semantic safety. Production remains8192. A tiled encoder solves allocation, not this quality failure.

## Transport and calibration caveats

Current API body cap262144 bytes and aggregate rendered text cap524288 bytes are independent of token limits. Instructions/criterion strings have8192-character caps. Many64k/128k text requests will exceed the body limit even if the encoder can represent them. Do not weaken these guards automatically; a later explicitly budgeted API transport design must preserve denial-of-service bounds.

The original trained head uses mean-pooled state embeddings and fixed FP32 temperatures. A single important fact can be diluted by long filler. Parity against the reference does not measure task accuracy or calibration. The first probe includes paired truth labels; broader multilingual heldout validation is needed before production approval.

## Tiled pilot status

Standalone experimental_encoder.py changes query computation only: tiles256/512, all global keys, local absolute distance<=512, no square constructor mask, original absolute RoPE and PLE.27 independent tiny-model tests pass on mini, including GQA, window/seams, holes/empty padded tails, valid-output padding isolation. Constructor powers through131072 pass meta-memory allocation checks; this is not real128k inference. Real-weight FP32 comparison at exactly16384 tokens passed: cosine0.99999988,maxabs7.45e-8,tiled18.30s vs reference34.67s for one run; combined process peak8.50GiB, not isolated tiled peak. This repeated-token numerical probe does NOT measure quality or API performance. A native512 tiled CoreML FP16 package exported successfully in29.04s; native512 load/inference passed on CPU_AND_NE: cosine0.99998564,maxabs0.0007179,load36.91s,warm median34.66ms (three repetitions). CoreML conversion warned of constant-cast overflow; finite output and numerical gate passed on this single input only, not proof every mask/input is safe. See reports/mini-m4-tiled-512-coreml-parity.json. No16k CoreML package exported or production promoted. The pilot is NOT imported by production runtime or converter.

## Stage gates

1. Reference-only16k probe changes max_seq_length in memory, never on disk. Verify full rendered token consumption. Paired safe/unsafe states, varied fillers and beginning/middle/end evidence; compare independent truth labels and identical short-evidence baselines. Wrong short baselines mean the scenario is not a valid long-context success test.
2. Only after quality evidence, design/export memory-bounded CoreML attention with native-length parity and extended-reference parity. Measure resource peaks and per-text/API timing.
3. HTTP isolation, idle stability, rollback-preserving production gating. Synthetic probes are limited quality evidence, not a general benchmark. No automatic production cap bump based solely on parity.
4. Repeat16k→32k→64k→128k; report failed quality/resource gates honestly. Larger contexts may require adaptation training and different compute resources.
