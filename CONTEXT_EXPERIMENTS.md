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

## Native4k pooling diagnostic

Same contextual already-projected token outputs, original fixed head/question/options/temperature: unsafe full masked mean predicts false, evidence-span mean predicts true, standalone short true. Signed review logit margin full -0.195227, span+0.349894, short+0.110455. This supports an aggregation contribution for this pair, not a general causal verdict. Exact oracle character offsets identify evidence; span pooling is OOD and must NOT become production prediction or a quality approval. Native config unchanged. See diagnostic_pooling.py and reports/mini-m4-pooling-diagnostic.json.

## Matched-prefix filler control

Fixed facts and original full pooling/head/question/temperature, identical filler record sequence/count for safe and unsafe at each count0/1/4/8/16/32. Unsafe is correct evidence-only46 tokens (review logit margin+0.110455), but adding ONE museum record gives75 tokens and false-safe margin-0.165934; all tested nonzero counts remain false-safe. Safe stays correct. Thus this probe failure needs neither long context nor positional extrapolation. Filler sensitivity and aggregation contribution remain limited-pair observations; no general accuracy claim, no calibration or threshold tuning. Evidence lengths differ8tokens, so this is matched filler, not exactly length-matched inputs. See diagnostic_filler_curve.py and reports/mini-m4-matched-filler-curve.json.

## Frozen independent typed-decision benchmark

quality_cases.json was committed BEFORE any model inference (SHA256 e077a1d244985fac91d96529b9c1ce40d88ed2f5acd4e5863041210989c53933).12 factual synthetic cases in held-out domains,4 each choice/score/noul; no labels revised after results. Original reference short answers:4/12 correct (choice1/4,score1/4,noul2/4); identical added filler1record:5/12,16records:4/12. Sequential scoring agrees original decide on every short case, max probability discrepancy1.073e-6. Therefore failures are reproduced by original trained model, not explained by CoreML conversion or long-context extrapolation. Not a representative sample or estimate of general accuracy; it demonstrates unsupported reasoning/threshold tasks and prevents treating encoder parity as semantic safety. No retraining, threshold tuning or replacement algorithm introduced. See evaluate_quality.py and reports/mini-m4-independent-quality.json. Production runtime is unchanged but should not be relied on as sole safety authority for these tasks.

## Training scope and original structured rendering control

[Pinned upstream decision card](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision/raw/09fc4497a95529ec66842caac291e21c62490fab/README.md) explicitly warns unseen schemas less reliable, numerical states harder, and rendering part of model. Claimed76.4% dataset accuracy is upstream-reported, not independently reproduced here, and test-informed hyperparameter selection noted by author. Our held-out threshold tasks therefore demonstrate transfer limitations, not contradiction of in-domain upstream score.

Short control with unchanged original dictionary renderer/head and same evidence: explicit fact-derived trace_summary fields keeps safe/unsafe correct at0/1/16 irrelevant records (unsafe pTrue0.587/0.577/0.578); dictionary evidence-only still fails with one record (pTrue0.482). Added explicit fields repeat factual information rather than extracting it automatically; this is a diagnostic input distribution, not an algorithm repair or deployment quality approval. Raw-string failures and frozen independent benchmark remain in report unchanged. See reports/mini-m4-structured-filler-control.json. Fixed native structured control0/1/16/128records passed8/8 synthetic decisions; largest safe/unsafe3883/3875 tokens, unsafe review logit margin+0.270607. Short sequential-vs-original decides agree. Explicit summary fields persist factual signal on this pair; no general benchmark or raw long-trace repair established. Subsequent fixed266-record control also passed both at8079/8071 tokens (unsafe margin+0.206347), total10/10 including smaller counts. See diagnostic_structured_context.py, reports/mini-m4-structured-native-control.json and reports/mini-m4-structured-native8k-control.json. This does NOT supersede failed raw-string or independent unseen-schema quality gates.

## Bounded structured16k tiled FP32 pilot

Fixed original structured facts/counts0/266/538, unscaled absolute positions, original config8192 unchanged, original frozen head/question/options: all6 synthetic decisions correct. Native8k same-ID/mask reference parity maxabs1.062e-7. Long consumed16345safe/16337unsafe tokens,21.58/21.29s tiled state CPU2threads, cumulative process peak6.21GiB (includes8k reference). Unsafe review margin shrinks from+0.206345 at8k to+0.030178 at16k: very near boundary, not robust quality approval. Raw/unseen failures still apply. Source was copied, so report repository_revision null; actual source hashes bind code. See diagnostic_tiled_structured.py and reports/mini-m4-tiled-structured16k-quality.json. No32k progression or production approval based on this pair.

## Large-graph compiler feasibility and reuse pilot

16kQ512 trace contains362735 nodes; native load still incomplete after more than60minutes. A privileged stack sample shows active mutable-weight-offset analysis, dominated by producer-of-blob map construction. This is evidence of graph bookkeeping overhead, not deadlock or confirmed ANE execution. Production health remains unchanged. Load eventually finished in5690.35s; one embedding numerical gate passed with37.47s warmmedian. Native validator process exited and wrote report while SSH channel remained open; channel was stopped after report retrieval. Reference shape(1,768) and exact16384 mask independently checked after validator shape fix, so no broadcasting scenario here. Full decision/API/quality gates remain unresolved.

experiment_attention_reuse.py exports two small weight-free native512/Q256 attention kernels and compares FP32 with independent whole512 dense oracle. Export passed allvalid/paddingholes/sparse/allmasked; hardware CPU_AND_NE FP16 validation subsequently passed8/8 kernel/mask combinations, each package instance reused for starts0/256: worst nonempty-row cosine0.9999108, maxabs0.000887, masked rows exact zero. Combined load+allcase predictions40.60s local/77.05s global; these are NOT warm per-tile latency and ANE execution unverified. See reports/mini-m4-attention-reuse-512.json. Same compiled tile must be reused twice with fixed full K/V, then outputs concatenated before later layers/pooling. This pilot is NOT full encoder or original-head quality, does not verify RoPE/normalization or the local512 boundary. Future per-layer reusable host graph would preserve immutable full-layer snapshots and exact global softmax; no independently encoded chunk averaging.

## Reuse native1024 boundary and actual trained layers

Attention reuse1024Q256 passed10/10 CPU_AND_NE FP16 kernel/case combinations; one model instance per kind reused four querytiles. Independent one-hot-value boundary probe confirmed distance512 included/513excluded local, while global retains allkeys. Worst rowcos0.9991705,maxabs0.002471.

Single original trained local0/global5 post-QKV layer graphs (original full QKV/RoPE/PLE prepared FP32 once from immutable snapshot) exported and reused fourtiles at1024. Independent original eager layer.forward reference precedes stabilization; FP32 stable tiledmaxabs<=5.782e-6. CoreML minima rowcos local0.9999804/global0.9999730, maxabs0.026406/0.046809. This larger absolute hidden-state error must be assessed across ALLlayers/final head before fullencoder approval; cosine pass alone does NOT approve model accuracy. QKV/preparation remained CPUFP32, no all-CoreML/fullencoder/ANE-execution claim. See experiment_layer_reuse.py and reports/mini-m4-layer-reuse-1024.json; boundary report mini-m4-attention-reuse-1024.json. Production untouched.

## Transport and calibration caveats

Current API body cap262144 bytes and aggregate rendered text cap524288 bytes are independent of token limits. Instructions/criterion strings have8192-character caps. Many64k/128k text requests will exceed the body limit even if the encoder can represent them. Do not weaken these guards automatically; a later explicitly budgeted API transport design must preserve denial-of-service bounds.

The original trained head uses mean-pooled state embeddings and fixed FP32 temperatures. A single important fact can be diluted by long filler. Parity against the reference does not measure task accuracy or calibration. The first probe includes paired truth labels; broader multilingual heldout validation is needed before production approval.

## Tiled pilot status

Standalone experimental_encoder.py changes query computation only: tiles256/512, all global keys, local absolute distance<=512, no square constructor mask, original absolute RoPE and PLE.27 independent tiny-model tests pass on mini, including GQA, window/seams, holes/empty padded tails, valid-output padding isolation. Constructor powers through131072 pass meta-memory allocation checks; this is not real128k inference. Real-weight FP32 comparison at exactly16384 tokens passed: cosine0.99999988,maxabs7.45e-8,tiled18.30s vs reference34.67s for one run; combined process peak8.50GiB, not isolated tiled peak. This repeated-token numerical probe does NOT measure quality or API performance. A native512 tiled CoreML FP16 package exported successfully in29.04s; native512 load/inference passed on CPU_AND_NE: cosine0.99998564,maxabs0.0007179,load36.91s,warm median34.66ms (three repetitions). CoreML conversion warned of constant-cast overflow; finite output and numerical gate passed on this single input only, a later padding pilot also passed lengths32/256/257/512 (minimum cosine0.9999792) and an all-empty finite zero-output check; still not proof every mask/input is safe. See reports/mini-m4-tiled-512-mask-validation.json. See reports/mini-m4-tiled-512-coreml-parity.json. 16kQ512 isolated FP16 package subsequently exported successfully: conversion410.40s, trace362735 nodes, process-lifetime peak8.65GiB (includes FP32 checks and conversion). CPU_AND_NE full16384 repeated-token numerical inference later passed cosine0.99999565/maxabs0.001328. Cold load5690.35s (94m50s), warmmedian37.47s/text, validation process peak6.49GiB. No production promotion; this is one embedding-only repeated-token input, not original-head/semantic/API stability approval. See reports/mini-m4-tiled-16k-coreml-validation.json. See reports/mini-m4-tiled-16k-export.json. The pilot is NOT imported by production runtime or converter.

## Stage gates

1. Reference-only16k probe changes max_seq_length in memory, never on disk. Verify full rendered token consumption. Paired safe/unsafe states, varied fillers and beginning/middle/end evidence; compare independent truth labels and identical short-evidence baselines. Wrong short baselines mean the scenario is not a valid long-context success test.
2. Isolated numerical/compiler research may proceed while quality is unresolved, but packages must have no production manifest/integration and remain explicitly quality-unapproved. Production-candidate16k and later32k/64k/128k progression require positive quality evidence. Measure native/extended parity, resource peaks and per-text/API timing; none substitutes for quality.
3. HTTP isolation, idle stability, rollback-preserving production gating. Synthetic probes are limited quality evidence, not a general benchmark. No automatic production cap bump based solely on parity.
4. Repeat16k→32k→64k→128k; report failed quality/resource gates honestly. Larger contexts may require adaptation training and different compute resources.
