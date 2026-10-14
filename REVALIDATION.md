# Sequential16k–128k revalidation

Requested2026-10-14: revalidate every level16k→32k→64k→128k without replacing original trained decision head or changing production8192/rollback2048. Negative quality does not prohibit isolated next-level research after resource preflight; it DOES prohibit deployment. No64k/128k unsafe dense or current24model host-buffer schedule.

## Protocol

1. Bind actual checkpoint weights, model/config/tokenizer files and experimental package to export evidence; declared HuggingFace revision is not independently authenticated by local-file hashes.
2. Freeze raw safe/unsafe facts and structured fact-derived versions. Preserve independent gold, original renderer/question/options/head/temperatures. Do not use positive parity as a truth substitute.
3. Direct untruncated tokenizer prompt+rendered IDs, exact masks, absolute positions and original unscaled local/global RoPE; never change source/native8192 configuration.
4. Validate new bounded FP32 encoder first against independent upstream512/1024 and existing tiled16k, including real local boundary and padding. Per-layer PLE from original E0 and original projection rows, shared original RMS; whole-layer K/V immutable until tile barrier; global softmax spans allkeys.
5. At each requested size record actual tokens, hashes, finite output, original-head probability/label agreement when reference available, independent truth outcomes, memory/resource checks, latency and repeat stability. Lack of independent full-length reference must be labeled explicitly; no reference self-comparison approval.
6. Run sizes sequentially, never multiple long CPU probes together. Stop a level before unsafe memory allocations; record resource rejection rather than pretend inference passed. Preserve original gold and failures.
7. Keep CoreML hardware proof distinct from CPUFP32 research; CPU_AND_NE permits CPU fallback and is not ANE placement proof.

## State at protocol creation

| Size | Existing evidence | New revalidation | Deployment |
|---|---|---|---|
|16k|FP32 embedding parity; CoreML repeated-input cosine0.99999565, warm~37.5s, cold~94min; raw false-safe failure; structured pair narrowly correct FP32|Package+checkpoint files byte-equal export verified. New boundedFP32 full8case run completed: raw3/6, structured2/2, total5/8truth; all three raw unsafe false-safe. Same-ID oldtiled16k embedding maxabs0, finalcase repeat/head difference0. CPU2threads ~21.7–22.3s/state, peak4.03GiB includingoldpilotcomparison. ActualCoreML four-state probe remotely active|NOT approved|
|32k|Only algebraic resource/position bounds; no full model result|New boundedFP32 schedule validated native512/1024 and16k;First32k launch rejected before long inference because JSON serialization changes integer id2label keys into strings. Canonical JSON comparison fixed; second16k completed same5/8truth, allparitypassed, repeat0, peak4.17GiB.32k relaunched sequentially after corrected prior gate. No32k result yet|NOT approved|
|64k|Only algebraic resource/position bounds; no full model result|Pending earlier bounded scheduler validation+resourcepreflight|NOT approved|
|128k|Currentreuse persistentinput+working minimum>17.5GiB, unsafe on16GiB Mini|Needs separately validated boundedownership/masks/PLE; no hardware approval|NOT approved|

Current source and report details: CONTEXT_EXPERIMENTS.md. A512/1024 full-encoder reuse test is a prerequisite architectural control, not evidence for these larger token limits. Production remains8192; artifacts-long2048 and rollback runner remain preserved.
