"""Actual-weight multi-bucket boundary, decision parity and warm latency checks."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from convert import reference
from runtime import DecisionRuntime, RequestError
from integrity import package_sha256

def exact_text(tokenizer,prompt,length,unit=" document"):
    # This unit contributes one token; prefix/suffix punctuation still checked.
    text=unit*length
    ids=tokenizer(prompt+text,add_special_tokens=True)["input_ids"]
    while len(ids)>length:
        text=text[:-len(unit)]
        ids=tokenizer(prompt+text,add_special_tokens=True)["input_ids"]
    while len(ids)<length:
        text+=" a"
        ids=tokenizer(prompt+text,add_special_tokens=True)["input_ids"]
    if len(ids)!=length: raise RuntimeError("Failed exact token boundary generation")
    return text

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path("artifacts")); args=ap.parse_args()
    torch.set_num_threads(4)
    ref=reference(args.artifacts/"upstream")
    tok=ref.get_tokenizer(); prompt=ref.config.prompt
    runtime=DecisionRuntime(args.artifacts)
    lengths=[32,512,513,1024,1025,1536,2048]
    results=[]; passed=True
    for n in lengths:
        text=exact_text(tok,prompt,n)
        with torch.inference_mode(): expected=ref.encode([text],batch_size=1).numpy()
        actual=runtime.encode([text]).numpy()
        times=[]
        for _ in range(3):
            start=time.perf_counter(); runtime.encode([text]); times.append(time.perf_counter()-start)
        cosine=float((expected*actual).sum()/(np.linalg.norm(expected)*np.linalg.norm(actual)))
        error=float(np.abs(expected-actual).max())
        ok=bool(np.isfinite(actual).all() and np.isfinite(cosine) and cosine>=.998)
        passed &= ok
        results.append({"tokens":n,"bucket":next(b for b in runtime.buckets if n<=b),"cosine":cosine,"max_abs_error":error,"warm_seconds_median":float(np.median(times)),"passed":ok})
    questions={"action":{"type":"choice","instructions":"What should happen next?","criteria":{"continue":"Proceed safely","review":"Ask a human","stop":"Stop unsafe actions"}},"risk":{"type":"score","instructions":"How risky is this?","criteria":["no risk","low risk","high risk","severe risk"]},"needs_review":{"type":"noul","instructions":"Does this need human review?"}}
    decisions=[]
    for n in [513,1025,2000]:
        # State rendering adds a short workflow prefix; stay below2048.
        state=exact_text(tok,prompt,n,unit=" review")+" Irreversible production deletion occurred without approval."
        gold=ref.decide(state,questions,workflow="audit")
        answer=runtime.decide(state,questions,workflow="audit")
        for qid,out in answer.items():
            original=gold[qid].to_dict(); diff=max(abs(out["probabilities"][k]-p) for k,p in original["probabilities"].items())
            ok=out["label"]==original["label"] and diff<=.03
            passed &= ok
            decisions.append({"state_base_tokens":n,"question":qid,"reference_label":original["label"],"coreml_label":out["label"],"max_probability_abs_error":diff,"passed":ok})
    try: runtime.encode([exact_text(tok,prompt,2049)])
    except RequestError: rejected=True
    else: rejected=False
    passed &= rejected
    report={"host":platform.node(),"platform":platform.platform(),"buckets":runtime.buckets,"max_tokens":runtime.max_tokens,"boundary_embeddings":results,"long_decisions":decisions,"overlength_2049_rejected":rejected,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"encoder_sha256":package_sha256(args.artifacts),"source_revision":runtime.manifest["source_revision"],"compute_units":"CPU_AND_NE","ane_execution_trace_verified":False,"passed":bool(passed)}
    print(json.dumps(report,indent=2),flush=True)
    (args.artifacts/"validation-long.json").write_text(json.dumps(report,indent=2)+"\n")
    if not passed: raise RuntimeError("Long-context gate failed")
if __name__=="__main__": main()
