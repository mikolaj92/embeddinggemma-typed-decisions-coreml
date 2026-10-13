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

def checked_reference_tokens(ref, tokenizer, prompt, text, limit):
    """Compare untruncated IDs with the actual reference tokenize contract."""
    ids=tokenizer(prompt+text,add_special_tokens=True,truncation=False)["input_ids"]
    if len(ids)>limit: raise RuntimeError(f"Validation input exceeds rendered token limit: {len(ids)}>{limit}")
    features=ref.tokenize([text],tokenizer=tokenizer)
    consumed=features["input_ids"][0][features["attention_mask"][0].bool()].tolist()
    if consumed!=ids:
        raise RuntimeError(f"Reference changed or truncated validation input: expected {len(ids)} tokens, consumed {len(consumed)}")
    return len(ids)


def varied_state(tokenizer, prompt, upstream, bucket, position):
    """Fit unique structured audit records while retaining evidence in full."""
    evidence="CRITICAL EVIDENCE: Irreversible production backup deletion occurred without approval; the last recovery copy was removed."
    templates=[
        "record {i}: component=archive; operation=checksum; object=snapshot-{i}; result=verified; bytes={value}.",
        "record {i}: component=network; operation=read-only probe; endpoint=zone-{i}; latency_ms={value}; result=reachable.",
        "record {i}: component=inventory; operation=list; resource=volume-{i}; available_blocks={value}; status=observed.",
        "record {i}: component=scheduler; operation=inspect; job=batch-{i}; queue_depth={value}; changes=none.",
        "record {i}: component=monitor; operation=measure; sensor=rack-{i}; sample={value}; notification=recorded.",
    ]
    def assemble(records):
        index={"beginning":0,"middle":len(records)//2,"end":len(records)}[position]
        return "\n".join(records[:index]+[evidence]+records[index:])
    def count(state):
        return len(tokenizer(prompt+upstream.render_state("audit",state),add_special_tokens=True,truncation=False)["input_ids"])
    # Leave headroom, but count the actual renderer rather than assuming its overhead.
    target=bucket-32
    records=[]
    while count(assemble(records))<=target:
        i=len(records)
        records.append(templates[i%len(templates)].format(i=i,value=(i*7919+104729)%1000003))
    low,high=0,len(records)
    while low<high:
        middle=(low+high+1)//2
        if count(assemble(records[:middle]))<=target: low=middle
        else: high=middle-1
    state=assemble(records[:low])
    tokens=count(state)
    if not bucket//2<tokens<=target:
        raise RuntimeError(f"Varied state did not exercise intended large bucket: {tokens}/{bucket}")
    return state,tokens


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path("artifacts")); args=ap.parse_args()
    torch.set_num_threads(4)
    ref=reference(args.artifacts/"upstream")
    tok=ref.get_tokenizer(); prompt=ref.config.prompt
    runtime=DecisionRuntime(args.artifacts)
    if runtime.max_tokens>ref.config.max_seq_length:
        raise RuntimeError("Requested context exceeds original reference limit")
    lengths=sorted({32, *runtime.buckets, *(b+1 for b in runtime.buckets if b<runtime.max_tokens), *(b*3//4 for b in runtime.buckets if b>=2048)})
    results=[]; passed=True
    for n in lengths:
        text=exact_text(tok,prompt,n)
        consumed=checked_reference_tokens(ref,tok,prompt,text,runtime.max_tokens)
        if consumed!=n: raise RuntimeError("Boundary token count mismatch")
        with torch.inference_mode(): expected=ref.encode([text],batch_size=1).numpy()
        actual=runtime.encode([text]).numpy()
        times=[]
        for _ in range(3):
            start=time.perf_counter(); runtime.encode([text]); times.append(time.perf_counter()-start)
        cosine=float((expected*actual).sum()/(np.linalg.norm(expected)*np.linalg.norm(actual)))
        error=float(np.abs(expected-actual).max())
        ok=bool(np.isfinite(actual).all() and np.isfinite(cosine) and cosine>=.998)
        passed &= ok
        results.append({"tokens":n,"reference_consumed_tokens":consumed,"bucket":next(b for b in runtime.buckets if n<=b),"cosine":cosine,"max_abs_error":error,"warm_seconds_median":float(np.median(times)),"passed":ok})
    questions={"action":{"type":"choice","instructions":"What should happen next?","criteria":{"continue":"Proceed safely","review":"Ask a human","stop":"Stop unsafe actions"}},"risk":{"type":"score","instructions":"How risky is this?","criteria":["no risk","low risk","high risk","severe risk"]},"needs_review":{"type":"noul","instructions":"Does this need human review?"}}
    decisions=[]
    specs=[runtime.upstream.parse_question(qid,q) for qid,q in questions.items()]
    for spec in specs:
        for text in [runtime.upstream.render_question(spec)]+[runtime.upstream.render_option(spec,i) for i in range(len(spec.options))]:
            checked_reference_tokens(ref,tok,prompt,text,runtime.max_tokens)
    states=[]
    for n in sorted({513,1025, *(b-48 for b in runtime.buckets if b>=2048)}):
        # Reserve tokens for the workflow rendering and decision-bearing suffix.
        state=exact_text(tok,prompt,n,unit=" review")+" Irreversible production deletion occurred without approval."
        states.append(({"case":"repeated_review","state_base_tokens":n},state))
    for bucket in runtime.buckets:
        if 4096<=bucket<=runtime.max_tokens:
            for position in ["beginning","middle","end"]:
                state,tokens=varied_state(tok,prompt,runtime.upstream,bucket,position)
                states.append(({"case":"varied_audit","evidence_position":position,"target_bucket":bucket},state))
    varied_embeddings=[]
    for metadata,state in states:
        rendered=runtime.upstream.render_state("audit",state)
        tokens=checked_reference_tokens(ref,tok,prompt,rendered,runtime.max_tokens)
        metadata={**metadata,"rendered_state_tokens":tokens,"bucket":next(b for b in runtime.buckets if tokens<=b),"reference_consumed_tokens":tokens}
        if metadata["case"]=="varied_audit":
            with torch.inference_mode(): expected=ref.encode([rendered],batch_size=1).numpy()
            actual=runtime.encode([rendered]).numpy()
            cosine=float((expected*actual).sum()/(np.linalg.norm(expected)*np.linalg.norm(actual)))
            error=float(np.abs(expected-actual).max())
            ok=bool(np.isfinite(actual).all() and np.isfinite(cosine) and cosine>=.998)
            passed &= ok
            varied_embeddings.append({**metadata,"cosine":cosine,"max_abs_error":error,"passed":ok})
        gold=ref.decide(state,questions,workflow="audit")
        answer=runtime.decide(state,questions,workflow="audit")
        for qid,out in answer.items():
            original=gold[qid].to_dict(); diff=max(abs(out["probabilities"][k]-p) for k,p in original["probabilities"].items())
            ok=out["label"]==original["label"] and diff<=.03
            passed &= ok
            decisions.append({**metadata,"question":qid,"reference_label":original["label"],"coreml_label":out["label"],"max_probability_abs_error":diff,"passed":ok})
    try: runtime.encode([exact_text(tok,prompt,runtime.max_tokens+1)])
    except RequestError: rejected=True
    else: rejected=False
    passed &= rejected
    report={"host":platform.node(),"platform":platform.platform(),"buckets":runtime.buckets,"max_tokens":runtime.max_tokens,"boundary_embeddings":results,"varied_embeddings":varied_embeddings,"thresholds":{"embedding_cosine_min":.998,"probability_max_abs_error":.03,"labels_must_match":True},"long_decisions":decisions,"overlength_tokens":runtime.max_tokens+1,"overlength_rejected":rejected,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"encoder_sha256":package_sha256(args.artifacts),"source_revision":runtime.manifest["source_revision"],"compute_units":"CPU_AND_NE","ane_execution_trace_verified":False,"passed":bool(passed)}
    print(json.dumps(report,indent=2),flush=True)
    (args.artifacts/"validation-long.json").write_text(json.dumps(report,indent=2)+"\n")
    if not passed: raise RuntimeError("Long-context gate failed")
if __name__=="__main__": main()
