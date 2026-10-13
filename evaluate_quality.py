"""Independent fixed-gold short/native quality benchmark; no deployment approval."""
import argparse,hashlib,json,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--artifacts",type=Path,default=Path("artifacts-8192"))
 p.add_argument("--cases",type=Path,default=Path("quality_cases.json"))
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 raw=a.cases.read_bytes();cases=json.loads(raw)
 from experiment_context import claim_report_output
 out=claim_report_output(p,a.artifacts,a.output)
 import torch
 from convert import reference,SOURCE,REVISION
 from experiment_context import checked_features,FILLER
 torch.set_num_threads(2);ref=reference(a.artifacts.resolve()/"upstream");tok=ref.get_tokenizer();up=sys.modules[type(ref).__module__]
 config=ref.config.to_dict();assert ref.config.max_seq_length==8192
 report={"scope":"Fixed synthetic gold, frozen before inference; diagnostic not representative deployment approval","cases_sha256":hashlib.sha256(raw).hexdigest(),"source":SOURCE,"revision":REVISION,"production_changed":False,"quality_approved":False,"rows":[]}
 def save():out.write_text(json.dumps(report,indent=2)+"\n")
 for case in cases:
  if len(case["questions"])!=1:raise ValueError("Expected one fixed question per case")
  qid,question=next(iter(case["questions"].items()));spec=up.parse_question(qid,question);gold=str(case["expected"][qid]["label"])
  if gold not in spec.options:raise ValueError("Gold outside options")
  question_texts=[up.render_question(spec),*[up.render_option(spec,i) for i in range(len(spec.options))]]
  with torch.inference_mode():
   fixed=[]
   for text in question_texts:
    f,_=checked_features(ref,tok,text);fixed.append(ref.embed(**f))
   for count in (0,1,16):
    records=[FILLER[i%len(FILLER)].format(i=i,v=(i*7919+104729)%1000003) for i in range(count)]
    state="\n".join([case["state"],*records]);text=up.render_state(case["workflow"],state);f,proof=checked_features(ref,tok,text)
    if proof["consumed_tokens"]>2048:raise ValueError("Benchmark exceeds native small budget")
    emb=ref.embed(**f);kind=torch.tensor([spec.kind_id]);mask=torch.ones((1,len(spec.options)),dtype=torch.bool)
    logits=ref.head(emb,fixed[0],torch.cat(fixed[1:]).unsqueeze(0),kind,mask).float();prob=(logits/ref.temperatures.float()[kind].unsqueeze(-1)).softmax(-1)
    if not torch.isfinite(prob).all():raise RuntimeError("Non-finite original head")
    chosen=int(prob[0].argmax());label=spec.options[chosen]
    row={"id":case["id"],"kind":spec.kind,"filler_records":count,"gold_label":gold,"rationale":case["rationale"],"label":label,"correct":label==gold,"probabilities":dict(zip(spec.options,prob[0].tolist())),"logits":logits[0].tolist(),"token_proof":proof}
    if count==0:
     original=ref.decide(case["state"],{qid:question},workflow=case["workflow"],tokenizer=tok)[qid].to_dict()
     err=max(abs(original["probabilities"][o]-row["probabilities"][o]) for o in spec.options)
     row["original_decide_max_probability_error"]=err
     if err>1e-5 or original["label"]!=label:raise RuntimeError("Original decide sequential parity failed")
    report["rows"].append(row);save();print(json.dumps({k:row[k] for k in ("id","kind","filler_records","correct","label")}),flush=True)
 report["accuracy_by_filler"]={str(n):sum(r["correct"] for r in report["rows"] if r["filler_records"]==n)/len(cases) for n in (0,1,16)}
 report["configuration_unchanged"]=ref.config.to_dict()==config;assert report["configuration_unchanged"]
 report["status"]="complete";save()

if __name__=="__main__":main()
