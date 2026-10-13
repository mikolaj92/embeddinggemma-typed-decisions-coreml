"""Matched filler prefixes with fixed facts and frozen original decision rule.
Native <=2048 only, diagnostic not representative benchmark or approval.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,default=Path("artifacts-8192"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();source=a.artifacts.resolve()
    from experiment_context import claim_report_output
    out=claim_report_output(p,source,a.output)
    import torch
    from convert import reference,SOURCE,REVISION
    from experiment_context import CASES,FILLER,WORKFLOW,predict,peak_rss_bytes
    torch.set_num_threads(2)
    r=reference(source/"upstream");tok=r.get_tokenizer();up=sys.modules[type(r).__module__]
    config=r.config.to_dict();assert r.config.max_seq_length==8192
    report={"diagnostic":"matched-prefix-filler-curve","scope":"One pair, fixed question/workflow/head/temperature. Identical filler for safe/unsafe at each count; no rewording, tuning, or new pooling.","source":SOURCE,"revision":REVISION,"native_limit":8192,"configuration":config,"quality_approved":False,"production_changed":False,"cases":[]}
    def save():
        report["process_peak_rss_bytes"]=peak_rss_bytes();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n")
    for count in (0,1,4,8,16,32):
        records=[FILLER[i%len(FILLER)].format(i=i,v=(i*7919+104729)%1000003) for i in range(count)]
        filler="\n".join(records);digest=hashlib.sha256(filler.encode()).hexdigest()
        for c in CASES:
            state="\n".join([c["evidence"],*records])
            n=len(tok(r.config.prompt+up.render_state(WORKFLOW,state),truncation=False)["input_ids"])
            assert n<=2048
            result=predict(r,tok,up,state)
            row={"id":c["id"],"filler_records":count,"filler_sha256":digest,"gold_label":c["gold_label"],**result}
            row["truth_correct"]=row["answer"]["label"]==c["gold_label"]
            row["review_logit_margin"]=row["logits"][1]-row["logits"][0]
            report["cases"].append(row);save()
            print(json.dumps({"id":c["id"],"count":count,"tokens":n,"correct":row["truth_correct"],"margin":row["review_logit_margin"]}),flush=True)
    report["configuration_unchanged"]=r.config.to_dict()==config
    assert report["configuration_unchanged"];report["status"]="diagnostic_complete";save()

if __name__=="__main__":main()
