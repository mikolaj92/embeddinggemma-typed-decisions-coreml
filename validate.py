"""Real-weight reference comparison, not a model-quality benchmark."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from convert import reference
from runtime import DecisionRuntime, RequestError

def cases():
    questions={"action":{"type":"choice","instructions":"What should the system do?","criteria":{"continue":"Proceed safely.","review":"Ask a human.","stop":"Stop dangerous actions."}},"risk":{"type":"score","instructions":"How risky is this?","criteria":["no risk","low risk","high risk","severe risk"]},"review":{"type":"noul","instructions":"Does this need human review?"}}
    return [{"workflow":"agent_trace_observability","state":s,"questions":questions} for s in [
        {"task":"Read documentation","constraint_violations":0,"irreversible_actions":0},
        {"task":"Delete production backups","constraint_violations":2,"irreversible_actions":1},
        {"zadanie":"Usuń stare kopie zapasowe","ograniczenia":["Nie usuwaj ostatniej kopii"],"naruszenia":1},
        {"task":"Send an invoice for 500 EUR","status":"awaiting approval"},
        {"task":"Actualizar precios","notes":"Aplicó un 90% de descuento a todo el catálogo"},
        {"task":"Uncertain request","notes":"The human and automated policy disagree."},
        "安全なドキュメント検索", "Sprawdź dokumentację, bez zmiany plików.",
        {"notes":"review the document carefully. "*65},
    ]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path("artifacts")); ap.add_argument("--reference-only",action="store_true"); args=ap.parse_args()
    torch.set_num_threads(4)
    ref=reference(args.artifacts/"upstream")
    samples=cases()
    rows=[]
    texts=[]
    for sample in samples:
        up=__import__("runtime").load_upstream(args.artifacts/"upstream")
        qs=[up.parse_question(qid,q) for qid,q in sample["questions"].items()]
        texts += [up.render_state(sample["workflow"],sample["state"])]
        for q in qs: texts += [up.render_question(q)]+[up.render_option(q,i) for i in range(len(q.options))]
        answer=ref.decide(sample["state"],sample["questions"],workflow=sample["workflow"])
        rows.append({"request":sample,"reference":{k:v.to_dict() for k,v in answer.items()}})
    tokenizer=ref.get_tokenizer()
    # Exact accepted boundary and repeated-token stress, generated without truncation.
    for word in ["document ", "!!! ", "bezpieczeństwo "]:
        token_ids=tokenizer(ref.config.prompt + word*600,add_special_tokens=True)["input_ids"][:512]
        # Decode a prefix, then choose the largest text that actually fits the prompt.
        candidate=tokenizer.decode(token_ids[8:-1],skip_special_tokens=True)
        while len(tokenizer(ref.config.prompt+candidate,add_special_tokens=True)["input_ids"])>512:
            candidate=candidate[:-1]
        texts.append(candidate)
    # Construct exactly 512 input tokens (including actual prompt and specials).
    base="stress "
    candidate=base
    while len(tokenizer(ref.config.prompt+candidate,add_special_tokens=True)["input_ids"])<512:
        candidate+=base
    while len(tokenizer(ref.config.prompt+candidate,add_special_tokens=True)["input_ids"])>512:
        candidate=candidate[:-1]
    assert len(tokenizer(ref.config.prompt+candidate,add_special_tokens=True)["input_ids"])==512
    texts.append(candidate)
    with torch.no_grad(): expected=ref.encode(texts,batch_size=1).numpy()
    np.save(args.artifacts/"reference_embeddings.npy",expected)
    (args.artifacts/"reference_cases.json").write_text(json.dumps({"texts":texts,"cases":rows},ensure_ascii=False,indent=2))
    if args.reference_only: return
    runtime=DecisionRuntime(args.artifacts)
    lengths=[len(tokenizer(ref.config.prompt+t,add_special_tokens=True)["input_ids"]) for t in texts]
    start=time.perf_counter(); actual=runtime.encode(texts).numpy(); elapsed=time.perf_counter()-start
    if not np.isfinite(actual).all(): raise RuntimeError("non-finite encoder")
    cos=(expected*actual).sum(-1)/(np.linalg.norm(expected,axis=-1)*np.linalg.norm(actual,axis=-1))
    diffs=[]; matches=0; total=0
    for row in rows:
        answer=runtime.decide(**row["request"])
        for qid,out in answer.items():
            gold=row["reference"][qid]; matches+=out["label"]==gold["label"]; total+=1
            diffs += [abs(out["probabilities"][k]-p) for k,p in gold["probabilities"].items()]
    try: runtime.encode(["token "*(runtime.max_tokens*2)])
    except RequestError: overlength_rejected=True
    else: overlength_rejected=False
    report={"host":platform.node(),"platform":platform.platform(),"encoder_texts":len(texts),"cosine_min":float(cos.min()),"cosine_mean":float(cos.mean()),"max_embedding_abs_error":float(np.abs(expected-actual).max()),"probability_max_abs_error":max(diffs),"label_matches":matches,"questions":total,"encode_seconds":elapsed,"overlength_rejected":overlength_rejected,"compute_units":"CPU_AND_NE","ane_placement_verified":False,"passed":bool(cos.min()>=0.998 and max(diffs)<=0.03 and matches==total and overlength_rejected)}
    from integrity import package_sha256
    report["encoder_sha256"]=package_sha256(args.artifacts)
    report["source_revision"]=runtime.manifest["source_revision"]
    report["max_tested_tokens"]=max(len(tokenizer(ref.config.prompt+t,add_special_tokens=True)["input_ids"]) for t in texts)
    print(json.dumps(report,indent=2),flush=True)
    (args.artifacts/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]: raise RuntimeError("Core ML parity gate failed")
if __name__=="__main__": main()
