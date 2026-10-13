"""Verify actual authenticated service accepts long input and rejects inputs above the manifest token limit."""
import argparse
import json
import math
from pathlib import Path
import urllib.request
import urllib.error
from transformers import AutoTokenizer
from validate_long import exact_text
from runtime import load_upstream

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path(__file__).parent/"artifacts-long"); ap.add_argument("--port",type=int,default=8005); args=ap.parse_args()
    artifacts=args.artifacts.resolve()
    tok=AutoTokenizer.from_pretrained(artifacts/"tokenizer",local_files_only=True,trust_remote_code=False)
    config=json.loads((artifacts/"upstream/config.json").read_text())
    key=(Path.home()/".config/typed-decisions/api-key").read_text().strip()
    upstream=load_upstream(artifacts/"upstream")
    def request(length):
        # Count the actual rendered state, not raw state text.
        prefix=config["prompt"]+upstream.render_state("", "MARKER").removesuffix("MARKER")
        state=exact_text(tok,prefix,length)
        actual=len(tok(config["prompt"]+upstream.render_state("",state),add_special_tokens=True)["input_ids"])
        while actual>length:
            state=state[:-1]
            actual=len(tok(config["prompt"]+upstream.render_state("",state),add_special_tokens=True)["input_ids"])
        while actual<length:
            state+=" a"
            actual=len(tok(config["prompt"]+upstream.render_state("",state),add_special_tokens=True)["input_ids"])
        if actual!=length: raise RuntimeError("Rendered length mismatch")
        payload={"state":state,"questions":{"review":{"type":"noul","instructions":"Does this need human review?"}}}
        req=urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/decisions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
        try:
            with urllib.request.urlopen(req,timeout=120) as response: return response.status,json.load(response)
        except urllib.error.HTTPError as exc: return exc.code,json.load(exc)
    manifest=json.loads((artifacts/"manifest.json").read_text())
    limit=manifest["max_tokens"]
    statuses={}
    lengths=sorted({*manifest["buckets"], *(b+1 for b in manifest["buckets"] if b<limit)})
    for length in lengths:
        accepted,answer=request(length)
        if accepted!=200: raise RuntimeError((length,accepted,answer))
        if set(answer)!={"review"}: raise RuntimeError("Missing typed review answer")
        review=answer["review"]; probabilities=review.get("probabilities",{})
        if review.get("kind")!="noul" or type(review.get("value")) is not bool or set(probabilities)!={"false","true"}:
            raise RuntimeError("Invalid typed review answer")
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values()) or abs(sum(probabilities.values())-1)>1e-5:
            raise RuntimeError("Invalid review probabilities")
        label=review.get("label")
        if label not in probabilities or review["value"]!=(label=="true") or abs(review.get("confidence",-1)-probabilities[label])>1e-5:
            raise RuntimeError("Inconsistent typed review decision")
        statuses[str(length)]=accepted
    rejected,error=request(limit+1)
    if rejected!=422 or f"rendered text exceeds max_tokens={limit}" not in str(error.get("detail","")):
        raise RuntimeError((limit+1,rejected,error))
    print(json.dumps({"accepted_rendered_state_tokens":statuses,"overlength_rendered_state_tokens":limit+1,"overlength_status":rejected,"error":error},indent=2))
if __name__=="__main__": main()
