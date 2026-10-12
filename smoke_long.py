"""Verify actual authenticated service accepts long input and rejects 2049 tokens."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
from transformers import AutoTokenizer
from validate_long import exact_text
from runtime import load_upstream

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path(__file__).parent/"artifacts-long"); args=ap.parse_args()
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
        req=urllib.request.Request("http://127.0.0.1:8005/v1/decisions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
        try:
            with urllib.request.urlopen(req,timeout=120) as response: return response.status,json.load(response)
        except urllib.error.HTTPError as exc: return exc.code,json.load(exc)
    statuses={}
    for length in [513,1536,2048]:
        accepted,answer=request(length)
        if accepted!=200: raise RuntimeError((length,accepted,answer))
        statuses[str(length)]=accepted
    rejected,error=request(2049)
    if rejected!=422: raise RuntimeError((2049,rejected))
    print(json.dumps({"accepted_rendered_state_tokens":statuses,"overlength_rendered_state_tokens":2049,"overlength_status":rejected,"error":error},indent=2))
if __name__=="__main__": main()
