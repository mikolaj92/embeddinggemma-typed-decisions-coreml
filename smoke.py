"""Probe installed API with local private credential (never prints it)."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--url",default="http://127.0.0.1:8005"); args=ap.parse_args()
    key=(Path.home()/".config/typed-decisions/api-key").read_text().strip()
    sample=json.loads((Path(__file__).parent/"examples.json").read_text())[0]
    health=json.load(urllib.request.urlopen(args.url+"/health",timeout=10))
    assert health["ready"] is True
    try: urllib.request.urlopen(urllib.request.Request(args.url+"/v1/decisions",data=b"{}",headers={"Content-Type":"application/json"}),timeout=10)
    except urllib.error.HTTPError as exc: assert exc.code==401
    else: raise RuntimeError("Unauthenticated request accepted")
    req=urllib.request.Request(args.url+"/v1/decisions",data=json.dumps(sample).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    answer=json.load(urllib.request.urlopen(req,timeout=60))
    assert set(answer)==set(sample["questions"])
    print(json.dumps({"health":health,"unauthorized_status":401,"answers":answer},indent=2))
if __name__=="__main__": main()
