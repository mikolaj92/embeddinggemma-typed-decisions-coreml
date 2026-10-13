"""Exercise HTTP mixed-bucket boundaries plus idle cleanup and PID stability."""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path(__file__).parent/"artifacts-long"); ap.add_argument("--port",type=int,default=8005); ap.add_argument("--startup-timeout",type=float,default=1800); args=ap.parse_args()
    root=Path(__file__).parent
    manifest=json.loads((args.artifacts/"manifest.json").read_text())
    from integrity import package_sha256
    digest=package_sha256(args.artifacts)
    deadline=time.monotonic()+args.startup_timeout
    while time.monotonic()<deadline:
        try:
            health=json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health",timeout=3))
            if health.get("buckets")==manifest["buckets"] and health.get("max_tokens")==manifest["max_tokens"] and health.get("encoder_sha256")==digest and health.get("ready"): break
        except Exception: pass
        time.sleep(2)
    else: raise RuntimeError("readiness timeout")
    def listener_pid():
        pids=set(subprocess.check_output(["lsof","-t","-nP",f"-iTCP:{args.port}","-sTCP:LISTEN"],text=True).split())
        if len(pids)!=1: raise RuntimeError("Expected exactly one listening service PID")
        return pids.pop()
    pid=listener_pid()
    for i in range(6):
        subprocess.run([sys.executable,str(root/"smoke_long.py"),"--artifacts",str(args.artifacts),"--port",str(args.port)],check=True)
        time.sleep(20)
        health=json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health",timeout=5))
        current=listener_pid()
        if current!=pid or not health.get("ready") or health.get("encoder_sha256")!=digest: raise RuntimeError("service restarted or unavailable after idle")
        print(f"cycle={i} idle20s health_ok pid_stable",flush=True)
    print(json.dumps({"idle_cycles":6,"idle_seconds_each":20,"pid_stable":True,"passed":True}))
if __name__=="__main__": main()
