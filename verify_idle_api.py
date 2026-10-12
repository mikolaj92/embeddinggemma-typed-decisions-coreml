"""Exercise HTTP mixed-bucket boundaries plus idle cleanup and PID stability."""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--artifacts",type=Path,default=Path(__file__).parent/"artifacts-long"); args=ap.parse_args()
    root=Path(__file__).parent
    for _ in range(150):
        try:
            health=json.load(urllib.request.urlopen("http://127.0.0.1:8005/health",timeout=3))
            if health.get("buckets")==[512,1024,2048] and health.get("ready"): break
        except Exception: pass
        time.sleep(2)
    else: raise RuntimeError("readiness timeout")
    pid=subprocess.check_output(["pgrep","-f","python -m uvicorn server:app"],text=True).strip()
    for i in range(6):
        subprocess.run([sys.executable,str(root/"smoke_long.py"),"--artifacts",str(args.artifacts)],check=True)
        time.sleep(20)
        health=json.load(urllib.request.urlopen("http://127.0.0.1:8005/health",timeout=5))
        current=subprocess.check_output(["pgrep","-f","python -m uvicorn server:app"],text=True).strip()
        if current!=pid or not health.get("ready"): raise RuntimeError("service restarted or unavailable after idle")
        print(f"cycle={i} idle20s health_ok pid_stable",flush=True)
    print(json.dumps({"idle_cycles":6,"idle_seconds_each":20,"pid_stable":True,"passed":True}))
if __name__=="__main__": main()
