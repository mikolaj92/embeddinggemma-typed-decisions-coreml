"""Switch the existing launchd service only after hash-bound parity gates pass."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
from integrity import package_sha256

LABEL="pl.mikomac.typed-decisions"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("artifacts",type=Path); ap.add_argument("--port",type=int,default=8005); args=ap.parse_args()
    root=args.artifacts.resolve(); digest=package_sha256(root)
    for name in ["validation.json","validation-long.json"]:
        report=json.loads((root/name).read_text())
        if report.get("passed") is not True or report.get("encoder_sha256")!=digest:
            raise RuntimeError("Parity gate missing or mismatched: "+name)
    script=Path.home()/".config/typed-decisions/run.sh"
    old=script.read_text()
    lines=[line for line in old.splitlines() if not line.startswith("export DECISIONS_ARTIFACTS=")]
    pos=next(i for i,line in enumerate(lines) if line.startswith("exec "))
    lines.insert(pos,"export DECISIONS_ARTIFACTS="+shlex.quote(str(root)))
    backup=script.with_name("run-before-long-context-"+str(time.time_ns())+".sh")
    if backup.exists(): raise RuntimeError("Rollback backup already exists; inspect before retrying")
    backup.write_text(old); backup.chmod(0o700)
    tmp=script.with_suffix(".new"); tmp.write_text("\n".join(lines)+"\n"); tmp.chmod(0o700); tmp.replace(script)
    target=f"gui/{os.getuid()}/{LABEL}"
    try:
        subprocess.run(["launchctl","kickstart","-k",target],check=True)
        deadline=time.monotonic()+180
        while True:
            try:
                response=json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health",timeout=3))
                manifest=json.loads((root/"manifest.json").read_text())
                if response.get("ready") is True and response.get("buckets")==manifest["buckets"] and response.get("max_tokens")==manifest["max_tokens"]: break
            except Exception:
                pass
            if time.monotonic()>deadline: raise RuntimeError("New service readiness deadline exceeded")
            time.sleep(2)
        subprocess.run([str(Path(__file__).parent/".venv/bin/python"),str(Path(__file__).parent/"verify_idle_api.py"),"--artifacts",str(root)],check=True)
    except Exception:
        script.write_text(old); script.chmod(0o700)
        subprocess.run(["launchctl","kickstart","-k",target],check=True)
        raise
    print(json.dumps({"status":"ready","buckets":json.loads((root/"manifest.json").read_text())["buckets"],"encoder_sha256":digest,"rollback_script":str(backup)}))
if __name__=="__main__": main()
