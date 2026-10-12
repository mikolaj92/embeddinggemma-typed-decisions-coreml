"""Install a per-user launchd service on this Mac; no secrets in the plist/repo."""
import argparse
import os
from pathlib import Path
import plistlib
import secrets
import subprocess

LABEL="pl.mikomac.typed-decisions"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8005); ap.add_argument("--host",default="127.0.0.1"); args=ap.parse_args()
    root=Path(__file__).resolve().parent
    import json
    report=json.loads((root/"artifacts"/"validation.json").read_text())
    from integrity import package_sha256
    if report.get("passed") is not True or report.get("encoder_sha256") != package_sha256(root/"artifacts"):
        raise RuntimeError("Real-weight parity must pass for this exact package before service installation")
    state=Path.home()/".config"/"typed-decisions"; state.mkdir(parents=True,exist_ok=True); state.chmod(0o700)
    key=state/"api-key"
    if not key.exists():
        fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w") as f: f.write(secrets.token_urlsafe(32))
    key.chmod(0o600)
    script=state/"run.sh"
    import shlex
    script.write_text("#!/bin/sh\nset -eu\nexport DECISIONS_API_KEY=\"$(/bin/cat "+shlex.quote(str(key))+")\"\nexport OMP_NUM_THREADS=4\nexport TOKENIZERS_PARALLELISM=false\ncd "+shlex.quote(str(root))+"\nexec "+shlex.quote(str(root/".venv/bin/python"))+" -m uvicorn server:app --host "+shlex.quote(args.host)+" --port "+str(args.port)+"\n")
    script.chmod(0o700)
    agents=Path.home()/"Library/LaunchAgents"; agents.mkdir(parents=True,exist_ok=True)
    path=agents/(LABEL+".plist")
    if path.exists(): raise RuntimeError("Service already installed; inspect existing plist before replacing")
    spec={"Label":LABEL,"ProgramArguments":["/bin/sh",str(script)],"RunAtLoad":True,"KeepAlive":True,"ThrottleInterval":30,"StandardOutPath":str(state/"stdout.log"),"StandardErrorPath":str(state/"stderr.log")}
    with path.open("wb") as f: plistlib.dump(spec,f)
    path.chmod(0o600)
    subprocess.run(["launchctl","bootstrap",f"gui/{os.getuid()}",str(path)],check=True)
    print(f"Installed {LABEL} on {args.host}:{args.port}; credential stored privately, not printed.")
if __name__=="__main__": main()
