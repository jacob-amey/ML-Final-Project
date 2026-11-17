"""Run the next declared experiment only after its reference finishes safely."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-run",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--log",type=Path,required=True)
    args=parser.parse_args()
    state={"stopped":False,"child":None}
    def stop(signum,frame):
        state["stopped"]=True
        if state["child"] is not None and state["child"].poll() is None:
            state["child"].send_signal(signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    while not state["stopped"]:
        path=args.wait_run / "status.json"
        status=json.loads(path.read_text()) if path.exists() else {}
        if status.get("state")=="complete":
            assert status["checkpoint_reload_verified"]
            break
        if status.get("state") in {"interrupted","failed"}:
            raise SystemExit("Reference stopped before completion; next experiment was not started.")
        time.sleep(3)
    if state["stopped"]:return
    assert not args.run.exists(),"Use a new run directory; never overwrite an experiment"
    command=[sys.executable,"-m","leafcheck.train","--config",str(args.config),"--run",str(args.run)]
    print("Reference complete; starting",args.run,flush=True)
    with args.log.open("w") as file:
        state["child"]=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
        code=state["child"].wait()
    raise SystemExit(code)


if __name__=="__main__":main()
