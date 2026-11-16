"""Record real pipeline/API tests and execute the final review notebook."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leafcheck.common import atomic_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--artifact",type=Path,default=ROOT / "artifacts/leafcheck177-v1")
    args = parser.parse_args()
    artifact = json.loads((args.artifact / "artifact.json").read_text())
    assert artifact["final_evaluation_complete"] and artifact["supplementary_evaluation_complete"]
    checkpoint_hash = sha256(args.artifact / "model.pt")
    assert checkpoint_hash == artifact["checkpoint_sha256"]
    with urlopen(args.url + "/api/status", timeout=10) as response: status = json.load(response)
    assert status["models"]["final177"]["checkpoint_sha256"] == checkpoint_hash
    assert all(model["ready"] for model in status["models"].values())
    output = ROOT / "runs/final177/verification"; output.mkdir(exist_ok=True)
    environment = os.environ.copy(); environment["LEAFCHECK_TEST_URL"] = args.url
    records = []
    for label, directory in [("pipeline", "tests"), ("website", "leafcheck-demo/tests")]:
        log = output / (label + "_tests.log")
        command = [sys.executable,"-m","unittest","discover","-s",directory,"-v"]
        began = time.perf_counter()
        with log.open("w") as file:
            result = subprocess.run(command,cwd=ROOT,env=environment,stdout=file,stderr=subprocess.STDOUT)
        text = log.read_text(); count = re.search(r"Ran (\d+) tests? in",text)
        passed = result.returncode == 0 and count is not None and text.rstrip().endswith("OK")
        records.append({"suite":label,"command":command,"exit_code":result.returncode,"passed":passed,
                        "test_cases":int(count.group(1)) if count else None,"seconds":time.perf_counter()-began,
                        "log":str(log.relative_to(ROOT)),"log_sha256":sha256(log)})
        if not passed:
            atomic_json(ROOT / "docs/test_results.json",{"created_utc":utc_now(),"checkpoint_sha256":checkpoint_hash,"suites":records})
            raise RuntimeError(f"{label} verification failed; inspect {log}")
        print(label, "passed", records[-1]["test_cases"], "cases",flush=True)
    notebook_path = ROOT / "cnn_final_model_review.ipynb"
    notebook = nbformat.read(notebook_path,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and 'ARTIFACT = ROOT / "artifacts/leafcheck177-v1"' in cell.source:
            cell.source = cell.source.replace('ARTIFACT = ROOT / "artifacts/leafcheck177-v1"',
                                              "ARTIFACT = Path(" + repr(str(args.artifact.resolve())) + ")")
    began = time.perf_counter()
    NotebookClient(notebook,timeout=180,kernel_name="leafcheck",resources={"metadata":{"path":str(ROOT)}}).execute()
    executed = output / "cnn_final_model_review.executed.ipynb"
    nbformat.write(notebook,executed)
    assert all(out.get("output_type") != "error" for cell in notebook.cells if cell.cell_type=="code" for out in cell.get("outputs",[]))
    assert sha256(args.artifact / "model.pt") == checkpoint_hash
    record = {"created_utc":utc_now(),"source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "url":args.url,"app_revision":status["app_revision"],"checkpoint_sha256":checkpoint_hash,"suites":records,
              "review_notebook":{"executed_without_errors":True,"seconds":time.perf_counter()-began,
                                  "executed_artifact":str(args.artifact.resolve()),
                                  "source":str(notebook_path.relative_to(ROOT)),"source_sha256":sha256(notebook_path),
                                  "executed_copy":str(executed.relative_to(ROOT)),"executed_sha256":sha256(executed)},
              "unchanged_frozen_checkpoint_after_checks":True}
    atomic_json(ROOT / "docs/test_results.json",record)
    print(json.dumps(record,indent=2),flush=True)


if __name__ == "__main__":main()
