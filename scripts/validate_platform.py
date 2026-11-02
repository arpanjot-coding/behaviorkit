"""Record actual hardware and run the portable suite on that hardware."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from behaviorkit.benchmark import environment
from behaviorkit.storage import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('runs/platform-validation'))
p.add_argument('--require-vendor',choices=['Intel','AMD','Apple'])
p.add_argument('--require-architecture',choices=['arm64','x86_64','AMD64'])
a=p.parse_args();info=environment()
if a.require_vendor and info['cpu_vendor']!=a.require_vendor:raise SystemExit(f"Expected {a.require_vendor} hardware; found {info['cpu_name']}")
if a.require_architecture and info['architecture'].lower()!=a.require_architecture.lower():raise SystemExit(f"Expected {a.require_architecture}; found {info['architecture']}")
a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1]
command=[sys.executable,'-m','pytest',str(root/'tests'),'-q','--basetemp',str((a.output/'pytest-temp').resolve()),'--junitxml',str((a.output/'tests.xml').resolve())]
run=subprocess.run(command,check=False)
info.update(test_exit_code=run.returncode,status='passed' if run.returncode==0 else 'failed',note='Actual host recorded. A skipped CoreML test is not a CoreML hardware verification.')
write_json(a.output/'platform.json',info)
print(json.dumps(info,indent=2))
raise SystemExit(run.returncode)
