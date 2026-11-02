"""Archive one committed revision, never ignored weights, credentials or run state."""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--revision',default='HEAD')
p.add_argument('--output',type=Path,default=ROOT.parent/'behaviorkit-prototype-source.zip')
a=p.parse_args()
if a.revision.startswith('-'): p.error('revision cannot be an option')
a.output.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','-C',str(ROOT),'archive','--format=zip',
                '--prefix=behaviorkit-prototype/',f'--output={a.output.resolve()}',a.revision],check=True)
print(a.output.resolve())
