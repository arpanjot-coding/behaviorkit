"""Bounded hashing and atomic run metadata, independent of ML packages."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name+'-',suffix='.tmp',delete=False) as target:
            temporary=Path(target.name)
            json.dump(value,target,indent=2,allow_nan=False);target.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def guard_outputs(output,inputs,names):
    output=Path(output)
    for source in map(Path,inputs):
        for name in names:
            target=output/name
            if source.resolve()==target.resolve() or (source.exists() and target.exists() and source.samefile(target)):
                raise ValueError(f'input/output paths overlap: {source} and {target}')
