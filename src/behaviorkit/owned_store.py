"""Retention only for files created by this store, never arbitrary run/input paths."""
from __future__ import annotations
from contextlib import closing
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from .storage import sha256, write_json
from .contracts import read_json
from .core import number


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True,exist_ok=True)
        if self.root.is_symlink() or self.root.resolve()!=self.root:
            raise ValueError('store root cannot be a link or junction')
        self.root = self.root.resolve()
        marker = self.root/'ownership.json'
        if not marker.exists():
            if any(self.root.iterdir()): raise ValueError('refusing to adopt a nonempty directory')
            write_json(marker,{'schema_version':'owned-store/1','id':uuid.uuid4().hex})
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_nlink!=1: raise ValueError('invalid store ownership marker')
        metadata = read_json(marker)
        if metadata.get('schema_version')!='owned-store/1': raise ValueError('unsupported artifact store')
        self.identity = metadata['id']
        self._db_path = self.root/'index.sqlite3'
        if self._db_path.is_symlink() or self._db_path.exists() and self._db_path.stat().st_nlink!=1:
            raise ValueError('store database cannot be a link')
        with closing(self._connect()) as db:
            db.execute('CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,name TEXT UNIQUE,digest TEXT,bytes INTEGER,created REAL,expires REAL,pinned INTEGER,kind TEXT,available INTEGER)')
            db.commit()

    def _connect(self):
        if self.root.is_symlink() or self.root.resolve()!=self.root:
            raise ValueError('store root changed')
        marker=self.root/'ownership.json'
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_nlink!=1 or marker.resolve().parent!=self.root:
            raise ValueError('store ownership marker changed to link')
        if read_json(marker).get('id')!=self.identity:
            raise ValueError('store ownership changed')
        if self._db_path.is_symlink() or self._db_path.resolve().parent!=self.root or self._db_path.exists() and self._db_path.stat().st_nlink!=1:
            raise ValueError('store database changed to link')
        db=sqlite3.connect(self._db_path,timeout=5)
        db.row_factory=sqlite3.Row
        return db

    def _owned_path(self,name):
        if Path(name).name!=name or not name.endswith('.artifact'):
            raise ValueError('invalid owned artifact name')
        path=self.root/name
        if path.is_symlink() or path.resolve().parent!=self.root:
            raise ValueError('artifact path escapes store or is a link')
        if path.exists() and (not path.is_file() or path.stat().st_nlink!=1):
            raise ValueError('artifact must be a singly linked regular file')
        return path

    def put(self, source, *, kind='metadata', retention_seconds=None, pinned=False):
        source=Path(source)
        if not source.is_file(): raise ValueError('source must be a regular file')
        if retention_seconds is not None: number(retention_seconds,'retention_seconds',0,10*365*86400)
        if type(pinned) is not bool: raise ValueError('pinned must be boolean')
        if kind not in {'metadata','trace','image','recording'}: raise ValueError('unsupported artifact kind')
        identity=uuid.uuid4().hex; name=identity+'.artifact'; target=self._owned_path(name)
        before=source.stat()
        try:
            with source.open('rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,length=1024*1024)
                dst.flush(); os.fsync(dst.fileno())
            after=source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('input changed during artifact copy')
            digest=sha256(target); created=time.time()
            with closing(self._connect()) as db:
                db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,1)',
                           (identity,name,digest,target.stat().st_size,created,
                            created+retention_seconds if retention_seconds is not None else None,int(pinned),kind))
                db.commit()
        except BaseException:
            target.unlink(missing_ok=True); raise
        return {'id':identity,'sha256':digest,'path':str(target),'bytes':target.stat().st_size}

    def availability(self, identity):
        with closing(self._connect()) as db:
            row=db.execute('SELECT * FROM artifacts WHERE id=?',(identity,)).fetchone()
        if row is None: raise ValueError('unknown artifact ID')
        path=self._owned_path(row['name'])
        available=bool(row['available'] and path.is_file())
        if available and (path.stat().st_size!=row['bytes'] or sha256(path)!=row['digest']):
            raise ValueError('owned artifact integrity failure')
        return {'id':identity,'available':available,'replay_available':available,
                'reason':None if available else 'evidence removed or missing'}

    def prune(self, *, max_bytes=None, max_trace_bytes=None, now=None, protected_paths=(), apply=False):
        for name,value in [('max_bytes',max_bytes),('max_trace_bytes',max_trace_bytes)]:
            if value is not None and (type(value) is not int or value<0): raise ValueError(f'{name} must be nonnegative integer')
        now=time.time() if now is None else number(now,'now',0)
        protected={Path(p).resolve() for p in protected_paths}
        if type(apply) is not bool: raise ValueError('apply must be boolean')
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute('SELECT * FROM artifacts WHERE available=1 ORDER BY created,id').fetchall()
            total=sum(r['bytes'] for r in rows); traces=sum(r['bytes'] for r in rows if r['kind']=='trace')
            selected=[]
            for row in rows:
                path=self._owned_path(row['name'])
                if row['pinned'] or path in protected: continue
                expired=row['expires'] is not None and row['expires']<=now
                over= max_bytes is not None and total>max_bytes
                trace_over=row['kind']=='trace' and max_trace_bytes is not None and traces>max_trace_bytes
                if not (expired or over or trace_over): continue
                if path.exists() and (path.stat().st_size!=row['bytes'] or sha256(path)!=row['digest']):
                    raise ValueError('refusing cleanup of modified owned artifact')
                selected.append({'id':row['id'],'bytes':row['bytes'],'reason':'expired' if expired else 'quota'})
                if apply:
                    path.unlink(missing_ok=True)
                    db.execute('UPDATE artifacts SET available=0 WHERE id=?',(row['id'],))
                total-=row['bytes']
                if row['kind']=='trace': traces-=row['bytes']
            db.commit()
        return {'applied':apply,'artifacts':selected,'remaining_bytes':total,'remaining_trace_bytes':traces,
                'quota_satisfied':(max_bytes is None or total<=max_bytes) and (max_trace_bytes is None or traces<=max_trace_bytes),
                'note':'Pinned and protected files can prevent satisfying a quota. Deleted evidence cannot be replayed.'}


def redact_rgb(rgb, boxes):
    """Solid-fill explicit export boxes; caller supplies their chosen regions."""
    import numpy as np
    from .core import Detection
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('redaction requires HWC uint8 RGB')
    result=rgb.copy(); height,width=result.shape[:2]
    import math
    for i,box in enumerate(boxes):
        box=Detection(str(i),'redaction',1,box).box
        x1,y1=math.floor(box[0]*width),math.floor(box[1]*height)
        x2,y2=math.ceil(box[2]*width),math.ceil(box[3]*height)
        result[y1:y2,x1:x2]=0
    return result
