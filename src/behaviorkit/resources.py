"""Optional process/native resource sampling, streamed with a hard record limit."""
from contextlib import AbstractContextManager
from pathlib import Path
import threading
import time
from .contracts import json_line
from .core import number


def snapshot(*, include_vram=False):
    import psutil
    process=psutil.Process()
    children=process.children(recursive=True)
    alive=[]
    for child in [process,*children]:
        try: alive.append((child,child.memory_info().rss,child.num_threads()))
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    result={'unix':time.time(),'rss_bytes':sum(x[1] for x in alive),'native_threads':sum(x[2] for x in alive),
            'processes':len(alive),'scope':'current process and accessible children'}
    try:
        result['handles_or_fds']=sum(p.num_handles() if hasattr(p,'num_handles') else p.num_fds() for p,_,_ in alive)
    except (psutil.NoSuchProcess,psutil.AccessDenied): result['handles_or_fds']=None
    # Avoid an implicit SDK/driver dependency. GPU callers may provide a separate
    # device-level report, which is never mislabeled as process VRAM.
    result['vram_bytes']=None
    if include_vram:
        import subprocess
        try:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                            '--format=csv,noheader,nounits'],text=True,timeout=2,
                                           stderr=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            pids={p.pid for p,_,_ in alive}; memory=[]
            for line in output.splitlines():
                fields=[x.strip() for x in line.split(',')]
                if len(fields)==2 and fields[0].isdigit() and int(fields[0]) in pids and fields[1].isdigit():
                    memory.append(int(fields[1])*1024*1024)
            result['vram_bytes']=sum(memory) if memory else None
            result['vram_note']='NVIDIA process query; WDDM/driver may not expose per-process memory'
        except (OSError,subprocess.SubprocessError): result['vram_note']='NVIDIA process memory unavailable'
    return result


class ResourceMonitor(AbstractContextManager):
    def __init__(self,path,*,interval=1,max_samples=10000,include_vram=False):
        number(interval,'interval',.05,60)
        if type(max_samples) is not int or not 1<=max_samples<=1000000: raise ValueError('invalid max_samples')
        self.path,self.interval,self.max_samples=Path(path),interval,max_samples
        self.include_vram=include_vram
        self.stop=threading.Event(); self.error=None; self.samples=0
    def __enter__(self):
        import psutil
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x',encoding='utf-8',newline='\n')
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start(); return self
    def _run(self):
        try:
            while not self.stop.is_set() and self.samples<self.max_samples:
                json_line(self.stream,snapshot(include_vram=self.include_vram)); self.stream.flush(); self.samples+=1
                self.stop.wait(self.interval)
        except Exception as error: self.error=f'{type(error).__name__}: {error}'
    def __exit__(self,*exc):
        self.stop.set(); self.thread.join(timeout=3)
        if self.thread.is_alive(): raise RuntimeError('resource sampler failed to stop')
        self.stream.close()
        if self.error and not exc[0]: raise RuntimeError(self.error)
