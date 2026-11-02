from __future__ import annotations
from dataclasses import asdict
import json
import platform
import statistics
import subprocess
import sys
import time
import os
from pathlib import Path

from . import __version__
from .core import BehaviorEngine, SceneConfig, Observation, Detection


def stats(values):
    ordered = sorted(values)
    def percentile(p):
        position = (len(ordered)-1)*p
        lo = int(position); hi = min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    return {"mean":statistics.mean(values), "p50":percentile(.5), "p95":percentile(.95), "p99":percentile(.99), "min":ordered[0], "max":ordered[-1]}


def environment(include_nvidia=False):
    result = {"python":sys.version,"system":platform.system(),"architecture":platform.machine(),"platform":platform.platform(),"behaviorkit":__version__,"logical_cpus":os.cpu_count()}
    name=platform.processor()
    try:
        if platform.system()=="Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                name=winreg.QueryValueEx(key,'ProcessorNameString')[0]
        elif platform.system()=="Darwin":
            name=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,timeout=5).strip()
        elif Path('/proc/cpuinfo').exists():
            with Path('/proc/cpuinfo').open() as source:
                for line in source:
                    if line.lower().startswith('model name'):
                        name=line.split(':',1)[1].strip();break
    except (OSError,subprocess.SubprocessError):pass
    result['cpu_name']=name or platform.machine()
    result['cpu_vendor']='Intel' if 'intel' in name.lower() else ('AMD' if 'amd' in name.lower() else ('Apple' if 'apple' in name.lower() else 'unknown'))
    if not include_nvidia:return result
    try:
        result["nvidia_smi"] = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):
        result["nvidia_smi"] = None
    try:
        result["gpu_point_sample"] = subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader"],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):
        result["gpu_point_sample"] = None
    return result


def engine_benchmark(batches=5000, tracks=100):
    if not 10 <= batches <= 1000000 or not 1 <= tracks <= 1000:
        raise ValueError("engine benchmark: batches 10..1000000; tracks 1..1000")
    config = SceneConfig.from_dict({"schema_version":"0.1", "zones":{"all":[[0,0],[1,0],[1,1],[0,1]]}, "rules":[{"id":f"rule-{i}","kind":"count" if i%2==0 else "dwell", "zone":"all", "seconds":3,"max_gap_seconds":0.5} for i in range(20)], "max_tracks":tracks})
    engine = BehaviorEngine(config)
    detections = tuple(Detection(str(i),"person",.9,(.2,.2,.4,.8)) for i in range(tracks))
    durations=[]; events=0
    for i in range(batches+100):
        batch = Observation(i/5,detections)
        start=time.perf_counter(); evaluation=engine.process(batch); elapsed=(time.perf_counter()-start)*1000
        if i>=100:
            durations.append(elapsed)
        events+=len(evaluation.events)
    return {"schema_version":"0.1", "kind":"engine_only", "environment":environment(), "workload":{"tracks":tracks,"rules":20,"batches":batches,"warmup_batches":100,"sampling_hz":5}, "evaluation_ms":stats(durations), "final_dwell_states":engine.state_size,"emitted_events":events,"limits":"Excludes model inference, decoding, input construction, serialization and tracking; synthetic identical-box observations."}


def model_benchmark(names, models_dir, provider="auto", iterations=50, warmup=10,threads=0,profile_dir=None):
    import numpy as np
    import cv2
    from .vision import OnnxDetector, sha256
    if not 1 <= iterations <= 10000 or not 0 <= warmup <= 1000:
        raise ValueError("iterations must be 1..10000; warmup 0..1000")
    rgb=np.random.default_rng(2026).integers(0,256,(720,1280,3),dtype=np.uint8)
    include_nvidia=provider in {'cuda','tensorrt','auto'}
    report={"schema_version":"0.1","kind":"model_latency_synthetic","environment":environment(include_nvidia),"input":{"kind":"seeded synthetic RGB image","seed":2026,"width":1280,"height":720,"warning":"Latency only. No detection or event accuracy measured; excludes video decode and tracking."},"profiling_enabled":False,"profiling_note":"Timed session has profiling disabled. A separate session verifies the selected execution provider. session.run timings include transfers and wait for CPU outputs.","system_conditions":"Shared host; background workloads were not controlled. Point samples are context only, not idle-system performance or a definitive model ranking.","threads_requested":threads,"opencv_threads":1,"iterations":iterations,"warmup":warmup,"requested_provider":provider,"models":[]}
    previous_cv_threads=cv2.getNumThreads();cv2.setNumThreads(1)
    try:
        for name in names:
            item={"name":name};detector=verification=None
            try:
                start=time.perf_counter();detector=OnnxDetector(name,models_dir,provider,profile=False,threads=threads);item["session_create_seconds"]=time.perf_counter()-start
                for _ in range(warmup):detector.predict(rgb)
                timings=[]
                for _ in range(iterations):
                    _,timing=detector.predict(rgb);timings.append(timing)
                selected=detector.selected_provider
                item.update({"status":"ok","model_sha256":sha256(detector.path),"manifest":detector.manifest,"onnxruntime":detector.ort_version,"selected_provider":selected,"provider_note":detector.selection_note,"intra_op_threads":detector.threads,"session_providers":detector.session.get_providers(),"timings_ms":{key:stats([t[key] for t in timings]) for key in timings[0]}})
                detector=None
                verification=OnnxDetector(name,models_dir,selected,profile=True,threads=threads,profile_dir=profile_dir)
                verification.predict(rgb);item["profile"]=verification.end_profile()
                if not item["profile"]["requested_provider_observed"]:
                    item.update(status="error",error=f"No {selected} provider kernel events observed; no acceleration claim is made")
            except Exception as e:item.update(status="error",error=f"{type(e).__name__}: {e}")
            finally:detector=verification=None
            report["models"].append(item)
    finally:cv2.setNumThreads(previous_cv_threads)
    report["environment_after"]=environment(include_nvidia)
    return report
