"""Learned-model qualification of the new capture/track/apply path.

Supply a rights-cleared one-person image and local, exported models. This is a
controlled regression, not a held-out camera-event accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import time

from behaviorkit import __version__
from behaviorkit.adapters import PresetAdapter
from behaviorkit.benchmark import environment
from behaviorkit.contracts import read_jsonl
from behaviorkit.pipeline import capture_video
from behaviorkit.recording import apply_rules,retrack
from behaviorkit.storage import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True); p.add_argument('--models-dir',type=Path,required=True)
    p.add_argument('--provider',choices=['cpu','cuda','coreml'],required=True)
    p.add_argument('--models',nargs='+',default=['rtmdet-tiny','d-fine-n','rf-detr-nano'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--frames',type=int,default=31)
    p.add_argument('--monitor-resources',action='store_true'); a=p.parse_args()
    if a.frames<31: p.error('--frames must be at least 31')
    import av
    import cv2
    import numpy as np
    a.output.mkdir(parents=True,exist_ok=False)
    image=cv2.imread(str(a.image))
    if image is None: raise ValueError('image cannot be decoded')
    rgb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(512,512))
    clip=a.output/'positive-negative.mp4'
    with av.open(str(clip),'w') as container:
        stream=container.add_stream('mpeg4',rate=10); stream.width=stream.height=512; stream.pix_fmt='yuv420p'; stream.bit_rate=2000000
        for i in range(a.frames):
            # Repeat separate positive/empty episodes to exercise cleanup/re-entry.
            pixels=rgb if i%31<20 else np.zeros_like(rgb)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    config=a.output/'scene.json'
    write_json(config,{'schema_version':'0.1','zones':{'all':[[0,0],[1,0],[1,1],[0,1]]},
                      'rules':[{'id':'count','kind':'count','zone':'all'},
                               {'id':'dwell','kind':'dwell','zone':'all','seconds':1,'max_gap_seconds':.5}]})
    report={'schema_version':'system-validation/1','behaviorkit':__version__,'environment':environment(include_nvidia=a.provider=='cuda'),
            'provider':a.provider,'image_sha256':sha256(a.image),'video_sha256':sha256(clip),'cases':[],
            'scope':'controlled learned-model positive/negative pipeline and exact replay; not surveillance accuracy'}
    for name in a.models:
        base=a.output/name; base.mkdir()
        begin=time.perf_counter()
        # ORT retains profiler events until end_profiling. Never use continuous
        # profiling for a native-memory soak or a normal production capture.
        probe=PresetAdapter(name,a.models_dir,a.provider,confidence_floor=.1,threads=4,profile=True,profile_dir=base/'profiles')
        probe.predict(rgb); probe.predict(np.zeros_like(rgb))
        profile=probe.detector.end_profile()
        probe=None
        import gc
        gc.collect()
        adapter=PresetAdapter(name,a.models_dir,a.provider,confidence_floor=.1,threads=4)
        calls=0; original=adapter.predict
        def counted(pixels):
            nonlocal calls
            calls+=1; return original(pixels)
        adapter.predict=counted
        capture_video(clip,base/'candidates',adapter,epoch='controlled-qualification',monitor_resources=a.monitor_resources)
        capture_calls=calls
        retrack(base/'candidates',base/'tracks')
        for version in ('rules','replayed'): apply_rules(base/'tracks',config,base/version,trace=True)
        parity=all((base/'rules'/(n+'.jsonl')).read_bytes()==(base/'replayed'/(n+'.jsonl')).read_bytes() for n in ('events','counts','transitions'))
        counts=list(read_jsonl(base/'rules/counts.jsonl')); events=list(read_jsonl(base/'rules/events.jsonl'))
        lengths=[]; length=0
        for index in range(0,a.frames,2):
            if index%31<20: length+=1
            elif length: lengths.append(length); length=0
        if length: lengths.append(length)
        expected_events=sum(length>=7 for length in lengths)
        passed=bool(parity and calls==capture_calls and capture_calls>0 and max(r['count'] for r in counts)==1 and
                    any(r['count']==0 for r in counts) and len(events)==expected_events and profile['requested_provider_observed'])
        case={'model':name,'status':'passed' if passed else 'failed','inference_calls':capture_calls,'replay_inference_calls':calls-capture_calls,
              'exact_replay':parity,'count_samples':len(counts),'max_count':max(r['count'] for r in counts),
              'dwell_events':len(events),'first_confirmation':events[0]['timestamp'] if events else None,
              'expected_dwell_events':expected_events,'provider_profile_scope':'separate two-frame probe; continuous capture profiling disabled',
              'wall_seconds':time.perf_counter()-begin,'provider_profile':profile,'model_sha256':adapter.detector.manifest['sha256']}
        if a.monitor_resources:
            metrics=list(read_jsonl(base/'candidates/resources.jsonl'))
            case['resources']={'samples':len(metrics),'peak_rss_bytes':max(r['rss_bytes'] for r in metrics),
                               'first_rss_bytes':metrics[0]['rss_bytes'],'last_rss_bytes':metrics[-1]['rss_bytes'],
                               'peak_native_threads':max(r['native_threads'] for r in metrics),
                               'peak_vram_bytes':max((r['vram_bytes'] for r in metrics if r['vram_bytes'] is not None),default=None)}
        report['cases'].append(case); write_json(a.output/'validation.json',report)
    report['status']='passed' if all(c['status']=='passed' for c in report['cases']) else 'failed'
    write_json(a.output/'validation.json',report)
    print(json.dumps(report,indent=2)); return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
