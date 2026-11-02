"""Measure retained Python allocations while track IDs churn; no vision model."""
import argparse
import gc
import json
from pathlib import Path
import tracemalloc
from behaviorkit import BehaviorEngine,SceneConfig,Observation,Detection
from behaviorkit.benchmark import environment
from behaviorkit.storage import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--batches',type=int,default=20000);p.add_argument('--tracks',type=int,default=10)
p.add_argument('--output',type=Path,default=Path('runs/soak.json'))
a=p.parse_args()
if not 2000<=a.batches<=1000000 or not 1<=a.tracks<=200:raise SystemExit('batches must be 2000..1000000; tracks 1..200')
config=SceneConfig.from_dict({'schema_version':'0.1','zones':{'all':[[0,0],[1,0],[1,1],[0,1]]},'rules':[{'id':'n','kind':'count','zone':'all'},{'id':'d','kind':'dwell','zone':'all','seconds':3,'max_gap_seconds':.5}]})
engine=BehaviorEngine(config);tracemalloc.start();baseline=None;samples=[];max_states=0
for i in range(a.batches+1000):
    detections=tuple(Detection(f'{i}-{j}','person',.9,(.2,.2,.4,.8)) for j in range(a.tracks))
    engine.process(Observation(i/5,detections))
    max_states=max(max_states,engine.state_size)
    if i==999:
        gc.collect();baseline=tracemalloc.get_traced_memory()[0]
    if i>=1000 and (i-999)%1000==0:samples.append({'batch':i-999,'retained_bytes':tracemalloc.get_traced_memory()[0]})
gc.collect();current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
passed=max_states<=a.tracks and current-baseline<2*1024*1024
result={'kind':'bounded_state_soak','status':'passed' if passed else 'failed','environment':environment(),'batches':a.batches,'warmup_batches':1000,'tracks_per_batch':a.tracks,'track_ids':'all replaced every batch','max_dwell_states':max_states,'retained_start_bytes':baseline,'retained_end_bytes':current,'retained_growth_bytes':current-baseline,'peak_traced_bytes':peak,'samples':samples,'limits':'Python allocation tracing only; not a multi-hour decoded-video/RSS/VRAM test. Acceptance: bounded states and <2 MiB post-warm-up retained growth.'}
write_json(a.output,result);print(json.dumps(result,indent=2));raise SystemExit(0 if passed else 1)
