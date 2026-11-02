"""Runnable, dependency-free demonstration of all schema 0.2 operators.

These generated boxes test orchestration and rules, not a trained detector.
"""
import argparse
import json
from pathlib import Path

from behaviorkit.contracts import Capabilities,EvidenceFrame,read_jsonl
from behaviorkit.core import Detection,Observation
from behaviorkit.recording import apply_rules,record_candidates,retrack


def frames():
    for index in range(126):
        t=index/5
        if t<20:
            x=.2 if t<5 else .2+.12*(t-5) if t<10 else .8 if t<14 else max(.2,.8-.12*(t-14))
            detections=(Detection('candidate-0','person',.9,(x-.04,.1,x+.04,.8)),)
        elif t<24.5:
            detections=tuple(Detection(str(i),'person',.9,(x-.04,.1,x+.04,.4)) for i,x in enumerate((.15,.4,.65)))
        else: detections=()
        yield EvidenceFrame(Observation(t,detections,'synthetic','operator-demo'),640,480)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    config=Path(__file__).resolve().parents[1]/'configs/operators.json'
    record_candidates(frames(),args.output/'candidates',Capabilities(('person',),.1,sample_fps=5),{'fixture':'generated normalized boxes; no ML inference'})
    retrack(args.output/'candidates',args.output/'tracks')
    result=apply_rules(args.output/'tracks',config,args.output/'rules',trace=True)
    kinds={e['kind'] for e in read_jsonl(args.output/'rules/events.jsonl')}
    if not {'dwell','enter','exit','cross','occupancy','sequence'}<=kinds:
        raise AssertionError(f'missing operator demonstration: {kinds}')
    print(json.dumps({'status':'passed','event_kinds':sorted(kinds),'totals':result['totals'],
                      'note':'Synthetic orchestration only, not camera-event quality'},indent=2))


if __name__=='__main__': main()
