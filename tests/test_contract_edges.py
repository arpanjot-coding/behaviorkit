import json
from pathlib import Path
import time

import pytest

from behaviorkit.advanced import AdvancedEngine
from behaviorkit.contracts import Capabilities,EvidenceFrame,read_jsonl
from behaviorkit.core import Detection,Observation
from behaviorkit.evaluation import evaluate_tracks
from behaviorkit.recording import apply_rules,cache_hit,record_candidates,retrack,verified_manifest
from test_system_foundations import frame,scene,observation


def test_tracking_metrics_hand_computable():
    def row(t,identity='truth',covered=True):
        from dataclasses import asdict
        return asdict(Observation(t,(Detection(identity,'person',.9,(.1,.1,.3,.8)),) if covered else (),covered=covered))
    truth=[row(i) for i in range(4)]
    predicted=[row(0,'one'),row(1,'one'),{**row(2),'detections':[]},row(3,'two')]
    report=evaluate_tracks(truth,predicted)
    assert (report['matched'],report['missed'],report['id_switches'],report['fragments'])==(3,1,1,1)
    predicted[2]=row(2,covered=False)
    report=evaluate_tracks(truth,predicted)
    assert report['id_switches']==0 and report['fragments']==0 and report['unknown_frames']==1


def test_tracker_reset_changes_epoch_and_candidate_inspection(tmp_path):
    cap=Capabilities(('person',),.1,sample_fps=2)
    frames=[frame(0),frame(.5),frame(1,covered=False),frame(1.5),frame(2)]
    record_candidates(frames,tmp_path/'c',cap,{})
    retrack(tmp_path/'c',tmp_path/'t',tracker_parameters={'adapter':'iou','min_hits':1})
    saved=list(read_jsonl(tmp_path/'t/frames.jsonl'))
    assert saved[0]['observation']['epoch']!=saved[-1]['observation']['epoch']
    from behaviorkit.setup import Inspector
    inspect=Inspector(scene(),candidate_stage=True)
    inspect.observe(frames[0])
    assert inspect.report()['counters'][0]['reason']=='untracked_candidate'


def test_implementation_hash_invalidates_cache_not_historical_evidence(tmp_path,monkeypatch):
    from behaviorkit import recording
    out=tmp_path/'c'; record_candidates([frame(0)],out,Capabilities(('person',),.1),{},parameters={'x':1})
    assert cache_hit(out,'candidates',{}, {'x':1})
    monkeypatch.setattr(recording,'implementation_hash',lambda kind:'changed')
    assert not cache_hit(out,'candidates',{}, {'x':1})
    assert verified_manifest(out)['status']=='complete'


def test_legacy_golden_generated_by_installed_v03_wheel(tmp_path):
    from behaviorkit.cli import replay
    fixture=Path(__file__).parent/'fixtures/golden-v03'
    replay(fixture/'observations.jsonl',fixture/'config.json',tmp_path/'replay')
    for name in ('events','counts','transitions','diagnostics'):
        assert (fixture/(name+'.jsonl')).read_bytes()==(tmp_path/'replay'/(name+'.jsonl')).read_bytes()


def test_resource_monitor_streams_native_metrics_with_hard_bound(tmp_path):
    pytest.importorskip('psutil')
    from behaviorkit.resources import ResourceMonitor
    with ResourceMonitor(tmp_path/'resources.jsonl',interval=.05,max_samples=2) as monitor:
        deadline=time.monotonic()+2
        while monitor.samples<2 and time.monotonic()<deadline: time.sleep(.01)
    rows=list(read_jsonl(tmp_path/'resources.jsonl'))
    assert len(rows)==2 and all(r['rss_bytes']>0 and r['native_threads']>0 for r in rows)


def test_low_score_capture_does_not_change_default_iou_association(tmp_path):
    def f(t,strong,weak):
        return EvidenceFrame(Observation(t,(Detection('high','person',.9,strong),Detection('low','person',.15,weak))),640,480)
    a,b=(.1,.1,.6,.9),(.2,.1,.7,.9)
    c=tmp_path/'c'; t=tmp_path/'t'
    record_candidates([f(0,a,b),f(.2,b,a),f(.4,b,a)],c,Capabilities(('person',),.1),{})
    retrack(c,t,tracker_parameters={'adapter':'iou','min_hits':1})
    rows=list(read_jsonl(t/'frames.jsonl'))
    ids=[r['observation']['detections'][0]['track_id'] for r in rows]
    assert len(set(ids))==1 and all(len(r['observation']['detections'])==1 for r in rows)
    assert verified_manifest(t)['capabilities']['confidence_floor']==.3
    with pytest.raises(ValueError,match='capture floor'):
        apply_rules(t,scene(confidence=.1),tmp_path/'impossible')
