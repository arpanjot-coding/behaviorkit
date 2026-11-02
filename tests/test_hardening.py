"""Audit failures, independent matching checks and actual-video snapshot parity."""
from dataclasses import replace
import itertools
import json
import random

import pytest

from behaviorkit import BehaviorEngine, Detection, Observation, SceneConfig
from behaviorkit.cli import replay, run_video
from behaviorkit.tracking import IoUTracker, iou
from test_engine import config, obs
from test_pipeline import arguments


def test_config_identity_uses_effective_values_not_json_formatting():
    first=config(seconds=1)
    second=SceneConfig.from_dict(json.loads(json.dumps(first.to_dict(),indent=4)))
    assert first.fingerprint==second.fingerprint
    changed=replace(first,rules=(first.rules[0],replace(first.rules[1],seconds=2)))
    ids=[]
    for scene in (first,changed):
        engine=BehaviorEngine(scene)
        for i in range(11):
            ids.extend(e['id'] for e in engine.process(obs(i/5)).events)
    assert len(ids)==2 and ids[0]!=ids[1]
    with pytest.raises(AttributeError):engine.config=first


def test_source_close_releases_capacity_and_cannot_preserve_dwell():
    engine=BehaviorEngine(replace(config(),max_sources=1),trace=True)
    engine.process(obs(0));engine.process(obs(.5))
    closed=engine.close_source('s')
    assert closed.transitions[0]['reason']=='source_closed'
    assert engine.state_size==0 and not engine.close_source('s').diagnostics
    assert not engine.process(obs(0,source='new',epoch='new')).events
    assert engine.dwell_states()[0]['start']==0


@pytest.mark.parametrize('reason', ['track_missing','source_not_covered','zone_exit','confidence_below_threshold','label_changed','observation_gap','epoch_changed'])
def test_public_trace_explains_resets(reason):
    engine=BehaviorEngine(config(),trace=True);engine.process(obs(0))
    observation={'track_missing':obs(.2,()),'source_not_covered':obs(.2,(),False),
        'zone_exit':Observation(.2,(Detection('1','person',.9,(.91,.2,.99,.8)),),'s'),
        'confidence_below_threshold':Observation(.2,(Detection('1','person',.1,(.2,.2,.4,.8)),),'s'),
        'label_changed':Observation(.2,(Detection('1','car',.9,(.2,.2,.4,.8)),),'s'),
        'observation_gap':obs(1),'epoch_changed':obs(.2,epoch='new')}[reason]
    result=engine.process(observation)
    assert any(t['transition']=='reset' and t['reason']==reason for t in result.transitions)


def test_inspection_is_an_owned_copy_and_tracing_does_not_change_events():
    plain=BehaviorEngine(config());traced=BehaviorEngine(config(),trace=True)
    for i in range(6):
        a=plain.process(obs(i/5));b=traced.process(obs(i/5))
        assert a.counts==b.counts and a.events==b.events and not a.transitions
    state=traced.dwell_states('s')[0];state['start']=-100
    assert traced.dwell_states('s')[0]['start']==0
    assert traced.dwell_states('other')==()
    assert b.transitions[0]['transition']=='confirmed'


def test_maximum_cardinality_retains_both_ids_in_a_greedy_failure():
    tracker=IoUTracker(min_hits=1)
    def d(box):return Detection('raw','person',.9,box)
    tracker.update([d((0,0,.6,.5)),d((.3,0,.9,.5))],0)
    result=tracker.update([d((.1,0,.7,.5)),d((0,0,.4,.5))],.2)
    assert [d.track_id for d in result]==['2','1']


def test_matching_cardinality_against_exhaustive_small_assignment_oracle():
    rng=random.Random(527)
    for _ in range(35):
        old=[];new=[]
        for target in (old,new):
            for i in range(4):
                x=rng.random()*.65;w=.1+rng.random()*.25
                target.append(Detection(str(i),'person',.9,(x,.1,min(1,x+w),.8)))
        feasible=[]
        for assignment in itertools.product(range(-1,4),repeat=4):
            used=[i for i in assignment if i>=0]
            if len(used)!=len(set(used)):continue
            if all(j<0 or iou(old[j].box,new[i].box)>=.3 for i,j in enumerate(assignment)):feasible.append(len(used))
        tracker=IoUTracker(min_hits=1,max_tracks=8);tracker.update(old,0)
        result=tracker.update(new,.2)
        assert sum(int(d.track_id)<=4 for d in result)==max(feasible)


def test_retire_unmatched_tracks_preserves_current_matches_and_reports_eviction():
    tracker=IoUTracker(min_hits=1,max_tracks=2)
    a=Detection('a','person',.9,(.1,.1,.2,.2));b=Detection('b','person',.9,(.4,.4,.5,.5));c=Detection('c','person',.9,(.7,.7,.8,.8))
    tracker.update((a,b),0)
    result=tracker.update((b,c),.2)
    assert [d.track_id for d in result]==['2','3']
    assert len(tracker.tracks)==2 and tracker.last_diagnostics==('track_evicted_for_capacity: 1',)
    with pytest.raises(ValueError,match='capacity'):tracker.update((a,b,c),.4)
    assert tracker.last_timestamp==.2


def test_video_freezes_config_evaluates_before_eof_and_replays_exactly(signal_model,tmp_path,monkeypatch):
    from behaviorkit.vision import OnnxDetector
    args=arguments(tmp_path,signal_model);args.trace=True
    initial=json.loads(open(args.config).read());calls=0;early=[]
    predict=OnnxDetector.predict
    def change_config(detector,pixels):
        nonlocal calls
        calls+=1
        if calls==1:
            edited=json.loads(json.dumps(initial))
            for rule in edited['rules']:rule['label']='car'
            with open(args.config,'w') as target:json.dump(edited,target)
        if calls==10:early.extend(json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines())
        return predict(detector,pixels)
    monkeypatch.setattr(OnnxDetector,'predict',change_config)
    summary=run_video(args)
    assert summary['events']==1 and early and summary['status']=='ok'
    assert json.loads((tmp_path/'run/config.snapshot.json').read_text())==initial
    replayed=replay(tmp_path/'run/observations.jsonl',tmp_path/'run/config.snapshot.json',tmp_path/'replayed',trace=True)
    assert summary['config_fingerprint']==replayed['config_fingerprint']
    assert summary['observations_sha256']==replayed['observations_sha256']
    for name in ('counts','events','transitions'):
        assert (tmp_path/f'run/{name}.jsonl').read_bytes()==(tmp_path/f'replayed/{name}.jsonl').read_bytes()
    timings=[json.loads(line) for line in (tmp_path/'run/stage_timings.jsonl').read_text().splitlines()]
    assert all(t['sample_pipeline_ms']>=t['rules_ms']>=0 for t in timings)


def test_separate_file_runs_get_distinct_source_epochs(signal_model,tmp_path):
    args=arguments(tmp_path,signal_model);first=run_video(args)
    first_events=(tmp_path/'run/events.jsonl').read_bytes()
    args.output=str(tmp_path/'second');second=run_video(args)
    assert first['source_epoch']!=second['source_epoch']
    assert first_events!=(tmp_path/'second/events.jsonl').read_bytes()
    args.epoch='caller-managed-epoch';args.output=str(tmp_path/'third');third=run_video(args)
    assert third['source_epoch']=='caller-managed-epoch'


def test_named_but_wrong_shape_model_is_rejected_before_processing(signal_model):
    import onnx
    from behaviorkit.storage import sha256
    from behaviorkit.vision import OnnxDetector
    path=signal_model/'d-fine-n/model.onnx'
    model=onnx.load(path);model.graph.input[0].type.tensor_type.shape.dim[2].dim_value=320
    onnx.save(model,path)
    manifest=path.with_name('manifest.json');value=json.loads(manifest.read_text());value['sha256']=sha256(path);manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='shape'):OnnxDetector('d-fine-n',signal_model,'cpu',threads=1)


@pytest.mark.parametrize('corruption',['shape','nonfinite','probability'])
def test_invalid_runtime_outputs_fail_instead_of_producing_observed_zero(signal_model,corruption):
    import numpy as np
    from types import SimpleNamespace
    from behaviorkit.vision import OnnxDetector
    detector=OnnxDetector('d-fine-n',signal_model,'cpu',threads=1)
    values={'labels':np.array([[0,0]]),'boxes':np.array([[[10,10,20,20],[30,30,40,40]]]),'scores':np.array([[.8,.9]])}
    if corruption=='shape':values['scores']=np.array([[.8]])
    elif corruption=='nonfinite':values['scores'][0,0]=float('nan')
    else:values['scores'][0,0]=1.1
    detector.session=SimpleNamespace(run=lambda names,feed:[values[name] for name in names])
    with pytest.raises(RuntimeError,match='ONNX output'):detector.predict(np.zeros((100,100,3),dtype=np.uint8))
