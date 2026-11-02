import json
from dataclasses import replace
import pytest
from behaviorkit import BehaviorEngine, Detection, Observation, SceneConfig
from behaviorkit.core import inside
from behaviorkit.tracking import IoUTracker
from behaviorkit.cli import demo, replay, main


def config(seconds=1,gap=.5):
    return SceneConfig.from_dict({"schema_version":"0.1","zones":{"z":[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]},"rules":[{"id":"count","kind":"count","zone":"z"},{"id":"dwell","kind":"dwell","zone":"z","seconds":seconds,"max_gap_seconds":gap}]})


def obs(t,ids=("1",),covered=True,source="s",epoch="0"):
    return Observation(t,tuple(Detection(i,"person",.9,(.2,.2,.4,.8)) for i in ids),source,epoch,covered)


def test_count_and_exact_dwell_boundary_once():
    e=BehaviorEngine(config());assert e.process(obs(0)).counts[0]['count']==1
    assert not e.process(obs(.5)).events
    result=e.process(obs(1));assert len(result.events)==1;assert result.events[0]['duration_seconds']==1
    assert not e.process(obs(1.5)).events


@pytest.mark.parametrize('disruption',["missing","unknown","gap","epoch","id"])
def test_duration_resets_on_disruption(disruption):
    e=BehaviorEngine(config());e.process(obs(0));e.process(obs(.5))
    if disruption=='missing':e.process(obs(.7,()))
    elif disruption=='unknown':
        r=e.process(obs(.7,(),False));assert r.counts[0]['count'] is None
    elif disruption=='gap':assert not e.process(obs(1.1)).events;return
    elif disruption=='epoch':assert not e.process(obs(1,epoch='1')).events;return
    elif disruption=='id':assert not e.process(obs(1,('2',))).events;return
    assert not e.process(obs(1)).events


def test_reentry_rearms_and_source_isolation():
    e=BehaviorEngine(config(seconds=.5));e.process(obs(0));e.process(obs(0,source='other'))
    assert len(e.process(obs(.5)).events)==1
    assert len(e.process(obs(.5,source='other')).events)==1
    e.process(obs(.7,()));e.process(obs(1));assert len(e.process(obs(1.5)).events)==1


def test_long_stream_state_bounded():
    e=BehaviorEngine(config())
    for i in range(2000):
        e.process(obs(i/5,(str(i),)));assert e.state_size==1
    e.process(obs(401,()));assert e.state_size==0


def test_bad_time_and_duplicate_ids():
    e=BehaviorEngine(config());e.process(obs(1))
    with pytest.raises(ValueError,match='strictly increase'):e.process(obs(1))
    with pytest.raises(ValueError,match='unique'):obs(2,('1','1'))
    with pytest.raises(ValueError):obs(float('nan'))


def test_polygon_boundary_anchor_and_confidence():
    assert inside((.1,.5),config().zones['z'])
    assert not inside((.95,.5),config().zones['z'])
    e=BehaviorEngine(config())
    r=e.process(Observation(0,(Detection('a','person',.2,(.2,.2,.4,.8)),Detection('b','car',.9,(.2,.2,.4,.8)),Detection('c','person',.9,(.2,.2,.4,.95)))))
    assert r.counts[0]['count']==0


@pytest.mark.parametrize('field,value',[('seconds',True),('seconds','20'),('max_gap_seconds',0),('kind','theft'),('zone','missing')])
def test_invalid_config(field,value):
    data={'schema_version':'0.1','zones':{'z':[[0,0],[1,0],[1,1]]},'rules':[{'id':'x','kind':'dwell','zone':'z',field:value}]}
    with pytest.raises(ValueError):SceneConfig.from_dict(data)


def test_capacity_fails_before_advancing_time():
    e=BehaviorEngine(replace(config(),max_tracks=1));e.process(obs(0))
    with pytest.raises(ValueError,match='capacity'):e.process(obs(.5,('1','2')))
    assert e.process(obs(.5)).counts[0]['count']==1


def test_tracker_expiry_and_observed_only():
    tracker=IoUTracker(min_hits=2,max_age_seconds=.5)
    d=Detection('raw','person',.9,(.2,.2,.4,.8))
    assert not tracker.update((d,),0)
    first=tracker.update((d,),.2)[0].track_id
    assert tracker.update((),.4)==()
    assert not tracker.update((d,),1)
    assert tracker.update((d,),1.2)[0].track_id!=first


def test_demo_and_replay_deterministic(tmp_path):
    a=tmp_path/'a';result=demo(a);assert result['events']==1;assert result['batches']==41
    replay(a/'observations.jsonl',a/'config.json',tmp_path/'b')
    assert (a/'events.jsonl').read_bytes()==(tmp_path/'b/events.jsonl').read_bytes()
    assert (a/'counts.jsonl').read_bytes()==(tmp_path/'b/counts.jsonl').read_bytes()


def test_no_ml_imports_for_core():
    import subprocess,sys
    code="import sys; import behaviorkit; assert all(x not in sys.modules for x in ['numpy','torch','onnxruntime','cv2','av'])"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_delimiters_in_names_do_not_collide_event_ids():
    base=config(seconds=.5)
    first=BehaviorEngine(replace(base,rules=(replace(base.rules[1],id='a:b'),)))
    second=BehaviorEngine(replace(base,rules=(replace(base.rules[1],id='a'),)))
    first.process(obs(0,source='c'));second.process(obs(0,source='b:c'))
    a=first.process(obs(.5,source='c')).events[0]['id']
    b=second.process(obs(.5,source='b:c')).events[0]['id']
    assert a!=b


def test_jsonl_limit_is_enforced_before_parsing(tmp_path):
    demo(tmp_path/'demo')
    path=tmp_path/'oversized.jsonl';path.write_bytes(b' '*(8*1024*1024+1))
    with pytest.raises(ValueError,match='exceeds 8 MiB'):
        replay(path,tmp_path/'demo/config.json',tmp_path/'failed')
