import importlib.util
import json
from pathlib import Path

import pytest

from behaviorkit.advanced import AdvancedEngine, config_diff, parse_scene
from behaviorkit.artifacts import APPROVED, verify_checkpoint
from behaviorkit.contracts import Capabilities, EvidenceFrame, read_jsonl
from behaviorkit.core import Detection, Observation
from behaviorkit.evaluation import evaluate, validate_split_groups
from behaviorkit.recording import apply_rules, record_candidates, retrack, verified_manifest
from behaviorkit.setup import Geometry, Inspector, SceneRouter

ZONE = [[0,0],[.5,0],[.5,1],[0,1]]


def scene(kind='dwell', **options):
    return parse_scene({'schema_version':'0.2','zones':{'a':ZONE,'b':[[.6,0],[1,0],[1,1],[.6,1]]},
                        'rules':[{'id':'r','kind':kind,'zone':'a','seconds':1,'max_gap_seconds':1,**options}]})


def observation(t, x=.2, *, epoch='0', covered=True, confidence=.9):
    return Observation(t, (Detection('p','person',confidence,(x-.05,.1,x+.05,.8)),) if covered and x is not None else (), 'cam', epoch, covered)


def frame(t, x=.2, **kw): return EvidenceFrame(observation(t,x,**kw),640,480)


def test_catalog_rejects_tamper_before_importing_checkpoint_loader(tmp_path, monkeypatch):
    path = tmp_path/'bad.pth'; path.write_bytes(b'not a trusted pickle')
    for name in APPROVED:
        assert len(APPROVED[name]['checkpoint_sha256']) == 64
        with pytest.raises(ValueError,match='before loading'): verify_checkpoint(name,path)
    spec = importlib.util.spec_from_file_location('prepare',Path(__file__).parents[1]/'scripts/prepare_models.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**k:pytest.fail('cached tamper must fail locally'))
    with pytest.raises(ValueError): module.download(module.URLS['d-fine-n'],path)
    with pytest.raises(ValueError): module.prepare_rtmdet(tmp_path,path)
    with pytest.raises(ValueError): module.prepare_dfine(tmp_path,path,None)


def test_hand_calculated_event_and_count_metrics():
    key = {'source':'cam','epoch':'0','rule':'r'}
    annotations = {'schema_version':'evaluation/1','split':'held-out',
                   'coverage':[{'source':'cam','epoch':'0','start':0,'end':3600},
                               {'source':'cam','epoch':'0','start':3600,'end':3700,'covered':False}],
                   'events':[{**key,'start':10,'end':12},{**key,'start':30,'end':32},
                             {**key,'start':3599,'end':3601,'censored':True}],
                   'counts':[{**key,'timestamp':1,'count':2},{**key,'timestamp':2,'count':0},
                             {**key,'timestamp':3,'count':1},{**key,'timestamp':4,'count':1}]}
    predictions = {'events':[{**key,'id':'1','start':10.1,'timestamp':12.5},
                             {**key,'id':'2','start':10.2,'timestamp':12.6},
                             {**key,'id':'3','start':20,'timestamp':22},
                             {**key,'id':'4','start':3599,'timestamp':3600}],
                   'counts':[{**key,'timestamp':1,'count':3},{**key,'timestamp':2,'count':0},
                             {**key,'timestamp':3,'count':None}]}
    report = evaluate(annotations,predictions)
    e, c = report['events'], report['counts']
    assert (e['tp'],e['fp'],e['fn'],e['duplicates']) == (1,2,1,1)
    assert e['precision'] == pytest.approx(1/3) and e['recall']==.5
    assert e['false_alerts_per_camera_hour']==2
    assert e['delay_seconds']['mean']==.5 and e['ignored_predictions']==1
    assert c['mae']==.5 and c['missing_samples']==1 and c['unknown_prediction_samples']==1
    assert report['coverage']['unknown_fraction']==pytest.approx(100/3700)


def test_evaluator_uses_maximum_matching_and_validates_denominators():
    k = {'source':'c','epoch':'0','rule':'r'}
    a = {'schema_version':'evaluation/1','coverage':[{'source':'c','epoch':'0','start':0,'end':10}],
         'events':[{**k,'start':1,'end':2},{**k,'start':1.4,'end':2.4}]}
    p = {'events':[{**k,'id':'a','start':1.2,'timestamp':2.2},{**k,'id':'b','start':.8,'timestamp':2}]}
    assert evaluate(a,p,start_tolerance=.3)['events']['tp']==2
    a['coverage'].append({'source':'c','epoch':'0','start':9,'end':11})
    with pytest.raises(ValueError,match='overlap'): evaluate(a,p)
    with pytest.raises(ValueError,match='both'): validate_split_groups([{'split':'tune','groups':['cam/day']},{'split':'test','groups':['cam/day']}])


def test_unknown_is_not_zero_or_an_exit():
    engine = AdvancedEngine(scene('exit'))
    engine.process(observation(0))
    assert not engine.process(observation(.5,covered=False)).events
    assert engine.state_size==0
    assert not engine.process(observation(1,.8)).events


@pytest.mark.parametrize('kind', ['dwell','occupancy'])
def test_duration_once_per_episode_and_epoch_reset(kind):
    engine = AdvancedEngine(scene(kind))
    assert not engine.process(observation(0)).events
    assert not engine.process(observation(.5)).events
    event = engine.process(observation(1)).events[0]
    assert event['identity_version']==3
    assert not engine.process(observation(1.5)).events
    assert not engine.process(observation(2,epoch='new')).events
    new = engine.process(observation(3,epoch='new')).events[0]
    assert new['id']!=event['id']


def test_debounced_dwell_does_not_accrue_outside_or_missing_time():
    engine = AdvancedEngine(scene('dwell',membership='debounced',exit_seconds=.6))
    events = []
    for t,x in [(0,.2),(.5,.2),(.7,.8),(.9,.2),(1.4,.2)]:
        events.extend(engine.process(observation(t,x)).events)
    assert len(events)==1 and events[0]['evidence']['supported_seconds']==pytest.approx(1)
    # Strict policy restarts at .9 instead.
    strict = AdvancedEngine(scene('dwell'))
    assert not [e for t,x in [(0,.2),(.5,.2),(.7,.8),(.9,.2),(1.4,.2)] for e in strict.process(observation(t,x)).events]


def test_entry_exit_debounce_missing_and_reentry():
    for kind in ('enter','exit'):
        engine = AdvancedEngine(scene(kind,membership='debounced',entry_seconds=.3,exit_seconds=.3))
        events = []
        for t,x in [(0,.2),(.3,.2),(.5,.8),(.6,.2),(.8,.8),(1.1,.8),(1.2,.2),(1.5,.2)]:
            events.extend(engine.process(observation(t,x)).events)
        assert len(events)==(2 if kind=='enter' else 1)
        assert len({e['id'] for e in events})==len(events)


def test_cross_direction_segment_tolerance_and_reversal():
    data=scene('cross',line=[[.5,.2],[.5,1]],direction='positive').to_dict()
    data['zones']['a']=[[0,0],[1,0],[1,1],[0,1]]
    engine = AdvancedEngine(parse_scene(data))
    events = []
    for t,x in [(0,.8),(.2,.5001),(.4,.2),(.6,.8),(.8,.2)]:
        events.extend(engine.process(observation(t,x)).events)
    assert len(events)==2
    assert all(e['evidence']['direction']=='positive' for e in events)
    assert len({e['id'] for e in events})==2


def test_sequence_timeout_and_occlusion_do_not_complete():
    engine = AdvancedEngine(scene('sequence',sequence_zones=['a','b'],sequence_timeout=.5))
    engine.process(observation(0))
    assert not engine.process(observation(.8,.8)).events
    engine.process(observation(1,.2))
    assert len(engine.process(observation(1.2,.8)).events)==1
    engine.process(observation(1.3,.2))
    engine.process(observation(1.4,covered=False))
    assert not engine.process(observation(1.5,.8)).events


def test_rule_semantics_stable_under_unrelated_edits_and_formatting():
    a = scene(); data = a.to_dict()
    data['rules'].append({'id':'other','kind':'count','zone':'b'})
    b = parse_scene(data)
    assert a.fingerprint != b.fingerprint
    assert a.semantic_fingerprint(a.rules[0])==b.semantic_fingerprint(b.rules[0])
    engines = [AdvancedEngine(x) for x in (a,b)]
    for e in engines: e.process(observation(0))
    events = [e.process(observation(1)).events[0] for e in engines]
    assert events[0]['id']==events[1]['id']
    assert config_diff(a,b)['unchanged']==['r']
    assert parse_scene(json.loads(json.dumps(a.to_dict()))).fingerprint==a.fingerprint


def test_layered_replay_without_detector_and_integrity(tmp_path,monkeypatch):
    cap = Capabilities(('person',),.3,sample_fps=2)
    source, tracks, rules = [tmp_path/p for p in ('candidates','tracks','rules')]
    record_candidates((frame(i/2) for i in range(6)),source,cap,{'video_sha256':'fixture'})
    # These functions do not import or call ONNX.
    retrack(source,tracks,tracker_parameters={'adapter':'iou','min_hits':1})
    apply_rules(tracks,scene().to_dict(),rules)
    assert len(list(read_jsonl(rules/'events.jsonl')))==1
    assert verified_manifest(rules)['totals']['events']==1
    bad = scene(confidence=.1)
    with pytest.raises(ValueError,match='capture floor'): apply_rules(tracks,bad.to_dict(),tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='pixels'): cap.require(classes=['person'],confidence=.3,pixels=True)
    with pytest.raises(ValueError,match='cadence'): cap.require(classes=['person'],confidence=.3,sample_fps=5)
    with (source/'frames.jsonl').open('a') as s: s.write('{}\n')
    with pytest.raises(ValueError,match='integrity'): retrack(source,tmp_path/'corrupt')


def test_incomplete_recordings_are_not_cache_hits(tmp_path):
    from behaviorkit.recording import cache_hit
    cap = Capabilities(('person',),.3)
    def broken():
        yield frame(0)
        raise OSError('disk fixture failure')
    out = tmp_path/'partial'
    with pytest.raises(OSError): record_candidates(broken(),out,cap,{})
    assert not cache_hit(out,'candidates',{}, {})
    with pytest.raises(ValueError,match='incomplete'): verified_manifest(out)


def test_scene_routing_geometry_and_predicted_positions():
    a,b=scene('count'),scene('count').to_dict(); b['zones']['a']=b['zones']['b']
    router=SceneRouter({'cam':(a,Geometry('1',640,480)),'other':(parse_scene(b),Geometry('1',640,480))})
    assert router.process(frame(0)).counts[0]['count']==1
    other=EvidenceFrame(Observation(0,observation(0).detections,'other'),640,480)
    assert router.process(other).counts[0]['count']==0
    predicted=EvidenceFrame(observation(.5),640,480,statuses=(('p','predicted'),))
    assert router.process(predicted).counts[0]['count']==0
    with pytest.raises(ValueError,match='aspect'): router.process(EvidenceFrame(observation(1),640,640))
    with pytest.raises(ValueError,match='unknown source'): router.process(EvidenceFrame(Observation(0,source='unknown'),640,480))
    assert router.process(EvidenceFrame(observation(1),1280,960)).counts[0]['count']==1
    changed=a.to_dict(); changed['geometry_revision']='2'
    audit=router.cutover('cam',parse_scene(changed),Geometry('2',640,480),timestamp=2,reason='operator revalidation')
    assert audit['state_policy']=='reset'
    with pytest.raises(ValueError,match='revision'): router.process(frame(2))


def test_inspection_bounded_and_does_not_change_events():
    a=scene(); e=AdvancedEngine(a,trace=True); inspect=Inspector(a,detail_limit=2)
    events=[]
    for f in (frame(0),frame(.5),frame(1),frame(1.5),frame(2,.8),frame(2.5,covered=False)):
        result=e.process(f.confirmed_observation()); events.extend(result.events)
        inspect.observe(f,result,e.dwell_states())
    report=inspect.report()
    assert len(events)==1 and len(report['details'])==2 and report['omitted_details']>0
    reasons={r['reason'] for r in report['counters']}
    assert {'insufficient_supported_duration','confirmed_now','already_emitted_episode','outside_zone','missing_coverage'} <= reasons
