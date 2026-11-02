from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.request import Request,urlopen

import pytest

from behaviorkit.core import Detection
from behaviorkit.delivery import JsonlSink,NotificationPolicy,Outbox
from behaviorkit.owned_store import ArtifactStore,redact_rgb
from behaviorkit.profiles import select_profile
from behaviorkit.wizard import make_server,validate_setup


def event(identity='event', **kw):
    return {'id':identity,'source':'cam','epoch':'0','rule':'dwell','start':0,'timestamp':3,**kw}


def test_outbox_facts_survive_suppression_and_unknown_clocks(tmp_path):
    box=Outbox(tmp_path/'events.sqlite3')
    policy=NotificationPolicy('open-hours',utc_minutes=((600,660),),required_fact='armed')
    result=box.enqueue(event(),policy=policy,now=100)
    assert result['reason']=='unknown_wall_clock' and box.status()=={'facts':1,'pending':0,'sent':0}
    p=NotificationPolicy('armed',required_fact='armed')
    facts={'armed':{'value':True,'observed_utc':0,'expires_utc':50}}
    assert box.enqueue(event('expired'),policy=p,external_facts=facts,now=100)['reason']=='expired_or_future_external_fact'
    facts['armed']['expires_utc']=200
    assert box.enqueue(event('valid'),policy=p,external_facts=facts,now=100)['reason']=='eligible'
    assert len(list(box.facts()))==3 and box.status()['pending']==1
    assert box.enqueue(event('valid'),policy=p,external_facts=facts,now=100)['duplicate']
    with pytest.raises(ValueError,match='collision'): box.enqueue(event('valid',timestamp=4),policy=p,now=100)


def test_outbox_lease_exclusivity_crash_retry_and_restart(tmp_path):
    path=tmp_path/'outbox.sqlite3'; a=Outbox(path); b=Outbox(path)
    a.enqueue(event(),now=0)
    claimed=a.claim(now=0,lease_seconds=10)
    assert len(claimed)==1 and not b.claim(now=1)
    recovered=b.claim(now=11,lease_seconds=10)
    assert len(recovered)==1 and recovered[0]['attempt']==2
    with pytest.raises(ValueError,match='lease'): a.settle(claimed[0],now=12)
    b.settle(recovered[0],error='sink offline',now=12)
    assert not a.claim(now=15)
    delivery=[]
    result=Outbox(path).dispatch(delivery.append,now=16)
    assert result['sent']==1 and delivery[0]['event_id']=='event'
    assert a.status()=={'facts':1,'pending':0,'sent':1}


def test_outbox_cooldown_raw_events_and_capacity_retry(tmp_path):
    box=Outbox(tmp_path/'outbox.sqlite3',max_pending=1)
    policy=NotificationPolicy(cooldown_seconds=10)
    box.enqueue(event('a'),policy=policy,now=0)
    assert box.enqueue(event('b'),policy=policy,now=1)['reason']=='cooldown'
    with pytest.raises(RuntimeError,match='fact retained'): box.enqueue(event('c'),policy=policy,now=11)
    assert box.status()['facts']==3
    box.dispatch(JsonlSink(tmp_path/'delivery.jsonl'),now=11)
    assert box.enqueue(event('c'),policy=policy,now=12)['reason']=='eligible'
    assert box.dispatch(JsonlSink(tmp_path/'delivery.jsonl'),now=12)['sent']==1
    assert len((tmp_path/'delivery.jsonl').read_text().splitlines())==2


def test_sink_failure_does_not_acknowledge_message(tmp_path):
    box=Outbox(tmp_path/'outbox.sqlite3'); box.enqueue(event(),now=0)
    def broken(message): raise OSError('disk full')
    assert box.dispatch(broken,now=0)['failed']==1
    assert box.status()['pending']==1
    assert not box.claim(now=1)
    assert len(box.claim(now=2))==1


def test_retention_only_deletes_owned_verified_unpinned_files(tmp_path):
    source=tmp_path/'input.mp4'; source.write_bytes(b'video not owned')
    store=ArtifactStore(tmp_path/'store')
    a=store.put(source,kind='trace',retention_seconds=0)
    b=store.put(source,pinned=True)
    c=store.put(source,kind='trace')
    unrelated=store.root/'never-owned.txt'; unrelated.write_text('keep me')
    preview=store.prune(max_bytes=0,protected_paths=[c['path']])
    assert not preview['applied'] and [x['id'] for x in preview['artifacts']]==[a['id']]
    assert Path(a['path']).exists() and not preview['quota_satisfied']
    store.prune(max_bytes=0,protected_paths=[c['path']],apply=True)
    assert source.exists() and unrelated.exists() and Path(b['path']).exists() and Path(c['path']).exists()
    assert not store.availability(a['id'])['replay_available']
    Path(c['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='modified'): store.prune(max_bytes=0,apply=True)


def test_store_refuses_adoption_and_hardlink_cleanup(tmp_path):
    import os
    occupied=tmp_path/'occupied'; occupied.mkdir(); (occupied/'model.onnx').write_bytes(b'input')
    with pytest.raises(ValueError,match='nonempty'): ArtifactStore(occupied)
    source=tmp_path/'input'; source.write_bytes(b'source'); store=ArtifactStore(tmp_path/'owned')
    a=store.put(source); target=Path(a['path']); target.unlink()
    try: os.link(source,target)
    except OSError: pytest.skip('hard links unavailable')
    with pytest.raises(ValueError,match='singly linked'): store.prune(max_bytes=0,apply=True)
    assert source.read_bytes()==b'source'


def test_redaction_is_explicit_and_does_not_change_inference_input():
    np=pytest.importorskip('numpy'); rgb=np.full((10,20,3),255,np.uint8)
    out=redact_rgb(rgb,[(.2,.2,.8,.8)])
    assert rgb.min()==255 and out[3:7,5:15].max()==0 and out[0,0].min()==255


def test_budget_planner_refuses_unqualified_or_unknown_quality():
    profile={'id':'measured','qualification':'learned_model_and_events','hardware_id':'h','dataset_id':'d',
             'precision':.95,'recall':.9,'false_alerts_per_hour':.2,'delay_seconds':2,'p95_ms':30,
             'rss_bytes':1000000,'evidence_sha256':'e'*64,'model_sha256':'a'*64}
    args=dict(min_precision=.9,min_recall=.8,max_false_alerts_per_hour=.5,max_delay_seconds=3,
              max_p95_ms=40,max_rss_bytes=2000000,hardware_id='h',dataset_id='d')
    result=select_profile([profile,{**profile,'id':'unqualified','qualification':'export_only','p95_ms':1},
                           {**profile,'id':'unknown','recall':None}],**args)
    assert result['selected']['id']=='measured' and len(result['rejected'])==2
    assert select_profile([profile],**{**args,'max_p95_ms':10})['selected'] is None


def test_real_bytetrack_outputs_only_observed_confirmed_boxes():
    pytest.importorskip('trackers')
    from behaviorkit.motion import ByteTrackAdapter
    tracker=ByteTrackAdapter(classes=('person','box'),min_hits=2)
    def detections(x): return (Detection('candidate','person',.9,(x,.1,x+.2,.8)),)
    first=tracker.update(detections(.1),0)
    second=tracker.update(detections(.11),.2)
    assert len(second)==1 and second[0].box==detections(.11)[0].box
    assert not tracker.update((),.4) # predicted state is never returned
    recovered=tracker.update(detections(.12),.6)
    assert recovered and recovered[0].track_id==second[0].track_id
    with pytest.raises(ValueError,match='increase'): tracker.update((),.6)


def test_loopback_wizard_validation_and_package_resource():
    setup={'scene':{'schema_version':'0.2','zones':{'all':[[0,0],[1,0],[1,1],[0,1]]},
                    'rules':[{'id':'d','kind':'dwell','zone':'all','seconds':3,'max_gap_seconds':.5}]},
           'geometry':{'revision':'1','width':640,'height':480},'sample_fps':5}
    assert validate_setup(setup)['valid']
    server=make_server(0); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}'
        assert 'Define what the camera' in urlopen(url).read().decode()
        request=Request(url+'/validate',json.dumps(setup).encode(),{'Content-Type':'application/json','Origin':url})
        assert json.load(urlopen(request))['valid']
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url+'/validate',json.dumps(setup).encode(),{'Origin':'https://untrusted.example'}))
        assert error.value.code==403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
