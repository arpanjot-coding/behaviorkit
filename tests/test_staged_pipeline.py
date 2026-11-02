from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from behaviorkit.adapters import PresetAdapter,BoxOnnxAdapter
from behaviorkit.cli import main
from behaviorkit.contracts import read_jsonl
from behaviorkit.pipeline import batch_videos,capture_video
from behaviorkit.recording import apply_rules,retrack,verified_manifest
from behaviorkit.storage import sha256
from test_pipeline import arguments,write_clip


def test_real_video_capture_replay_and_cli_zero_inference(signal_model,tmp_path,monkeypatch):
    pytest.importorskip('onnxruntime')
    args=arguments(tmp_path,signal_model)
    adapter=PresetAdapter('d-fine-n',signal_model,'cpu',confidence_floor=.1,threads=1)
    capture_video(args.video,tmp_path/'candidates',adapter,epoch='repeatable')
    calls=0
    def forbidden(*a,**k): pytest.fail('replay called ONNX')
    monkeypatch.setattr(adapter.detector.session,'run',forbidden)
    monkeypatch.setattr('behaviorkit.vision.OnnxDetector.__init__',forbidden)
    assert main(['track','--recording',str(tmp_path/'candidates'),'--output',str(tmp_path/'tracks')])==0
    for name in ('a','b'):
        assert main(['apply','--recording',str(tmp_path/'tracks'),'--config',args.config,'--output',str(tmp_path/name),'--trace'])==0
    for name in ('events','counts','transitions'):
        assert (tmp_path/f'a/{name}.jsonl').read_bytes()==(tmp_path/f'b/{name}.jsonl').read_bytes()
    events=list(read_jsonl(tmp_path/'a/events.jsonl'))
    assert len(events)==1 and events[0]['timestamp']==pytest.approx(1.2)
    counts=list(read_jsonl(tmp_path/'a/counts.jsonl'))
    assert max(r['count'] for r in counts)==2 and counts[-1]['count']==0


def test_custom_non_coco_box_adapter_real_onnx(tmp_path):
    np=pytest.importorskip('numpy'); onnx=pytest.importorskip('onnx'); pytest.importorskip('onnxruntime')
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    path=tmp_path/'custom.onnx'
    graph=h.make_graph([h.make_node('Identity',['b'],['boxes']),h.make_node('Identity',['s'],['scores'])],
        'custom',[h.make_tensor_value_info('rgb',T.FLOAT,[1,3,16,16])],
        [h.make_tensor_value_info('boxes',T.FLOAT,[1,2,4]),h.make_tensor_value_info('scores',T.FLOAT,[1,2,2])],
        [nh.from_array(np.array([[[.1,.1,.4,.9],[.11,.11,.41,.91]]],np.float32),'b'),
         nh.from_array(np.array([[[.9,.1],[.8,.2]]],np.float32),'s')])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=10); onnx.save(model,path)
    contract={'schema_version':'box-onnx/1','sha256':sha256(path),'classes':['pallet','forklift'],
              'width':16,'height':16,'input':'rgb','boxes':'boxes','scores':'scores','box_units':'normalized','license':'owned generated test fixture'}
    adapter=BoxOnnxAdapter(path,contract,provider='cpu',threads=1)
    boxes,_=adapter.predict(np.zeros((40,60,3),np.uint8))
    assert len(boxes)==1 and boxes[0].label=='pallet' # classwise NMS is explicit
    contract['sha256']='0'*64
    with pytest.raises(ValueError,match='digest'): BoxOnnxAdapter(path,contract,provider='cpu')


def test_batch_restart_reuses_complete_perception_and_isolates_failed_file(signal_model,tmp_path,monkeypatch):
    pytest.importorskip('onnxruntime')
    args=arguments(tmp_path,signal_model); bad=tmp_path/'corrupt.mp4'; bad.write_bytes(b'not a video')
    adapter=PresetAdapter('d-fine-n',signal_model,'cpu',threads=1)
    root=tmp_path/'batch'
    result=batch_videos([args.video,bad],args.config,root,adapter)
    assert result['status']=='partial_failure'
    assert [f['status'] for f in result['files']]==['complete','error']
    monkeypatch.setattr(adapter,'predict',lambda *a:pytest.fail('completed file reran perception'))
    second=batch_videos([args.video,bad],args.config,root,adapter)
    assert len(second['files'][0]['attempts'])==1
    # A canceled-before-start job does not invoke inference.
    cancel=tmp_path/'cancel'; cancel.touch()
    stopped=batch_videos([args.video],args.config,tmp_path/'cancelled',adapter,cancel_file=cancel)
    assert stopped['status']=='cancelled' and stopped['files'][0]['status']=='pending'


def test_batch_failed_rule_stage_can_reuse_finished_candidate_stage(signal_model,tmp_path,monkeypatch):
    pytest.importorskip('onnxruntime')
    args=arguments(tmp_path,signal_model); adapter=PresetAdapter('d-fine-n',signal_model,'cpu',threads=1)
    import behaviorkit.recording as recording
    apply=recording.apply_rules
    monkeypatch.setattr(recording,'apply_rules',lambda *a,**kw:(_ for _ in ()).throw(OSError('temporary write failure')))
    first=batch_videos([args.video],args.config,tmp_path/'batch',adapter)
    assert first['files'][0]['status']=='error'
    monkeypatch.setattr(recording,'apply_rules',apply)
    monkeypatch.setattr(adapter,'predict',lambda *a:pytest.fail('should reuse candidate stage'))
    recovered=batch_videos([args.video],args.config,tmp_path/'batch',adapter,retry_failed=True)
    assert recovered['status']=='complete'
    assert len(recovered['files'][0]['attempts'])==2
    assert len(list(read_jsonl(tmp_path/'batch'/recovered['files'][0]['result']/'events.jsonl')))==1


def _fault_decoder(spec,queue,stop,drops,heartbeat):
    if spec.id=='stalled':
        time.sleep(20)
    else:
        from behaviorkit.live import _decode_worker
        _decode_worker(spec,queue,stop,drops,heartbeat)


def test_live_decode_failure_and_stall_are_isolated(tmp_path,monkeypatch):
    pytest.importorskip('av')
    from behaviorkit.live import LiveSources,SourceSpec
    clip=tmp_path/'clip.mp4'; write_clip(clip)
    # One decoder intentionally stalls in its isolated process.
    monkeypatch.setattr('behaviorkit.live._decode_worker',_fault_decoder)
    specs=[SourceSpec('healthy',str(clip),policy='lossless',queue_size=2,stale_seconds=10,max_retries=0),
           SourceSpec('stalled',str(clip),open_timeout=.1,read_timeout=.1,stale_seconds=.1,max_retries=0)]
    frames=[]; health=[]
    with LiveSources(specs) as sources:
        until=time.monotonic()+10
        while time.monotonic()<until and not sources.finished:
            packet=sources.poll()
            if packet:
                (frames if packet['kind']=='frame' else health).append(packet)
        assert sources.finished
        workers=[s['process'] for s in sources._workers.values()]
    assert len(frames)==16 and all(p['source']=='healthy' for p in frames)
    assert any(p.get('reason')=='decoder_watchdog_timeout' for p in health)
    assert all(not p.is_alive() for p in workers)


def test_live_bad_media_retry_budget_and_latest_queue_drop_reporting(tmp_path):
    pytest.importorskip('av')
    from behaviorkit.live import LiveSources,SourceSpec
    clip=tmp_path/'clip.mp4'; write_clip(clip)
    bad=tmp_path/'bad.mp4'; bad.write_bytes(b'bad')
    specs=[SourceSpec('latest',str(clip),queue_size=1,stale_seconds=10),
           SourceSpec('broken',str(bad),max_retries=1,retry_seconds=.01)]
    packets=[]
    with LiveSources(specs) as sources:
        until=time.monotonic()+10
        while not sources.finished and time.monotonic()<until:
            packet=sources.poll()
            if packet: packets.append(packet)
            time.sleep(.02)
        assert sources.finished
        assert sources.status()['broken']['state']=='failed'
        assert sources.status()['latest']['dropped_frames']>0
    assert any(p.get('retry')==2 for p in packets)


def test_live_command_recording_replays_exact_events(signal_model,tmp_path):
    pytest.importorskip('onnxruntime'); from behaviorkit.commands import run_live
    args=arguments(tmp_path,signal_model)
    sources=tmp_path/'sources.json'
    sources.write_text(json.dumps({'sources':[{'id':'signals','uri':args.video,'scene':args.config,
        'geometry':{'revision':'1','width':320,'height':240},'policy':'lossless','queue_size':4,'stale_seconds':20}]}))
    from types import SimpleNamespace
    live_args=SimpleNamespace(**vars(args),custom_model=None,contract=None,classes=['person'],capture_floor=.1,
                             sources=str(sources),seconds=10,outbox=str(tmp_path/'outbox.sqlite3'))
    result=run_live(live_args)
    assert result['totals']=={'frames':16,'events':1}
    apply_rules(args.output,args.config,tmp_path/'replayed')
    a=list(read_jsonl(Path(args.output)/'events.jsonl')); b=list(read_jsonl(tmp_path/'replayed/events.jsonl'))
    assert a==b
    apply_rules(args.output,None,tmp_path/'routed-replay')
    assert a==list(read_jsonl(tmp_path/'routed-replay/events.jsonl'))
