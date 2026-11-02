"""Evidence-export checks, including real decoded-video/ONNX replay parity."""
import importlib.util
from pathlib import Path
import pytest
from behaviorkit import BehaviorEngine,SceneConfig,Detection,Observation

spec=importlib.util.spec_from_file_location('build_video_demo',Path(__file__).resolve().parents[1]/'scripts/build_video_demo.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)

def test_explanation_uses_actual_engine_timer_and_one_event():
    engine=BehaviorEngine(SceneConfig.from_dict(demo.SCENE));person=Detection('one','person',.9,(.2,.2,.4,.8));events=[]
    for i in range(21):
        result,states,resets=demo.explain(engine,Observation(i/5,(person,)));events.extend(result.events)
        assert states[0]['duration']==pytest.approx(i/5)
    assert len(events)==1 and events[0]['timestamp']==3
    assert states[0]['emitted'] and not resets

def test_explanation_resets_on_missing_observation():
    engine=BehaviorEngine(SceneConfig.from_dict(demo.SCENE));person=Detection('one','person',.9,(.2,.2,.4,.8))
    demo.explain(engine,Observation(0,(person,)))
    result,states,resets=demo.explain(engine,Observation(.2,()))
    assert result.counts[0]['count']==0 and states==[] and resets==['one']
    _,states,_=demo.explain(engine,Observation(.4,(person,)))
    assert states[0]['start']==.4

def test_manual_abstentions_and_missing_samples_are_not_scored():
    frames=[{'timestamp':1.,'count':{'count':0}}]
    score=demo.score_checks(frames,[{'t':1,'count':1},{'t':2,'count':None},{'t':3,'count':2}])
    assert score['scored']==1 and score['exact']==0 and score['mean_absolute_error']==1
    assert [c['status'] for c in score['checks']]==['scored','human_abstained','no_nearby_sample']

def test_evidence_export_runs_real_pipeline_and_exact_replay(signal_model,tmp_path):
    pytest.importorskip('onnxruntime');av=pytest.importorskip('av');np=pytest.importorskip('numpy')
    clip=tmp_path/'signal.mp4'
    with av.open(str(clip),'w') as out:
        stream=out.add_stream('mpeg4',rate=10);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p'
        for i in range(41):
            pixels=np.zeros((120,160,3),dtype=np.uint8)
            if i<36:pixels[:,:,0]=255
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    root=tmp_path/'results';(root/'media').mkdir(parents=True)
    detector=demo.OnnxDetector('d-fine-n',signal_model,'cpu',threads=1)
    result=demo.run_case(detector,clip,{'id':'signal'},{'checks':[{'t':1,'count':1}],'complex_label':'signal fixture; no action recognition'},root/'data/signal/d-fine-n')
    assert result['replay_verified'] and result['events']==1
    assert result['spot_checks']['exact']==1 and not result['complex_behavior']['supported']
    assert (root/'media/signal.jpg').is_file()

def test_clip_preparation_refuses_to_overwrite_its_input(tmp_path):
    spec=importlib.util.spec_from_file_location('prepare_video_demo',Path(__file__).resolve().parents[1]/'scripts/prepare_video_demo.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    video=tmp_path/'video.mp4';video.write_bytes(b'original source')
    with pytest.raises(ValueError,match='overlap'):prepare.prepare(video,video,0,1)
    assert video.read_bytes()==b'original source'

def test_export_failure_replaces_stale_completed_report(tmp_path,monkeypatch):
    import json,sys
    media=tmp_path/'media';media.mkdir();(media/'manifest.json').write_text(json.dumps({'sources':{},'cases':[]}))
    output=tmp_path/'results';output.mkdir();(output/'report.json').write_text(json.dumps({'status':'completed_execution_not_accuracy_certification'}))
    def unavailable(*args,**kwargs):raise RuntimeError('test provider initialization failure')
    monkeypatch.setattr(demo,'OnnxDetector',unavailable)
    monkeypatch.setattr(sys,'argv',['build_video_demo.py','--media-dir',str(media),'--output',str(output),'--provider','cpu'])
    with pytest.raises(RuntimeError,match='initialization failure'):demo.main()
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='error' and 'initialization failure' in report['error']
