import json
from types import SimpleNamespace
import numpy as np
import pytest
from behaviorkit.cli import run_video,next_sample_time


def write_clip(path):
    av=pytest.importorskip('av')
    with av.open(str(path),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=320;stream.height=240;stream.pix_fmt='yuv420p'
        for i in range(31):
            rgb=np.zeros((240,320,3),dtype=np.uint8)
            if i<20:rgb[:,:,0]=255
            if 8<=i<14:rgb[:,:,1]=255
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def arguments(tmp_path,signal_model):
    clip=tmp_path/'signals.mp4';write_clip(clip)
    config=tmp_path/'config.json';config.write_text(json.dumps({'schema_version':'0.1','zones':{'all':[[0,0],[1,0],[1,1],[0,1]]},'rules':[{'id':'count','kind':'count','zone':'all'},{'id':'dwell','kind':'dwell','zone':'all','seconds':1,'max_gap_seconds':.5}]}))
    return SimpleNamespace(video=str(clip),config=str(config),output=str(tmp_path/'run'),model='d-fine-n',models_dir=str(signal_model),provider='cpu',threads=1,sample_fps=5,source='signals',max_frames=0)


def test_video_to_onnx_to_tracking_count_and_dwell(signal_model,tmp_path):
    pytest.importorskip('onnxruntime')
    args=arguments(tmp_path,signal_model);result=run_video(args)
    assert result['status']=='ok' and result['sampled_frames']==16
    counts=[json.loads(line) for line in (tmp_path/'run/counts.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert max(row['count'] for row in counts)==2
    assert counts[-1]['count']==0
    assert len(events)==1 and events[0]['duration_seconds']==pytest.approx(1)
    assert events[0]['timestamp']==pytest.approx(1.2)
    assert result['provider']=='cpu'


def test_video_failure_does_not_leave_success_metadata(signal_model,tmp_path):
    pytest.importorskip('onnxruntime')
    args=arguments(tmp_path,signal_model);run_video(args)
    args.models_dir=str(tmp_path/'missing')
    with pytest.raises(FileNotFoundError):run_video(args)
    assert json.loads((tmp_path/'run/summary.json').read_text())['status']=='error'


def test_sampling_grid_handles_huge_gap_in_one_step():
    assert next_sample_time(10000000,0,5)==pytest.approx(10000000.2)
    assert next_sample_time(.4,0,5)==pytest.approx(.6)


def test_invalid_polygon_topology_rejected():
    from behaviorkit import SceneConfig
    data={'schema_version':'0.1','zones':{'all':[[0,0],[1,1],[0,1],[1,0]]},'rules':[{'id':'n','kind':'count','zone':'all'}]}
    with pytest.raises(ValueError,match='simple'):SceneConfig.from_dict(data)
