"""Positive real-model regression on a repeated single-person still image.

This controlled fixture is not a surveillance accuracy dataset. No downloads or
LLM calls occur here. Supply a rights-cleared image containing exactly one person.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import av

from behaviorkit.benchmark import environment
from behaviorkit.cli import run_video,replay
from behaviorkit.providers import CHOICES
from behaviorkit.storage import sha256,write_json
from behaviorkit.vision import OnnxDetector,PRESETS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--image-source',default='user-supplied; no image license asserted by this script')
    p.add_argument('--models-dir',type=Path,default=Path('models'))
    p.add_argument('--provider',choices=CHOICES,default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,default=Path('runs/positive-validation'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    bgr=cv2.imread(str(a.image))
    if bgr is None:raise ValueError('image could not be decoded')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    image=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
    clip=a.output/'single-person-still.mp4'
    with av.open(str(clip),'w') as container:
        stream=container.add_stream('mpeg4',rate=10);stream.width=512;stream.height=512;stream.pix_fmt='yuv420p';stream.bit_rate=2000000
        for index in range(51):
            pixels=image if index<42 else np.zeros_like(image)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    config=a.output/'scene.json'
    write_json(config,{'schema_version':'0.1','zones':{'all':[[0,0],[1,0],[1,1],[0,1]]},'rules':[{'id':'count','kind':'count','zone':'all'},{'id':'dwell','kind':'dwell','zone':'all','seconds':3,'max_gap_seconds':.5}]})
    report={'kind':'controlled_single_person_regression','environment':environment(a.provider in {'auto','cuda','tensorrt'}),'provider':a.provider,'image_sha256':sha256(a.image),'image_source':a.image_source,'conditions':'One still person repeated for 4.2 seconds, followed by blank frames; 5 Hz processing. Not real motion, occlusion or general camera accuracy.','models':[]}
    for name in PRESETS:
        result={'model':name}
        try:
            detector=OnnxDetector(name,a.models_dir,a.provider,threads=a.threads)
            actual,_=detector.predict(rgb)
            if len(actual)!=1:raise AssertionError(f'expected one detected person, found {len(actual)}')
            result['static_image']=[{'confidence':d.confidence,'box':d.box} for d in actual]
            selected=detector.selected_provider
            if selected!='cpu':
                reference=OnnxDetector(name,a.models_dir,'cpu',threads=a.threads)
                expected,_=reference.predict(rgb)
                if len(expected)!=1:raise AssertionError('CPU reference did not detect one person')
                box_error=float(np.max(np.abs(np.asarray(actual[0].box)-np.asarray(expected[0].box))))
                score_error=abs(actual[0].confidence-expected[0].confidence)
                if box_error>.005 or score_error>.005:raise AssertionError(f'CPU/provider disagreement: box {box_error}, score {score_error}')
                result['cpu_agreement']={'max_normalized_box_error':box_error,'confidence_error':score_error,'tolerance':.005}
                reference=None
            detector=None
            args=argparse.Namespace(video=str(clip),config=str(config),model=name,models_dir=str(a.models_dir),provider=a.provider,threads=a.threads,sample_fps=5,source='single-person-still',max_frames=0,trace=True,output=str(a.output/name))
            summary=run_video(args)
            folder=a.output/name
            counts=[json.loads(line) for line in (folder/'counts.jsonl').read_text().splitlines()]
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            if max(c['count'] for c in counts)!=1 or counts[-1]['count']!=0:raise AssertionError('visible-count expectation failed')
            if len(events)!=1 or abs(events[0]['timestamp']-3.2)>.21:raise AssertionError(f'dwell expectation failed: {events}')
            replayed=replay(folder/'observations.jsonl',folder/'config.snapshot.json',folder/'replayed',trace=True)
            for stream_name in ('counts','events','transitions'):
                if (folder/(stream_name+'.jsonl')).read_bytes()!=(folder/'replayed'/(stream_name+'.jsonl')).read_bytes():raise AssertionError(f'{stream_name} replay differed')
            result.update(status='passed',sampled_frames=summary['sampled_frames'],events=len(events),dwell_timestamp=events[0]['timestamp'],selected_provider=summary['provider'],exact_replay=True,config_fingerprint=summary['config_fingerprint'],trace_records=summary['transition_records'])
        except Exception as error:result.update(status='failed',error=f'{type(error).__name__}: {error}')
        report['models'].append(result)
    write_json(a.output/'validation.json',report)
    print(json.dumps(report,indent=2))
    return int(any(item['status']!='passed' for item in report['models']))


if __name__=='__main__':raise SystemExit(main())
