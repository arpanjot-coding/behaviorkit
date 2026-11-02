"""Run the real local pipeline and export replayable website evidence.

The website replays these results; it does not run inference in the browser.
Dwell explanations use the public state-inspection API without modifying state.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,statistics

from behaviorkit import BehaviorEngine,SceneConfig,Observation,__version__
from behaviorkit.tracking import IoUTracker
from behaviorkit.cli import replay,next_sample_time
from behaviorkit.storage import write_json,sha256
from behaviorkit.benchmark import environment
from behaviorkit.vision import OnnxDetector,PRESETS

ROOT=Path(__file__).resolve().parents[1]
SCENE={'schema_version':'0.1','zones':{'view':[[0,0],[1,0],[1,1],[0,1]]},'rules':[{'id':'visible_people','kind':'count','zone':'view','confidence':.3},{'id':'observed_dwell','kind':'dwell','zone':'view','confidence':.3,'seconds':3,'max_gap_seconds':.5}]}

def explain(engine,observation):
    previous={state['track_id'] for state in engine.dwell_states(observation.source)}
    result=engine.process(observation)
    states=[{name:state[name] for name in ('track_id','start','last','duration','emitted')} for state in engine.dwell_states(observation.source)]
    active={s['track_id'] for s in states}
    return result,states,sorted((previous-active)|{t['track_id'] for t in result.transitions if t['transition']=='reset'})

def score_checks(frames,checks,tolerance=.11):
    result=[]
    for check in checks:
        row=dict(check)
        if check['count'] is None:row['status']='human_abstained'
        else:
            sample=min(frames,key=lambda f:abs(f['timestamp']-check['t'])) if frames else None
            if sample is None or abs(sample['timestamp']-check['t'])>tolerance:row['status']='no_nearby_sample'
            else:row.update(status='scored',sample_timestamp=sample['timestamp'],observed=sample['count']['count'],absolute_error=abs(sample['count']['count']-check['count']))
        result.append(row)
    scored=[r for r in result if r['status']=='scored']
    return {'checks':result,'scored':len(scored),'exact':sum(r['absolute_error']==0 for r in scored),'mean_absolute_error':statistics.mean(r['absolute_error'] for r in scored) if scored else None,'note':'Sparse exploratory count snapshots, not clip-wide accuracy or event precision/recall.'}

def run_case(detector,video,case,annotations,output):
    import av
    config=SceneConfig.from_dict(SCENE);engine=BehaviorEngine(config,trace=True);tracker=IoUTracker(max_tracks=config.max_tracks)
    output.mkdir(parents=True,exist_ok=True);write_json(output/'config.json',SCENE)
    frames=[];all_events=[];next_due=None;origin=None
    with av.open(str(video)) as container,(output/'observations.jsonl').open('w',encoding='utf-8') as observations:
        previous=None
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:raise ValueError('timestamped frames required')
            t=float(frame.pts*frame.time_base)
            if previous is not None and t<=previous:raise ValueError('timestamps must strictly increase')
            previous=t
            if next_due is not None and t+1e-9<next_due:continue
            if origin is None:origin=t
            next_due=next_sample_time(t,origin,5)
            pixels=frame.to_ndarray(format='rgb24');raw,timing=detector.predict(pixels);tracked=tracker.update(raw,t)
            o=Observation(t,tracked,case['id'],'0');observations.write(json.dumps(asdict(o),allow_nan=False)+'\n')
            evaluated,states,resets=explain(engine,o);all_events.extend(evaluated.events)
            frames.append({'timestamp':t,'raw':[asdict(d) for d in raw],'tracks':[asdict(d) for d in tracked],'count':evaluated.counts[0],'dwell':states,'resets':resets,'events':list(evaluated.events),'timings_ms':timing})
            if len(frames)==1:
                import cv2
                if not cv2.imwrite(str(output.parents[2]/'media'/(case['id']+'.jpg')),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85]):
                    raise OSError('could not write poster image')
    # Independently replay the exact saved observations through the public CLI path.
    replay_summary=replay(output/'observations.jsonl',output/'config.json',output/'replay')
    saved_events=[json.loads(s) for s in (output/'replay/events.jsonl').read_text().splitlines()]
    saved_counts=[json.loads(s) for s in (output/'replay/counts.jsonl').read_text().splitlines()]
    assert json.loads(json.dumps(all_events))==saved_events,'event replay mismatch'
    assert [f['count'] for f in frames]==saved_counts,'count replay mismatch'
    report={'case':case['id'],'model':detector.name,'provider':detector.selected_provider,'status':'executed','sampled_frames':len(frames),'events':len(all_events),'count_min':min(f['count']['count'] for f in frames),'count_max':max(f['count']['count'] for f in frames),'track_ids_seen':len({d['track_id'] for f in frames for d in f['tracks']}),'reset_records':sum(len(f['resets']) for f in frames),'p50_detector_ms':statistics.median(f['timings_ms']['total_ms'] for f in frames),'spot_checks':score_checks(frames,annotations['checks']),'complex_behavior':{'supported':False,'library_prediction':None,'human_reference':annotations['complex_label'],'note':'Count/dwell outputs do not predict theft, concealment, payment, ownership or intent.'},'replay_verified':True,'model_sha256':detector.manifest['sha256'],'observations_sha256':replay_summary['observations_sha256']}
    write_json(output/'frames.json',{'summary':report,'frames':frames,'events':saved_events,'config':SCENE,'human_annotations':annotations})
    write_json(output/'summary.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--media-dir',type=Path,default=ROOT/'examples/video-lab/media');p.add_argument('--models-dir',type=Path,default=ROOT/'models');p.add_argument('--provider',choices=['cpu','cuda','coreml'],default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--output',type=Path,default=ROOT/'examples/video-lab/results');a=p.parse_args()
    manifest=json.loads((a.media_dir/'manifest.json').read_text(encoding='utf-8'));annotations=json.loads((ROOT/'examples/video-lab/annotations.json').read_text(encoding='utf-8'))
    (a.output/'media').mkdir(parents=True,exist_ok=True)
    report={'version':__version__,'kind':'exploratory_video_lab','status':'completed_execution_not_accuracy_certification','environment':environment(a.provider=='cuda'),'sample_fps':5,'confidence':.3,'dwell_seconds':3,'max_gap_seconds':.5,'threads':a.threads,'provider':a.provider,'config':SCENE,'annotation_protocol':annotations['protocol'],'sources':manifest['sources'],'cases':manifest['cases'],'models':list(PRESETS),'runs':[],'verification':{},'limits':['Four excerpts from three sources; the two film excerpts are not independent videos.','Fictional historical theft is a domain-shift stress case, not modern CCTV accuracy.','Manual spot checks are sparse and single-reviewer; ambiguous views are explicitly unscored.','No action/theft classifier is implemented. Dwell events do not establish crime.','All inference is precomputed locally. Browser playback has no model/LLM/API inference calls.']}
    report['status']='running';write_json(a.output/'report.json',report)
    try:
        for model in PRESETS:
            detector=OnnxDetector(model,a.models_dir,a.provider,threads=a.threads)
            for case in manifest['cases']:
                video=a.media_dir/(case['id']+'.mp4')
                if sha256(video)!=case['prepared']['sha256']:raise ValueError('prepared video hash mismatch')
                run=run_case(detector,video,case,annotations['cases'][case['id']],a.output/'data'/case['id']/model)
                report['runs'].append(run);print(model,case['id'],run['sampled_frames'],run['events'],run['spot_checks']['exact'],run['spot_checks']['scored'],flush=True)
            detector=None
            # Separate session: profile verification does not contaminate timed sessions.
            import av
            with av.open(str(a.media_dir/'shelf.mp4')) as c:rgb=next(c.decode(video=0)).to_ndarray(format='rgb24')
            verifier=OnnxDetector(model,a.models_dir,a.provider,threads=a.threads,profile=True,profile_dir=a.output/'profiles');verifier.predict(rgb);profile=verifier.end_profile()
            if not profile.get('requested_provider_observed'):raise RuntimeError('requested execution provider not observed')
            profile.pop('file',None);report['verification'][model]=profile;verifier=None
    except Exception as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}')
        write_json(a.output/'report.json',report);raise
    report['status']='completed_execution_not_accuracy_certification'
    write_json(a.output/'report.json',report)
    print('report',a.output/'report.json',flush=True)
if __name__=='__main__':main()
