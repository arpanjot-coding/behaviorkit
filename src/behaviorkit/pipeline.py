"""Local inference capture and restartable, isolated offline file jobs."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import time
import uuid

from .contracts import EvidenceFrame, fingerprint, json_line, read_json
from .core import Observation, number, text
from .recording import cache_hit, implementation_hash, new_stage, verified_manifest
from .storage import sha256, write_json


def capture_video(video, output, adapter, *, sample_fps=5, source='camera', epoch=None,
                  geometry_revision='1', max_frames=0, cancelled=None, monitor_resources=False):
    import av
    from .cli import next_sample_time
    video=Path(video)
    if not video.is_file(): raise ValueError('capture requires a local video file')
    number(sample_fps,'sample_fps',.001,120); text(source,'source')
    if type(max_frames) is not int or max_frames<0: raise ValueError('max_frames must be nonnegative')
    epoch=text(epoch or uuid.uuid4().hex,'epoch'); text(geometry_revision,'geometry_revision')
    cap=replace(adapter.capabilities,sample_fps=sample_fps)
    if cap.stage!='postprocessed_candidates' or cap.pixels or cap.embeddings:
        raise ValueError('capture writer requires box-only candidate adapter')
    before=video.stat(); video_hash=sha256(video)
    inputs={'video_sha256':video_hash,'adapter':adapter.provenance}
    parameters={'sample_fps':sample_fps,'source':source,'epoch':epoch,'geometry_revision':geometry_revision,
                'max_frames':max_frames,'capabilities':cap.to_dict(),'decode':'PyAV CPU RGB','monitor_resources':monitor_resources,
                'transform':'adapter-declared; boxes refer to decoded image'}
    count=0; last=origin=next_due=None
    from contextlib import ExitStack
    with new_stage(output,'candidates',inputs,parameters,cap) as (out,manifest), ExitStack() as stack:
        if monitor_resources:
            from .resources import ResourceMonitor
            stack.enter_context(ResourceMonitor(out/'resources.jsonl',include_vram=True))
        with av.open(str(video)) as container, (out/'frames.jsonl').open('w',encoding='utf-8',newline='\n') as frames, (out/'timings.jsonl').open('w',encoding='utf-8',newline='\n') as timings:
            start=time.perf_counter()
            for decoded in container.decode(video=0):
                if cancelled and cancelled(): raise InterruptedError('archive job cancelled')
                if decoded.pts is None or decoded.time_base is None: raise ValueError('video lacks source PTS')
                timestamp=float(decoded.pts*decoded.time_base)
                if timestamp<0: continue
                number(timestamp,'video timestamp')
                if last is not None and timestamp<=last: raise ValueError('video timestamps must strictly increase')
                last=timestamp
                if next_due is not None and timestamp+1e-9<next_due: continue
                if origin is None: origin=timestamp
                next_due=next_sample_time(timestamp,origin,sample_fps)
                decode_end=time.perf_counter(); rgb=decoded.to_ndarray(format='rgb24'); converted=time.perf_counter()
                detections,timing=adapter.predict(rgb)
                if len(detections)>cap.max_detections or any(d.label not in cap.classes or d.confidence<cap.confidence_floor-1e-9 for d in detections):
                    raise ValueError('adapter output violates capture capability contract')
                frame=EvidenceFrame(Observation(timestamp,detections,source,epoch),decoded.width,decoded.height,geometry_revision)
                write_begin=time.perf_counter(); json_line(frames,frame.to_dict())
                json_line(timings,{'timestamp':timestamp,'decode_sampling_ms':(decode_end-start)*1000,
                                  'rgb_conversion_ms':(converted-decode_end)*1000,**timing,
                                  'serialization_ms':(time.perf_counter()-write_begin)*1000,
                                  'capture_pipeline_ms':(time.perf_counter()-start)*1000})
                count+=1
                if max_frames and count>=max_frames: break
                start=time.perf_counter()
        if not count: raise ValueError('video produced no sampled frames')
        after=video.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or video_hash!=sha256(video):
            raise ValueError('video changed during inference capture')
        manifest['frames']=count
    return manifest


def batch_videos(videos, config, output, adapter, *, sample_fps=5, tracker_parameters=None,
                 cancel_file=None, retry_failed=False, monitor_resources=False):
    """Resume at complete stage boundaries; partial files restart from frame zero.

    No unsupported tracker/rule checkpoint is ever restored. A stable per-job
    source epoch lets a consumer deduplicate a restarted file's completed events.
    """
    from .advanced import parse_scene
    from .recording import apply_rules, retrack
    videos=tuple(Path(p).resolve() for p in videos)
    if not 1<=len(videos)<=10000: raise ValueError('batch requires 1..10000 files')
    if len(set(videos))!=len(videos): raise ValueError('duplicate batch input files')
    scene=parse_scene(read_json(config) if isinstance(config,(str,Path)) else config)
    adapter.capabilities.require(classes={r.label for r in scene.rules},confidence=min(r.confidence for r in scene.rules))
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    settings={'config':scene.to_dict(),'adapter':adapter.provenance,'capabilities':adapter.capabilities.to_dict(),
              'sample_fps':sample_fps,'tracker':tracker_parameters or {'adapter':'iou','min_hits':2},'monitor_resources':monitor_resources}
    request_hash=fingerprint({'files':[str(p) for p in videos],'settings':settings})
    jobfile=root/'job.json'
    if jobfile.exists():
        job=read_json(jobfile,16*1024*1024)
        if job.get('schema_version')!='batch/1' or job.get('request_hash')!=request_hash:
            raise ValueError('batch fingerprints differ; use a new output directory')
    else:
        if any(root.iterdir()): raise ValueError('batch output directory is not owned/empty')
        job={'schema_version':'batch/1','id':uuid.uuid4().hex,'request_hash':request_hash,'status':'running',
             'settings':settings,'files':[{'input':str(p),'status':'pending','attempts':[]} for p in videos]}
    cancelled=lambda: cancel_file is not None and Path(cancel_file).exists()
    for index,item in enumerate(job['files']):
        if cancelled(): job['status']='cancelled'; break
        video=videos[index]
        try:
            digest=sha256(video)
            if item.get('input_sha256') and item['input_sha256']!=digest:
                raise ValueError('input fingerprint changed; use a new batch')
            item['input_sha256']=digest
            if item['status']=='complete':
                finished=verified_manifest(root/item['result'],kind='rules')
                if finished['implementation_sha256']==implementation_hash('rules'): continue
                item['status']='pending'
            if item['status']=='error' and not retry_failed: continue
            attempt=root/f'file-{index:05d}-attempt-{len(item["attempts"])+1:03d}'
            item['attempts'].append(attempt.name); item['status']='running'
            job['status']='running'; write_json(jobfile,job)
            attempt.mkdir()
            epoch=fingerprint([job['id'],index,digest])
            candidates=None
            # A finished perception stage can survive a later rule/storage error.
            for old in reversed(item['attempts'][:-1]):
                try:
                    candidate=root/old/'candidates'; manifest=verified_manifest(candidate,kind='candidates')
                    if manifest['inputs']['video_sha256']==digest and manifest['implementation_sha256']==implementation_hash('candidates'):
                        candidates=candidate; break
                except (OSError,ValueError,KeyError): pass
            if candidates is None:
                candidates=attempt/'candidates'
                capture_video(video,candidates,adapter,sample_fps=sample_fps,source=f'file-{index}',epoch=epoch,cancelled=cancelled,monitor_resources=monitor_resources)
            retrack(candidates,attempt/'tracks',tracker_parameters=settings['tracker'])
            apply_rules(attempt/'tracks',scene,attempt/'rules')
            item.update(status='complete',result=str((attempt/'rules').relative_to(root)),candidate_stage=str(candidates.relative_to(root)))
        except InterruptedError:
            item['status']='interrupted'; job['status']='cancelled'; write_json(jobfile,job); break
        except Exception as error:
            item.update(status='error',error=f'{type(error).__name__}: {error}')
        write_json(jobfile,job)
    else:
        job['status']='complete' if all(f['status']=='complete' for f in job['files']) else 'partial_failure'
    write_json(jobfile,job)
    return job
