"""Versioned system commands, separated from the stable v0.3 CLI."""
from dataclasses import replace
from pathlib import Path
import time

from .contracts import Capabilities, EvidenceFrame, fingerprint, json_line, read_json, read_jsonl
from .storage import write_json

COMMANDS={'capture','track','apply','evaluate','inspect','config-diff','compare','recording-info',
          'batch','live','setup','store','outbox','plan-budget'}


def _detector_options(parser):
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model',choices=['rtmdet-tiny','d-fine-n','rf-detr-nano'])
    group.add_argument('--custom-model',help='Local ONNX model with an explicit --contract JSON')
    parser.add_argument('--contract'); parser.add_argument('--models-dir',default='models')
    parser.add_argument('--provider',choices=['auto','cpu','cuda','coreml','tensorrt'],default='auto')
    parser.add_argument('--classes',nargs='+',default=['person']); parser.add_argument('--capture-floor',type=float,default=.1)
    parser.add_argument('--threads',type=int,default=0); parser.add_argument('--sample-fps',type=float,default=5)
    parser.add_argument('--monitor-resources',action='store_true',help='Optional psutil RSS/native resource log plus NVIDIA process-memory query')


def register(sub):
    cap=sub.add_parser('capture',help='Record versioned post-NMS boxes once for layered replay')
    cap.add_argument('--video',required=True); cap.add_argument('--output',required=True)
    cap.add_argument('--source',default='camera'); cap.add_argument('--epoch'); cap.add_argument('--geometry-revision',default='1')
    cap.add_argument('--max-frames',type=int,default=0); _detector_options(cap)
    tr=sub.add_parser('track',help='Recompute tracking using saved candidates, without inference')
    tr.add_argument('--recording',required=True); tr.add_argument('--output',required=True)
    tr.add_argument('--tracker',choices=['iou','bytetrack'],default='iou'); tr.add_argument('--parameters',help='Tracker settings JSON')
    ap=sub.add_parser('apply',help='Apply scene 0.1 or 0.2 to verified tracked evidence')
    ap.add_argument('--recording',required=True)
    config=ap.add_mutually_exclusive_group(required=True); config.add_argument('--config'); config.add_argument('--saved-bindings',action='store_true')
    ap.add_argument('--output',required=True); ap.add_argument('--trace',action='store_true')
    ev=sub.add_parser('evaluate',help='Score events/counts against explicit covered annotations')
    ev.add_argument('--annotations',required=True); ev.add_argument('--events',required=True); ev.add_argument('--counts')
    ev.add_argument('--start-tolerance',type=float,default=.5); ev.add_argument('--min-interval-iou',type=float,default=0); ev.add_argument('--output',required=True)
    inspect=sub.add_parser('inspect',help='Explain retained evidence with bounded detail')
    inspect.add_argument('--recording',required=True); inspect.add_argument('--config',required=True); inspect.add_argument('--output',required=True)
    for name in ('source','rule','track'): inspect.add_argument('--'+name)
    inspect.add_argument('--start',type=float); inspect.add_argument('--end',type=float); inspect.add_argument('--detail-limit',type=int,default=100)
    for name in ('config-diff','compare'):
        p=sub.add_parser(name,help='Compare configuration semantics'+(' and replay both revisions' if name=='compare' else ''))
        p.add_argument('--before',required=True); p.add_argument('--after',required=True); p.add_argument('--output',required=True)
        if name=='compare': p.add_argument('--recording',required=True)
    info=sub.add_parser('recording-info',help='Verify completed recording integrity and show capabilities'); info.add_argument('--recording',required=True)
    batch=sub.add_parser('batch',help='Process independent files, reusing completed stages on restart')
    batch.add_argument('--videos',nargs='+',required=True); batch.add_argument('--config',required=True); batch.add_argument('--output',required=True)
    batch.add_argument('--cancel-file'); batch.add_argument('--retry-failed',action='store_true'); _detector_options(batch)
    live=sub.add_parser('live',help='Experimental bounded multi-source PyAV ingest with source health')
    live.add_argument('--sources',required=True,help='Source/binding JSON; keep credential-bearing files private')
    live.add_argument('--output',required=True); live.add_argument('--seconds',type=float,default=60); live.add_argument('--outbox'); _detector_options(live)
    live.add_argument('--tracking-floor',type=float,help='Independent IoU association floor; defaults to max(0.3,capture floor)')
    setup=sub.add_parser('setup',help='Start the loopback-only zone drawing wizard'); setup.add_argument('--port',type=int,default=8765)
    store=sub.add_parser('store',help='Manage only artifacts copied into an owned store')
    store.add_argument('--root',required=True); store.add_argument('--add'); store.add_argument('--kind',choices=['metadata','trace','recording','image'],default='metadata')
    store.add_argument('--retention-seconds',type=float); store.add_argument('--pinned',action='store_true')
    store.add_argument('--max-bytes',type=int); store.add_argument('--max-trace-bytes',type=int); store.add_argument('--protect',nargs='*',default=[]); store.add_argument('--apply',action='store_true')
    outbox=sub.add_parser('outbox',help='Inspect/retry the durable local JSONL outbox')
    outbox.add_argument('--database',required=True); outbox.add_argument('--events'); outbox.add_argument('--deliver-jsonl'); outbox.add_argument('--limit',type=int,default=100)
    budget=sub.add_parser('plan-budget',help='Choose a measured hardware/dataset profile meeting explicit budgets')
    budget.add_argument('--profiles',required=True); budget.add_argument('--budgets',required=True); budget.add_argument('--output',required=True)


def _adapter(args):
    from .adapters import BoxOnnxAdapter, PresetAdapter
    if args.custom_model:
        if not args.contract: raise ValueError('--custom-model requires --contract')
        return BoxOnnxAdapter(args.custom_model,read_json(args.contract),provider=args.provider,
                              confidence_floor=args.capture_floor,threads=args.threads)
    if args.contract: raise ValueError('--contract is only used with --custom-model')
    return PresetAdapter(args.model,args.models_dir,args.provider,classes=tuple(args.classes),
                         confidence_floor=args.capture_floor,threads=args.threads)


def _write_result(path, result, inputs):
    from .storage import guard_outputs
    target=Path(path)
    guard_outputs(target.parent,inputs,[target.name])
    if target.exists(): raise ValueError('result output already exists')
    write_json(target,result)
    return result


def dispatch(args):
    from .advanced import config_diff, make_engine, parse_scene
    from .recording import apply_rules, retrack, verified_manifest
    command=args.command
    if command=='setup':
        from .wizard import serve
        serve(args.port); return {'status':'stopped'}
    if command=='capture':
        from .pipeline import capture_video
        return capture_video(args.video,args.output,_adapter(args),sample_fps=args.sample_fps,source=args.source,
                             epoch=args.epoch,geometry_revision=args.geometry_revision,max_frames=args.max_frames,monitor_resources=args.monitor_resources)
    if command=='track':
        params=read_json(args.parameters) if args.parameters else {}
        if not isinstance(params,dict): raise ValueError('tracker parameters must be an object')
        if args.tracker=='iou': return retrack(args.recording,args.output,tracker_parameters={'adapter':'iou',**params})
        from .motion import ByteTrackAdapter
        cap=Capabilities(**verified_manifest(args.recording,kind='candidates')['capabilities'])
        # ByteTrack low-score recovery cannot recover boxes discarded at capture.
        if cap.confidence_floor>.1+1e-9: raise ValueError('ByteTrack comparison requires capture floor <=0.1')
        settings={'classes':cap.classes,'sample_fps':cap.sample_fps or 5,**params}
        return retrack(args.recording,args.output,tracker_factory=lambda:ByteTrackAdapter(**settings),
                       tracker_parameters={'adapter':'trackers-bytetrack/1','settings':settings})
    if command=='apply': return apply_rules(args.recording,args.config,args.output,trace=args.trace)
    if command=='recording-info': return verified_manifest(args.recording)
    if command=='evaluate':
        from .evaluation import evaluate
        result=evaluate(read_json(args.annotations,64*1024*1024),{'events':list(read_jsonl(args.events)),
                        'counts':list(read_jsonl(args.counts)) if args.counts else []},
                        start_tolerance=args.start_tolerance,min_interval_iou=args.min_interval_iou)
        return _write_result(args.output,result,[args.annotations,args.events]+([args.counts] if args.counts else []))
    if command=='inspect':
        from .setup import Inspector
        manifest=verified_manifest(args.recording); scene=parse_scene(read_json(args.config))
        if manifest['kind'] not in {'candidates','tracks'}: raise ValueError('inspection requires candidate or tracked evidence')
        Capabilities(**manifest['capabilities']).require(classes={r.label for r in scene.rules},confidence=min(r.confidence for r in scene.rules))
        candidates=manifest['kind']=='candidates'
        inspector=Inspector(scene,source=args.source,rule=args.rule,track=args.track,start=args.start,end=args.end,detail_limit=args.detail_limit,candidate_stage=candidates)
        engine=make_engine(scene,trace=True)
        for row in read_jsonl(Path(args.recording)/'frames.jsonl'):
            frame=EvidenceFrame.from_dict(row); result=None if candidates else engine.process(frame.confirmed_observation())
            inspector.observe(frame,result,engine.dwell_states())
        result=inspector.report(); result['recording_cache_key']=manifest['cache_key']
        return _write_result(args.output,result,[args.config,*[Path(args.recording)/n for n in manifest['artifacts']]])
    if command in {'config-diff','compare'}:
        before,after=read_json(args.before),read_json(args.after); result=config_diff(before,after)
        if command=='config-diff': return _write_result(args.output,result,[args.before,args.after])
        root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
        write_json(root/'comparison.json',{'status':'running',**result})
        try:
            a=apply_rules(args.recording,before,root/'before',trace=True)
            b=apply_rules(args.recording,after,root/'after',trace=True)
        except BaseException as error:
            write_json(root/'comparison.json',{'status':'error','error':f'{type(error).__name__}: {error}',**result})
            raise
        from .advanced import identity_migration
        write_json(root/'identity-migration.json',identity_migration(read_jsonl(root/'before/events.jsonl'),read_jsonl(root/'after/events.jsonl')))
        result['status']='complete'
        result.update(before_totals=a['totals'],after_totals=b['totals'],
                      note='Offline replay only. Original files remain immutable. Roll back by applying the earlier config to a new output stage.')
        write_json(root/'comparison.json',result); return result
    if command=='batch':
        from .pipeline import batch_videos
        return batch_videos(args.videos,args.config,args.output,_adapter(args),sample_fps=args.sample_fps,
                            cancel_file=args.cancel_file,retry_failed=args.retry_failed,monitor_resources=args.monitor_resources)
    if command=='store':
        from .owned_store import ArtifactStore
        store=ArtifactStore(args.root)
        if args.add: return store.put(args.add,kind=args.kind,retention_seconds=args.retention_seconds,pinned=args.pinned)
        return store.prune(max_bytes=args.max_bytes,max_trace_bytes=args.max_trace_bytes,protected_paths=args.protect,apply=args.apply)
    if command=='outbox':
        from .delivery import Outbox,JsonlSink
        from .storage import guard_outputs
        guard_outputs(Path(args.database).parent,[args.events] if args.events else [],[Path(args.database).name])
        if args.deliver_jsonl:
            guard_outputs(Path(args.deliver_jsonl).parent,[args.database]+([args.events] if args.events else []),[Path(args.deliver_jsonl).name])
        box=Outbox(args.database)
        if args.events:
            for event in read_jsonl(args.events): box.enqueue(event)
        if args.deliver_jsonl: return {**box.dispatch(JsonlSink(args.deliver_jsonl),limit=args.limit),**box.status()}
        return box.status()
    if command=='plan-budget':
        from .profiles import select_profile
        return _write_result(args.output,select_profile(read_json(args.profiles),**read_json(args.budgets)),[args.profiles,args.budgets])
    if command=='live': return run_live(args)
    raise ValueError('unknown system command')


def run_live(args):
    from contextlib import ExitStack
    from .advanced import parse_scene
    from .core import Observation, number
    from .delivery import Outbox
    from .live import LiveSources,SourceSpec
    from .recording import new_stage
    from .setup import Geometry,SceneRouter
    from .tracking import IoUTracker
    number(args.seconds,'seconds',.01,86400)
    data=read_json(args.sources); bindings={}; specs=[]
    for row in data['sources']:
        row=dict(row); scene_path=Path(row.pop('scene'))
        if not scene_path.is_absolute(): scene_path=Path(args.sources).parent/scene_path
        scene=parse_scene(read_json(scene_path)); geometry=Geometry(**row.pop('geometry'))
        spec=SourceSpec(**row)
        if spec.id in bindings: raise ValueError('duplicate source binding')
        specs.append(spec); bindings[spec.id]=(scene,geometry)
    adapter=_adapter(args)
    floor=getattr(args,'tracking_floor',None)
    floor=max(.3,adapter.capabilities.confidence_floor) if floor is None else floor
    adapter.capabilities.require(classes=adapter.capabilities.classes,confidence=floor)
    for spec in specs:
        scene=bindings[spec.id][0]
        replace(adapter.capabilities,confidence_floor=floor).require(classes={r.label for r in scene.rules},confidence=min(r.confidence for r in scene.rules))
        if any(r.kind!='count' and r.max_gap_seconds+1e-9<1/spec.sample_fps for r in scene.rules):
            raise ValueError('source cadence cannot support rule gap')
    router=SceneRouter(bindings,max_sources=16,trace=True); trackers={}; epochs={}; segments={}
    cap=replace(adapter.capabilities,stage='tracked_observations',sample_fps=None,confidence_floor=floor)
    parameters={'sources':[{'id':s.id,'queue_policy':s.policy,'queue_size':s.queue_size,'sample_fps':s.sample_fps,
                            'geometry':bindings[s.id][1].fingerprint,'scene':bindings[s.id][0].to_dict()} for s in specs],
                'duration_seconds':args.seconds,'status':'experimental','tracker':'iou/1; min_hits=2','tracking_confidence_floor':floor}
    outbox=Outbox(args.outbox) if args.outbox else None
    totals={'frames':0,'events':0}; begin=time.monotonic()
    with new_stage(args.output,'tracks',{'adapter':adapter.provenance},parameters,cap) as (out,manifest), ExitStack() as stack:
        if getattr(args,'monitor_resources',False):
            from .resources import ResourceMonitor
            stack.enter_context(ResourceMonitor(out/'resources.jsonl',include_vram=True))
        streams={name:stack.enter_context((out/(name+'.jsonl')).open('w',encoding='utf-8',newline='\n')) for name in ('frames','events','counts','health','timings')}
        live=stack.enter_context(LiveSources(specs))
        while time.monotonic()-begin<args.seconds and not live.finished:
            packet=live.poll()
            if packet is None: continue
            source=packet['source']
            if packet['kind']=='health':
                json_line(streams['health'],packet)
                if packet['state']!='healthy':
                    router.close_source(source); trackers.pop(source,None); epochs.pop(source,None)
                continue
            rgb=packet.pop('rgb'); t=time.perf_counter(); detections,timing=adapter.predict(rgb)
            # Frames can become stale during expensive inference, too.
            spec=next(s for s in specs if s.id==source)
            age=time.monotonic()-packet['received_monotonic']
            if age>spec.stale_seconds:
                json_line(streams['health'],{'source':source,'state':'stale','reason':'inference_evidence_age','age_seconds':age})
                router.close_source(source); trackers.pop(source,None); epochs.pop(source,None); continue
            geometry=bindings[source][1]
            if epochs.get(source)!=packet['epoch']:
                trackers[source]=IoUTracker(max_tracks=bindings[source][0].max_tracks); epochs[source]=packet['epoch']
                import uuid
                segments[source]=fingerprint([packet['epoch'],uuid.uuid4().hex])
            tracking_start=time.perf_counter(); tracks=trackers[source].update(tuple(d for d in detections if d.confidence>=floor),packet['timestamp'])
            frame=EvidenceFrame(Observation(packet['timestamp'],tracks,source,segments[source]),rgb.shape[1],rgb.shape[0],geometry.revision,packet['clock'])
            rule_start=time.perf_counter(); result=router.process(frame); evaluated=time.perf_counter()
            json_line(streams['frames'],router.prepare_frame(frame).to_dict())
            for event in result.events:
                json_line(streams['events'],event)
                if outbox: outbox.enqueue(event)
            for count in result.counts: json_line(streams['counts'],count)
            totals['frames']+=1; totals['events']+=len(result.events)
            json_line(streams['timings'],{'source':source,'timestamp':packet['timestamp'],**timing,
                                         'tracking_ms':(rule_start-tracking_start)*1000,'rules_ms':(evaluated-rule_start)*1000,
                                         'pipeline_ms':(time.perf_counter()-t)*1000,'evidence_age_seconds':age,
                                         'age_basis':'local decoded frame receipt; camera capture time unknown',
                                         'queue_age_seconds':packet['queue_age_seconds'],'dropped_frames':packet['dropped_frames']})
            for stream in streams.values(): stream.flush()
        health=live.status()
        outcome='failed' if not totals['frames'] else 'partial_failure' if any(s['state']=='failed' for s in health.values()) else 'completed'
        manifest.update(totals=totals,source_health=health,frames=totals['frames'],run_status=outcome,
                        termination_reason='sources_ended' if live.finished else 'time_budget')
    return manifest
