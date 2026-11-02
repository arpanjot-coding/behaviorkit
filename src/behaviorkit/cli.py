from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

from . import __version__
from .core import BehaviorEngine, Detection, Observation, SceneConfig, number, text
from .storage import write_json, guard_outputs, sha256
from .providers import CHOICES, select_provider


def load_config(path):
    return load_config_with_hash(path)[0]


def load_config_with_hash(path):
    config,digest,_=load_config_snapshot(path)
    return config,digest


def load_config_snapshot(path):
    with Path(path).open('rb') as source:raw=source.read(1024*1024+1)
    if len(raw)>1024*1024:raise ValueError("configuration exceeds 1 MiB")
    value=json.loads(raw)
    from .advanced import parse_scene
    return parse_scene(value),hashlib.sha256(raw).hexdigest(),value


RESULT_FILES=('events','counts','transitions','diagnostics')
OUTPUT_NAMES=tuple(name+'.jsonl' for name in RESULT_FILES)+('summary.json','config.snapshot.json')


def open_results(stack,output):
    return {name:stack.enter_context((output/(name+'.jsonl')).open('w',encoding='utf-8',newline='\n')) for name in RESULT_FILES}


def write_evaluation(result,observation,streams):
    rows={'events':result.events,'counts':result.counts,'transitions':result.transitions,
          'diagnostics':({'source':observation.source,'epoch':observation.epoch,'timestamp':observation.timestamp,'messages':result.diagnostics},) if result.diagnostics else ()}
    for name,values in rows.items():
        for value in values:streams[name].write(json.dumps(value,allow_nan=False)+'\n')
        if values:streams[name].flush()
    return {name:len(values) for name,values in rows.items()}


def replay(observations, config, output, *, trace=False):
    output=Path(output)
    guard_outputs(output,[observations,config],OUTPUT_NAMES)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',{"schema_version":"0.1","status":"running","kind":"replay"})
    counts=events=batches=line_number=transitions=diagnostics=0
    digest=hashlib.sha256()
    try:
        scene,config_hash,snapshot=load_config_snapshot(config)
        from .advanced import make_engine
        engine=make_engine(scene,trace=trace)
        write_json(output/'config.snapshot.json',snapshot)
        with ExitStack() as stack:
            source=stack.enter_context(Path(observations).open('rb'));streams=open_results(stack,output)
            while True:
                line=source.readline(8*1024*1024+1)
                if not line:break
                line_number+=1
                if len(line)>8*1024*1024:raise ValueError(f"observation line {line_number} exceeds 8 MiB")
                digest.update(line)
                if not line.strip():continue
                try:
                    observation=Observation.from_dict(json.loads(line))
                    result=engine.process(observation)
                except (KeyError,TypeError,ValueError) as e:
                    raise ValueError(f"observation line {line_number}: {e}") from e
                records=write_evaluation(result,observation,streams)
                events+=records['events'];counts+=records['counts'];transitions+=records['transitions'];diagnostics+=records['diagnostics']
                batches+=1
    except Exception as error:
        write_json(output/'summary.json',{"schema_version":"0.1","kind":"replay","status":"error","error":f"{type(error).__name__}: {error}","line":line_number,"batches":batches,"events":events,"count_samples":counts,"partial_outputs":True})
        raise
    summary={"schema_version":"0.1","behaviorkit":__version__,"status":"ok","kind":"replay","batches":batches,"events":events,"count_samples":counts,"final_dwell_states":engine.state_size,"config_sha256":config_hash,"config_fingerprint":engine.config_fingerprint,"config_snapshot_sha256":sha256(output/'config.snapshot.json'),"observations_sha256":digest.hexdigest(),"trace_enabled":trace,"transition_records":transitions,"diagnostic_records":diagnostics,"output":str(output.resolve())}
    write_json(output/"summary.json",summary)
    return summary


def demo(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={"schema_version":"0.1","zones":{"waiting_area":[[.1,.1],[.9,.1],[.9,.95],[.1,.95]]},"rules":[{"id":"people_in_zone","kind":"count","zone":"waiting_area"},{"id":"person_dwell","kind":"dwell","zone":"waiting_area","seconds":3,"max_gap_seconds":.5}]}
    write_json(output/"config.json",config)
    with (output/"observations.jsonl").open("w",encoding="utf-8") as f:
        for i in range(41):
            detections=[]
            if i<26:detections.append({"track_id":"person-1","label":"person","confidence":.95,"box":[.3,.2,.5,.8]})
            if 10<=i<21:detections.append({"track_id":"person-2","label":"person","confidence":.92,"box":[.6,.3,.7,.7]})
            f.write(json.dumps({"timestamp":i/5,"source":"demo","epoch":"0","covered":True,"detections":detections})+"\n")
    result=replay(output/"observations.jsonl",output/"config.json",output)
    result["note"]="Synthetic observations; person-1 dwells for 3 s, person-2 leaves before threshold. No ML inference."
    return result


def run_video(args):
    import av
    from .tracking import IoUTracker
    from .vision import OnnxDetector
    number(args.sample_fps,'sample-fps',0,120)
    if args.sample_fps==0:raise ValueError('sample-fps must be positive')
    if type(args.max_frames) is not int or args.max_frames<0:raise ValueError('max-frames must be a nonnegative integer')
    text(args.source,'source')
    run_id=uuid.uuid4().hex
    epoch=text(getattr(args,'epoch',None) or run_id,'epoch')
    trace=getattr(args,'trace',False)
    if type(trace) is not bool:raise ValueError('trace must be a boolean')
    video=Path(args.video)
    if not video.is_file():
        raise ValueError("--video must be an existing local file; RTSP/remote sources are not supported in this prototype")
    out=Path(args.output)
    guard_outputs(out,[video,args.config],OUTPUT_NAMES+('observations.jsonl','stage_timings.jsonl'))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',{"schema_version":"0.1","status":"running","kind":"video"})
    start=time.perf_counter();sampled=0;next_due=None;origin=None;previous=None
    totals=dict.fromkeys(RESULT_FILES,0);digest=hashlib.sha256()
    try:
        config,config_hash,snapshot=load_config_snapshot(args.config)
        if any(r.kind=='dwell' and r.max_gap_seconds+1e-9<1/args.sample_fps for r in config.rules):raise ValueError('dwell max_gap_seconds is shorter than the requested sample period')
        write_json(out/'config.snapshot.json',snapshot)
        from .advanced import make_engine
        engine=make_engine(config,trace=trace)
        video_stat=video.stat();video_hash=sha256(video)
        detector=OnnxDetector(args.model,args.models_dir,args.provider,labels={r.label for r in config.rules},confidence=min(r.confidence for r in config.rules),threads=getattr(args,'threads',0))
        tracker=IoUTracker(max_tracks=config.max_tracks)
        with ExitStack() as stack:
            container=stack.enter_context(av.open(str(video)))
            f=stack.enter_context((out/'observations.jsonl').open('wb'))
            timing_file=stack.enter_context((out/'stage_timings.jsonl').open('w',encoding='utf-8',newline='\n'))
            streams=open_results(stack,out);decode_begin=time.perf_counter()
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:raise ValueError("Video frame lacks PTS/time_base; provide timestamped observation replay instead")
                timestamp=float(frame.pts*frame.time_base)
                if timestamp<0:continue
                if not math.isfinite(timestamp):raise ValueError("video timestamp is not finite")
                if previous is not None and timestamp<=previous:raise ValueError("Video timestamps are not strictly increasing; normalize the source before running")
                previous=timestamp
                if next_due is not None and timestamp+1e-9<next_due:continue
                if origin is None:origin=timestamp
                next_due=next_sample_time(timestamp,origin,args.sample_fps)
                decoded=time.perf_counter();pixels=frame.to_ndarray(format='rgb24');converted=time.perf_counter()
                detections,timing=detector.predict(pixels)
                timing.update(decode_sampling_ms=(decoded-decode_begin)*1000,rgb_conversion_ms=(converted-decoded)*1000)
                begin=time.perf_counter();tracked=tracker.update(detections,timestamp);timing["tracking_ms"]=(time.perf_counter()-begin)*1000
                observation=Observation(timestamp,tracked,args.source,epoch)
                begin=time.perf_counter();evaluation=engine.process(observation);timing['rules_ms']=(time.perf_counter()-begin)*1000
                if tracker.last_diagnostics:evaluation=replace(evaluation,diagnostics=evaluation.diagnostics+tracker.last_diagnostics)
                begin=time.perf_counter();raw=(json.dumps(asdict(observation),allow_nan=False)+'\n').encode('utf-8')
                f.write(raw);digest.update(raw)
                records=write_evaluation(evaluation,observation,streams)
                for name,value in records.items():totals[name]+=value
                timing['serialization_ms']=(time.perf_counter()-begin)*1000
                timing['sample_pipeline_ms']=(time.perf_counter()-decode_begin)*1000
                timing_file.write(json.dumps({"timestamp":timestamp,**timing},allow_nan=False)+"\n")
                sampled+=1
                if args.max_frames and sampled>=args.max_frames:break
                decode_begin=time.perf_counter()
        if not sampled:raise ValueError("video produced no usable timestamped frames")
        final_stat=video.stat()
        if (final_stat.st_size,final_stat.st_mtime_ns)!=(video_stat.st_size,video_stat.st_mtime_ns):raise ValueError('video file changed during processing')
        result={'schema_version':'0.1','behaviorkit':__version__,'status':'ok','kind':'video','run_id':run_id,'source_epoch':epoch,'video_sha256':video_hash,'batches':sampled,'sampled_frames':sampled,'events':totals['events'],'count_samples':totals['counts'],'transition_records':totals['transitions'],'diagnostic_records':totals['diagnostics'],'trace_enabled':trace,'final_dwell_states':engine.state_size,'config_sha256':config_hash,'config_snapshot_sha256':sha256(out/'config.snapshot.json'),'config_fingerprint':engine.config_fingerprint,'observations_sha256':digest.hexdigest(),'output':str(out.resolve())}
        result.update(wall_seconds=time.perf_counter()-start,model=args.model,provider_requested=args.provider,provider=detector.selected_provider,provider_note=detector.selection_note,model_manifest=detector.manifest,sample_fps=args.sample_fps,tracker='max_cardinality_iou_min_hits_2',tracker_capacity_policy=tracker.capacity_policy,decode='PyAV CPU decode',timing_note='sample_pipeline_ms includes decode/sampling between selected frames, RGB conversion, detection, tracking, rules and result serialization; excludes initialization and timing-record writing. Offline processing time is not camera evidence age.',limits='No re-identification, camera-motion compensation or spatial hysteresis. Incremental evaluation of local files; not an RTSP service.')
        write_json(out/"summary.json",result)
        return result
    except Exception as error:
        write_json(out/'summary.json',{"schema_version":"0.1","status":"error","kind":"video","sampled_frames":sampled,"error":f"{type(error).__name__}: {error}","partial_outputs":True})
        raise


def next_sample_time(timestamp,origin,fps):
    """Advance a source-time grid in constant work, even after a long source gap."""
    return origin+(math.floor((timestamp-origin)*fps+1e-7)+1)/fps


def parser():
    root=argparse.ArgumentParser(prog="behaviorkit",description="Local counting, dwell and portable ONNX feasibility benchmarks. No LLM calls.")
    root.add_argument("--version",action="version",version=__version__)
    sub=root.add_subparsers(dest="command",required=True)
    d=sub.add_parser("demo",help="Run synthetic observations without ML dependencies");d.add_argument("--output",default="runs/demo")
    v=sub.add_parser("validate",help="Validate a counting/dwell scene config");v.add_argument("--config",required=True)
    r=sub.add_parser("replay",help="Evaluate timestamped observation JSONL");r.add_argument("--observations",required=True);r.add_argument("--config",required=True);r.add_argument("--output",default="runs/replay");r.add_argument('--trace',action='store_true',help='Record dwell starts, confirmations and reset reasons')
    run=sub.add_parser("run",help="Process a local video through ONNX, tracking and rules")
    run.add_argument("--video",required=True);run.add_argument("--config",required=True);run.add_argument("--model",choices=["rtmdet-tiny","d-fine-n","rf-detr-nano"],required=True);run.add_argument("--models-dir",default="models");run.add_argument("--provider",choices=CHOICES,default="auto");run.add_argument("--threads",type=int,default=0);run.add_argument("--sample-fps",type=float,default=5);run.add_argument("--source",default="camera");run.add_argument("--max-frames",type=int,default=0);run.add_argument("--output",default="runs/video")
    run.add_argument('--epoch',help='Explicit source epoch for reproducible repeated ingestion; defaults to a unique run ID')
    run.add_argument('--trace',action='store_true',help='Record dwell starts, confirmations and reset reasons')
    doc=sub.add_parser("doctor",help="Inspect CPU architecture and available execution providers");doc.add_argument("--provider",choices=CHOICES,default="auto")
    models=sub.add_parser("models",help="List model artifact contracts")
    bench=sub.add_parser("benchmark",help="Measure engine or actual ONNX models")
    bs=bench.add_subparsers(dest="benchmark_kind",required=True)
    en=bs.add_parser("engine");en.add_argument("--batches",type=int,default=5000);en.add_argument("--tracks",type=int,default=100);en.add_argument("--output",default="runs/engine-benchmark.json")
    mo=bs.add_parser("models");mo.add_argument("--models",nargs="+",default=["rtmdet-tiny","d-fine-n","rf-detr-nano"]);mo.add_argument("--models-dir",default="models");mo.add_argument("--provider",choices=CHOICES,default="auto");mo.add_argument("--threads",type=int,default=0);mo.add_argument("--iterations",type=int,default=50);mo.add_argument("--warmup",type=int,default=10);mo.add_argument("--output",default="runs/model-benchmark.json")
    from .commands import register
    register(sub)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        exit_code=0
        from .commands import COMMANDS, dispatch
        if args.command=="demo":result=demo(args.output)
        elif args.command=="validate":
            config=load_config(args.config);result={"valid":True,"rules":len(config.rules),"zones":list(config.zones)}
        elif args.command=="replay":result=replay(args.observations,args.config,args.output,trace=args.trace)
        elif args.command in COMMANDS:result=dispatch(args)
        elif args.command=="models":
            from .vision import PRESETS
            result=PRESETS
        elif args.command=="doctor":
            from .benchmark import environment
            result=environment(include_nvidia=args.provider in {'auto','cuda','tensorrt'})
            from .vision import runtime
            ort=runtime();selection=select_provider(args.provider,ort.get_available_providers())
            result.update(onnxruntime=ort.__version__,available_providers=ort.get_available_providers(),provider_requested=args.provider,provider_selected=selection.selected,selection_note=selection.reason,note="Availability only; benchmark or runtime validation must verify actual execution. CoreML events establish delegation, not which Apple compute unit was used.")
        elif args.command=="run":
            if not 0<args.sample_fps<=120 or args.max_frames<0:raise ValueError("sample-fps must be in (0,120]; max-frames must be nonnegative")
            result=run_video(args)
        elif args.benchmark_kind=="engine":
            from .benchmark import engine_benchmark
            result=engine_benchmark(args.batches,args.tracks);write_json(args.output,result)
        else:
            from .benchmark import model_benchmark
            result=model_benchmark(args.models,args.models_dir,args.provider,args.iterations,args.warmup,threads=args.threads,profile_dir=Path(args.output).parent/'profiles');write_json(args.output,result)
            if any(m["status"]!="ok" for m in result["models"]):exit_code=1
        if isinstance(result,dict) and result.get('run_status',result.get('status')) in {'failed','partial_failure','cancelled'}:exit_code=1
        print(json.dumps(result,indent=2,allow_nan=False));return exit_code
    except (ValueError,KeyError,TypeError,RuntimeError,OSError,ImportError) as e:
        print(f"behaviorkit: {e}",file=sys.stderr);return 2


if __name__=="__main__":
    raise SystemExit(main())
