"""Streaming, content-verified candidate/track/rule stages.

Stages are immutable directories. A complete manifest and hashes, rather than
directory existence, establish cache validity. No detector is imported by replay.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import json
import hashlib
import time

from .contracts import Capabilities, EvidenceFrame, fingerprint, json_line, read_json, read_jsonl
from .core import Observation
from .storage import sha256, write_json
from .tracking import IoUTracker

IMPLEMENTATION = 'behaviorkit-stages/1'


def implementation_hash(kind):
    """Invalidate caches when actual shipped stage/rule/adapter code changes."""
    shared=('contracts.py','core.py','recording.py')
    dependent={'candidates':('pipeline.py','vision.py','adapters.py','providers.py'),
               'tracks':('tracking.py','motion.py','setup.py'), 'rules':('advanced.py','setup.py')}
    return fingerprint({name:hashlib.sha256((Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                        for name in shared+dependent.get(kind,())})


def stage_key(kind, inputs, parameters):
    return fingerprint({'implementation': IMPLEMENTATION, 'code_sha256':implementation_hash(kind),
                        'kind': kind, 'inputs': inputs, 'parameters': parameters})


@contextmanager
def new_stage(output, kind, inputs, parameters, capabilities=None):
    # Freeze owned JSON snapshots before a generator or trusted adapter executes.
    inputs=json.loads(json.dumps(inputs,allow_nan=False))
    parameters=json.loads(json.dumps(parameters,allow_nan=False))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 'recording/1', 'status': 'running', 'kind': kind,
                'implementation': IMPLEMENTATION, 'inputs': inputs, 'parameters': parameters,
                'implementation_sha256': implementation_hash(kind),
                'cache_key': stage_key(kind, inputs, parameters), 'artifacts': {},
                'created_unix': time.time()}
    if capabilities is not None:
        manifest['capabilities'] = capabilities.to_dict()
    write_json(output/'manifest.json', manifest)
    try:
        yield output, manifest
        artifacts = {}
        for path in sorted(output.iterdir()):
            if path.name == 'manifest.json':
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('stage artifacts must be regular owned files')
            artifacts[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        manifest.update(status='complete', artifacts=artifacts)
        write_json(output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='error', error=f'{type(error).__name__}: {error}')
        write_json(output/'manifest.json', manifest)
        raise


def verified_manifest(directory, *, kind=None):
    directory = Path(directory)
    if directory.is_symlink() or (directory/'manifest.json').is_symlink():
        raise ValueError('symlink recording manifests are unsupported')
    manifest = read_json(directory/'manifest.json')
    if manifest.get('schema_version') != 'recording/1' or manifest.get('status') != 'complete':
        raise ValueError('recording is incomplete or unsupported')
    if kind is not None and manifest.get('kind') != kind:
        raise ValueError(f'expected {kind} recording')
    expected_key=fingerprint({'implementation':manifest['implementation'],'code_sha256':manifest.get('implementation_sha256'),
                              'kind':manifest['kind'],'inputs':manifest['inputs'],'parameters':manifest['parameters']})
    if manifest.get('cache_key') != expected_key:
        raise ValueError('recording cache key mismatch')
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('recording has no artifacts')
    for name, item in artifacts.items():
        if Path(name).name != name or name in {'', '.', '..'}:
            raise ValueError('invalid artifact path')
        path = directory/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'recording artifact integrity failure: {name}')
    return manifest


def cache_hit(directory, kind, inputs, parameters):
    try:
        return verified_manifest(directory, kind=kind)['cache_key'] == stage_key(kind, inputs, parameters)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _frames(directory, manifest):
    if 'frames.jsonl' not in manifest['artifacts']:
        raise ValueError('recording lacks frame evidence')
    for row in read_jsonl(Path(directory)/'frames.jsonl'):
        yield EvidenceFrame.from_dict(row)


def record_candidates(frames, output, capabilities, provenance, *, parameters=None):
    if capabilities.stage != 'postprocessed_candidates' or capabilities.pixels or capabilities.embeddings:
        raise ValueError('this writer retains postprocessed boxes only, without pixels or embeddings')
    parameters = parameters or {}
    with new_stage(output, 'candidates', provenance, parameters, capabilities) as (out, manifest):
        total, clocks = 0, {}
        with (out/'frames.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for frame in frames:
                if not isinstance(frame, EvidenceFrame):
                    raise ValueError('candidate writer requires EvidenceFrame objects')
                o = frame.observation
                key = (o.source, o.epoch)
                if len(clocks) >= 100 and key not in clocks:
                    raise ValueError('candidate writer source/epoch limit exceeded; split the recording')
                if key in clocks and o.timestamp <= clocks[key]:
                    raise ValueError('recorded timestamps must strictly increase within source epoch')
                clocks[key] = o.timestamp
                if len(o.detections) > capabilities.max_detections:
                    raise ValueError('candidate count exceeds declared capture limit')
                if any(d.label not in capabilities.classes or d.confidence < capabilities.confidence_floor-1e-9 for d in o.detections):
                    raise ValueError('candidate violates declared classes/capture floor')
                json_line(stream, frame.to_dict()); total += 1
        if not total:
            raise ValueError('cannot complete an empty candidate recording')
        manifest['frames'] = total
    return manifest


def retrack(recording, output, *, tracker_factory=None, tracker_parameters=None, max_sources=8):
    source = verified_manifest(recording, kind='candidates')
    capabilities = Capabilities(**source['capabilities'])
    capabilities.require(classes=capabilities.classes, confidence=capabilities.confidence_floor, tracker=True)
    parameters = tracker_parameters or {'adapter': 'iou', 'min_hits': 2}
    parameters = dict(parameters)
    floor = parameters.setdefault('confidence_floor',max(.3,capabilities.confidence_floor) if tracker_factory is None else capabilities.confidence_floor)
    capabilities.require(classes=capabilities.classes, confidence=floor)
    if tracker_factory is not None and 'adapter' not in parameters:
        raise ValueError('custom tracker must declare a versioned adapter identity in parameters')
    if type(max_sources) is not int or not 1 <= max_sources <= 100:
        raise ValueError('max_sources must be in 1..100')
    parameters = {**parameters, 'max_sources': max_sources}
    if tracker_factory is None:
        if parameters['adapter'] != 'iou':
            raise ValueError('custom tracker requires explicit factory')
        settings = {k:v for k,v in parameters.items() if k not in {'adapter', 'max_sources', 'confidence_floor'}}
        tracker_factory = lambda: IoUTracker(**settings)
    inputs = {'recording_manifest_sha256': sha256(Path(recording)/'manifest.json')}
    tracked_capabilities = replace(capabilities, stage='tracked_observations',confidence_floor=floor)
    trackers, identities, segments = {}, {}, {}
    pipeline_identity=fingerprint([implementation_hash('tracks'),source['cache_key'],source['artifacts']['frames.jsonl']['sha256']])
    with new_stage(output, 'tracks', inputs, parameters, tracked_capabilities) as (out, manifest):
        with (out/'frames.jsonl').open('w', encoding='utf-8', newline='\n') as stream, (out/'tracking.jsonl').open('w',encoding='utf-8',newline='\n') as diagnostics:
            for frame in _frames(recording, source):
                o = frame.observation
                binding = (o.epoch, frame.geometry_revision, frame.width/frame.height)
                if o.source not in trackers and len(trackers) >= max_sources:
                    raise ValueError('tracker source capacity exceeded')
                if identities.get(o.source) != binding:
                    trackers[o.source] = tracker_factory(); identities[o.source] = binding
                    segments[o.source] = (float(o.timestamp), 0)
                tracker = trackers[o.source]
                # Uncovered observations reset state instead of bridging unknown time.
                if not o.covered:
                    trackers[o.source] = tracker_factory()
                    segments[o.source] = (float(o.timestamp), segments[o.source][1]+1)
                    tracked = ()
                else:
                    tracked = tracker.update(tuple(d for d in o.detections if d.confidence>=floor), o.timestamp)
                json_line(diagnostics,{'source':o.source,'epoch':o.epoch,'timestamp':o.timestamp,
                                       'messages':tracker.last_diagnostics,'emitted':'current observed confirmed tracks only'})
                json_line(stream, replace(frame, observation=Observation(o.timestamp, tracked, o.source,
                          fingerprint([binding, parameters, segments[o.source], pipeline_identity]), o.covered), statuses=()).to_dict())
        verified_manifest(recording, kind='candidates')
        if inputs['recording_manifest_sha256'] != sha256(Path(recording)/'manifest.json'):
            raise ValueError('input recording changed during tracker replay')
        manifest['frames'] = source.get('frames')
    return manifest


def apply_rules(recording, config, output, *, trace=False):
    from .advanced import AdvancedScene, make_engine, parse_scene
    source = verified_manifest(recording, kind='tracks')
    routed = config is None
    bindings = {}
    if routed:
        rows=source['parameters'].get('sources',[])
        if not 1<=len(rows)<=100: raise ValueError('recording has no saved source bindings')
        for row in rows:
            if row['id'] in bindings: raise ValueError('duplicate saved source binding')
            bindings[row['id']]=parse_scene(row['scene'])
        scenes=tuple(bindings.values())
    else:
        scene = parse_scene(read_json(config) if isinstance(config, (str, Path)) else config)
        scenes=(scene,)
    capabilities = Capabilities(**source['capabilities'])
    for scene in scenes:
        capabilities.require(classes={r.label for r in scene.rules}, confidence=min(r.confidence for r in scene.rules))
        rates=[capabilities.sample_fps] if capabilities.sample_fps else [s['sample_fps'] for s in source['parameters'].get('sources',[]) if not routed or bindings[s['id']] is scene]
        if rates and any(r.kind != 'count' and r.max_gap_seconds + 1e-9 < max(1/fps for fps in rates) for r in scene.rules):
            raise ValueError('rule gap is shorter than recording sample period')
    inputs = {'recording_manifest_sha256': sha256(Path(recording)/'manifest.json')}
    parameters = {'bindings':{k:v.to_dict() for k,v in bindings.items()},'trace':trace} if routed else {'config': scene.to_dict(), 'trace': trace}
    engines={k:make_engine(v,trace=trace) for k,v in bindings.items()} if routed else {}
    engine, totals = (None if routed else make_engine(scene, trace=trace)), dict(events=0, counts=0, transitions=0)
    with new_stage(output, 'rules', inputs, parameters) as (out, manifest):
        from contextlib import ExitStack
        with ExitStack() as stack:
            streams = {name:stack.enter_context((out/(name+'.jsonl')).open('w', encoding='utf-8', newline='\n')) for name in totals}
            for frame in _frames(recording, source):
                if routed:
                    if frame.observation.source not in bindings: raise ValueError('unknown saved source binding')
                    scene=bindings[frame.observation.source]; engine=engines[frame.observation.source]
                if isinstance(scene,AdvancedScene) and frame.geometry_revision!=scene.geometry_revision:
                    raise ValueError('scene/recording geometry revision differs; revalidation required')
                result = engine.process(frame.confirmed_observation())
                for name in totals:
                    for row in getattr(result, name):
                        json_line(streams[name], row); totals[name] += 1
        verified_manifest(recording, kind='tracks')
        if inputs['recording_manifest_sha256'] != sha256(Path(recording)/'manifest.json'):
            raise ValueError('input recording changed during rule replay')
        manifest.update(totals=totals, config_fingerprint=fingerprint(parameters['bindings']) if routed else scene.fingerprint)
    return manifest
