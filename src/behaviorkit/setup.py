"""Explicit source/geometry binding and bounded offline inspection."""
from dataclasses import dataclass, replace
from collections import Counter
from types import MappingProxyType

from .advanced import AdvancedScene, make_engine, parse_scene
from .contracts import EvidenceFrame, fingerprint
from .core import Observation, inside, keys, number, text


@dataclass(frozen=True)
class Geometry:
    revision: str
    width: int
    height: int
    rotation: int = 0
    crop: tuple = (0, 0, 1, 1)

    def __post_init__(self):
        text(self.revision, 'geometry revision')
        for x in (self.width, self.height):
            if type(x) is not int or not 1 <= x <= 32768:
                raise ValueError('reference dimensions must be in 1..32768')
        if type(self.rotation) is not int or self.rotation not in {0,90,180,270}:
            raise ValueError('rotation must be 0,90,180,270')
        crop = tuple(self.crop)
        if len(crop) != 4:
            raise ValueError('crop must be normalized xyxy')
        for v in crop: number(v, 'crop', 0, 1)
        if crop[0] >= crop[2] or crop[1] >= crop[3]:
            raise ValueError('crop requires positive dimensions')
        object.__setattr__(self, 'crop', crop)

    @property
    def fingerprint(self):
        # Dimensions describe the image AFTER the declared transform. No metres.
        return fingerprint({'revision':self.revision,'aspect':self.width/self.height,
                            'rotation':self.rotation,'crop':self.crop,'units':'normalized'})

    def validate_frame(self, frame):
        if frame.geometry_revision != self.revision:
            raise ValueError('geometry revision differs; revalidate and bind a new scene')
        if abs(frame.width/frame.height-self.width/self.height) > .002:
            raise ValueError('aspect ratio changed without a valid coordinate transform')


class SceneRouter:
    """Source-specific engines. Cutover is explicit and always resets state."""
    def __init__(self, bindings, *, trace=False, max_sources=8):
        if type(max_sources) is not int or not 1 <= max_sources <= 100:
            raise ValueError('max_sources must be in 1..100')
        if not isinstance(bindings, dict) or not 1 <= len(bindings) <= max_sources:
            raise ValueError('bindings exceed source capacity')
        self._bindings, self._engines, self._generations, self.cutovers = {}, {}, {}, []
        self.trace, self.max_sources = trace, max_sources
        for source, (scene, geometry) in bindings.items():
            self._bind(source, scene, geometry)

    def _bind(self, source, scene, geometry):
        text(source, 'source')
        if not isinstance(geometry, Geometry): raise ValueError('Geometry required')
        scene = parse_scene(scene)
        if isinstance(scene,AdvancedScene) and scene.geometry_revision!=geometry.revision:
            raise ValueError('scene and binding geometry revisions differ')
        self._bindings[source] = (scene, geometry)
        self._engines[source] = make_engine(scene, trace=self.trace)
        self._generations.setdefault(source, '0')

    @property
    def bindings(self): return MappingProxyType(self._bindings)

    def cutover(self, source, scene, geometry, *, timestamp, reason):
        if source not in self._bindings: raise ValueError('source has no scene binding')
        number(timestamp, 'cutover timestamp'); text(reason, 'cutover reason')
        prior = self._bindings[source]
        parsed = parse_scene(scene)
        if not isinstance(geometry, Geometry): raise ValueError('Geometry required')
        record = {'source':source,'timestamp':timestamp,'reason':reason,'before_config':prior[0].fingerprint,
                  'after_config':parsed.fingerprint,'before_geometry':prior[1].fingerprint,
                  'after_geometry':geometry.fingerprint,'state_policy':'reset'}
        self._bind(source, parsed, geometry)
        self._generations[source] = fingerprint([self._generations[source], record])
        # Bounded audit tail; applications can durably persist each return value.
        self.cutovers = (self.cutovers+[record])[-100:]
        return record

    def prepare_frame(self, frame):
        if not isinstance(frame, EvidenceFrame): raise ValueError('EvidenceFrame required')
        o = frame.confirmed_observation()
        if o.source not in self._bindings: raise ValueError(f'unknown source binding: {o.source}')
        scene, geometry = self._bindings[o.source]
        geometry.validate_frame(frame)
        # Full provenance is separate from rule semantics. Explicit cutover adds
        # a generation; unrelated edits during offline comparison do not.
        epoch = fingerprint([o.epoch, geometry.fingerprint, self._generations[o.source]])
        return replace(frame, observation=replace(frame.observation, epoch=epoch))

    def process(self, frame):
        prepared = self.prepare_frame(frame)
        return self._engines[prepared.observation.source].process(prepared.confirmed_observation())

    def close_source(self, source):
        if source not in self._engines: raise ValueError('unknown source binding')
        return self._engines[source].close_source(source)


class Inspector:
    """Counters + capped details. Cannot explain objects absent from saved boxes."""
    def __init__(self, scene, *, source=None, rule=None, track=None, start=None, end=None, detail_limit=100, candidate_stage=False):
        self.scene = parse_scene(scene)
        if type(detail_limit) is not int or not 0 <= detail_limit <= 10000:
            raise ValueError('detail_limit must be in 0..10000')
        if start is not None: number(start,'start')
        if end is not None: number(end,'end')
        if start is not None and end is not None and end < start: raise ValueError('invalid inspection range')
        if rule is not None and rule not in {r.id for r in self.scene.rules}: raise ValueError('unknown inspection rule')
        self.source,self.rule,self.track,self.start,self.end,self.limit = source,rule,track,start,end,detail_limit
        self.candidate_stage = candidate_stage
        self.counters, self.details, self.frames, self.truncated = Counter(), [], 0, 0

    def _reason(self, o, r, tid, reason):
        self.counters[(r.id, reason)] += 1
        if len(self.details) < self.limit:
            self.details.append({'source':o.source,'epoch':o.epoch,'timestamp':o.timestamp,'rule':r.id,'track_id':tid,'reason':reason})
        else: self.truncated += 1

    def observe(self, frame, result=None, states=()):
        o = frame.observation
        if self.source is not None and o.source != self.source or self.start is not None and o.timestamp < self.start or self.end is not None and o.timestamp > self.end:
            return
        self.frames += 1
        statuses = dict(frame.statuses)
        state_map = {(s['rule'],s['track_id']):s for s in states if s['source'] == o.source}
        emitted = {(e['rule'],e['track_id']) for e in result.events} if result else set()
        for r in self.scene.rules:
            if self.rule is not None and r.id != self.rule: continue
            if not o.covered:
                self._reason(o,r,self.track,'missing_coverage'); continue
            if not o.detections:
                self._reason(o,r,self.track,'no_recorded_candidate'); continue
            for d in o.detections:
                if self.track is not None and d.track_id != self.track: continue
                state = state_map.get((r.id,d.track_id))
                reason = ('class_not_selected' if d.label != r.label else
                          'below_threshold' if d.confidence < r.confidence else
                          'tentative_track' if statuses.get(d.track_id) == 'tentative' else
                          'predicted_position' if statuses.get(d.track_id) == 'predicted' else
                          'outside_zone' if not inside(d.anchor,self.scene.zones[r.zone]) else
                          'confirmed_now' if (r.id,d.track_id) in emitted else
                          'already_emitted_episode' if state and state.get('emitted') else
                          'untracked_candidate' if self.candidate_stage else
                          'insufficient_supported_duration' if r.kind == 'dwell' else 'eligible_observation')
                self._reason(o,r,d.track_id,reason)
        if result:
            for t in result.transitions:
                rid = t.get('rule')
                if self.rule is not None and rid != self.rule or self.track is not None and t.get('track_id') != self.track: continue
                self.counters[(rid, 'transition:'+t.get('reason','unknown'))] += 1

    def report(self):
        return {'schema_version':'inspection/1','config_fingerprint':self.scene.fingerprint,'frames':self.frames,
                'counters':[{'rule':r,'reason':why,'count':n} for (r,why),n in sorted(self.counters.items())],
                'details':list(self.details),'omitted_details':self.truncated,
                'limit':'Only retained evidence is explained. An absent candidate does not prove physical absence or a detector miss.'}
