"""Opt-in scene/0.2 operators and semantic event identity v3.

Unknown coverage, missing identity, low confidence and excessive gaps reset state.
Debouncing only bridges an observed, eligible identity outside its polygon. Its
unsupported time never accrues toward dwell. No predicted box is visual evidence.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from types import MappingProxyType
from collections.abc import Mapping

from .contracts import fingerprint
from .core import (BehaviorEngine, Evaluation, Observation, SceneConfig, inside,
                   keys, number, polygon, sequence, text)


@dataclass(frozen=True)
class AdvancedRule:
    id: str
    kind: str
    zone: str
    label: str = 'person'
    confidence: float = .3
    seconds: float = 1.0
    max_gap_seconds: float = .5
    membership: str = 'strict'
    entry_seconds: float = 0.0
    exit_seconds: float = 0.0
    threshold: int = 1
    line: tuple = ()
    direction: str = 'both'
    boundary_tolerance: float = .002
    sequence_zones: tuple = ()
    sequence_timeout: float = 60.0

    def __post_init__(self):
        text(self.id, 'rule id'); text(self.zone, 'zone'); text(self.label, 'label')
        if self.kind not in {'count', 'dwell', 'enter', 'exit', 'cross', 'occupancy', 'sequence'}:
            raise ValueError('unsupported behavior operator')
        number(self.confidence, 'confidence', 0, 1)
        number(self.seconds, 'seconds', .001, 86400)
        number(self.max_gap_seconds, 'max_gap_seconds', .001, 60)
        for name in ('entry_seconds', 'exit_seconds'):
            number(getattr(self, name), name, 0, 60)
        number(self.boundary_tolerance, 'boundary_tolerance', 0, .1)
        number(self.sequence_timeout, 'sequence_timeout', .001, 86400)
        if self.membership not in {'strict', 'debounced'}:
            raise ValueError('membership must be strict or debounced')
        if self.membership == 'strict' and (self.entry_seconds or self.exit_seconds):
            raise ValueError('entry/exit durations require debounced membership')
        if type(self.threshold) is not int or not 1 <= self.threshold <= 10000:
            raise ValueError('occupancy threshold must be in 1..10000')
        if self.direction not in {'both', 'positive', 'negative'}:
            raise ValueError('direction must be both, positive or negative')
        object.__setattr__(self, 'sequence_zones', tuple(sequence(self.sequence_zones, 'sequence_zones')))
        if self.kind == 'sequence' and (not 2 <= len(self.sequence_zones) <= 20 or any(not isinstance(z, str) for z in self.sequence_zones)):
            raise ValueError('sequence requires 2..20 named zones')
        line = tuple(tuple(p) for p in sequence(self.line, 'line'))
        if self.kind == 'cross':
            if len(line) != 2 or any(len(p) != 2 for p in line):
                raise ValueError('cross requires a two-point normalized line')
            for p in line:
                for v in p: number(v, 'line coordinate', 0, 1)
            if line[0] == line[1]:
                raise ValueError('crossing line must have nonzero length')
        object.__setattr__(self, 'line', line)

    def to_dict(self):
        value = asdict(self)
        for k in ('confidence', 'seconds', 'max_gap_seconds', 'entry_seconds', 'exit_seconds', 'boundary_tolerance', 'sequence_timeout'):
            value[k] = float(value[k])
        value['line'] = [[float(v) for v in p] for p in self.line]
        value['sequence_zones'] = list(self.sequence_zones)
        return value


@dataclass(frozen=True)
class AdvancedScene:
    zones: Mapping
    rules: tuple[AdvancedRule, ...]
    max_tracks: int = 200
    max_sources: int = 8
    geometry_revision: str = '1'

    def __post_init__(self):
        if not isinstance(self.zones, Mapping) or not 1 <= len(self.zones) <= 100:
            raise ValueError('zones requires 1..100 polygons')
        zones = {text(k, 'zone name'):polygon(v) for k,v in self.zones.items()}
        rules = tuple(sequence(self.rules, 'rules'))
        if not 1 <= len(rules) <= 100 or not all(isinstance(r, AdvancedRule) for r in rules) or len({r.id for r in rules}) != len(rules):
            raise ValueError('rules requires 1..100 uniquely named AdvancedRule objects')
        for r in rules:
            if r.zone not in zones or any(z not in zones for z in r.sequence_zones):
                raise ValueError('rule references an unknown zone')
        for k, maximum in [('max_tracks', 10000), ('max_sources', 100)]:
            if type(getattr(self, k)) is not int or not 1 <= getattr(self, k) <= maximum:
                raise ValueError(f'{k} is out of bounds')
        text(self.geometry_revision, 'geometry_revision')
        object.__setattr__(self, 'zones', MappingProxyType(zones))
        object.__setattr__(self, 'rules', rules)

    def to_dict(self):
        return {'schema_version': '0.2', 'zones': {k:[list(p) for p in v] for k,v in self.zones.items()},
                'rules': [r.to_dict() for r in self.rules], 'max_tracks': self.max_tracks,
                'max_sources': self.max_sources, 'geometry_revision': self.geometry_revision}

    @property
    def fingerprint(self):
        return fingerprint(self.to_dict())

    def semantic_fingerprint(self, rule):
        names = {rule.zone, *rule.sequence_zones}
        return fingerprint({'identity_version': 3, 'rule': rule.to_dict(),
                            'geometry': {z:self.zones[z] for z in sorted(names)},
                            'geometry_revision': self.geometry_revision,
                            'anchor': 'bottom_center', 'coordinates': 'normalized_xyxy',
                            'unknown_policy': 'reset', 'implementation': 'operators/1'})


def parse_scene(data):
    if isinstance(data, (SceneConfig, AdvancedScene)):
        return data
    if data.get('schema_version') == '0.1':
        return SceneConfig.from_dict(data)
    keys(data, {'schema_version', 'zones', 'rules', 'max_tracks', 'max_sources', 'geometry_revision'}, 'scene', {'schema_version', 'zones', 'rules'})
    if data['schema_version'] != '0.2':
        raise ValueError('scene schema must be 0.1 or 0.2')
    rules = []
    fields = set(AdvancedRule.__dataclass_fields__)
    for row in sequence(data['rules'], 'rules'):
        keys(row, fields, 'rule', {'id', 'kind', 'zone'})
        rules.append(AdvancedRule(**row))
    return AdvancedScene(data['zones'], tuple(rules), data.get('max_tracks', 200), data.get('max_sources', 8), data.get('geometry_revision', '1'))


def make_engine(scene, *, trace=False):
    return BehaviorEngine(scene, trace=trace) if isinstance(scene, SceneConfig) else AdvancedEngine(scene, trace=trace)


@dataclass
class _State:
    start: float
    last: float
    member: bool = False
    pending_since: float | None = None
    raw_inside: bool = False
    supported: float = 0.0
    emitted: bool = False
    point: tuple | None = None
    side: int = 0
    step: int = 0
    last_zone: str | None = None


def _cross_side(point, line, tolerance):
    a, b = line
    dx, dy = b[0]-a[0], b[1]-a[1]
    signed = (dx*(point[1]-a[1])-dy*(point[0]-a[0])) / (dx*dx+dy*dy)**.5
    return 1 if signed > tolerance else -1 if signed < -tolerance else 0


def _crosses_segment(p, q, a, b):
    cross = lambda u,v: u[0]*v[1]-u[1]*v[0]
    r, s = (q[0]-p[0], q[1]-p[1]), (b[0]-a[0], b[1]-a[1])
    den = cross(r, s)
    if abs(den) < 1e-12: return False
    ap = (a[0]-p[0], a[1]-p[1])
    t, u = cross(ap,s)/den, cross(ap,r)/den
    return -1e-9 <= t <= 1+1e-9 and -1e-9 <= u <= 1+1e-9


class AdvancedEngine:
    def __init__(self, config, *, trace=False):
        if not isinstance(config, AdvancedScene) or type(trace) is not bool:
            raise ValueError('AdvancedScene and boolean trace required')
        self.config, self.trace = config, trace
        self.config_fingerprint = config.fingerprint
        self._semantic = {r.id:config.semantic_fingerprint(r) for r in config.rules}
        self._sources, self._states = {}, {}

    @property
    def state_size(self): return len(self._states)

    def dwell_states(self, source=None):
        return tuple({'source':k[0], 'rule':k[1], 'track_id':k[2], **asdict(s)} for k,s in sorted(self._states.items()) if source is None or k[0] == source)

    def close_source(self, source):
        self._sources.pop(source, None)
        self._states = {k:v for k,v in self._states.items() if k[0] != source}
        return Evaluation((), (), ('source_closed; state reset',))

    def _event(self, r, o, tid, start, evidence):
        semantic = self._semantic[r.id]
        return {'id': fingerprint(['behaviorkit-event-v3', semantic, r.id, o.source, o.epoch, tid, float(start)]),
                'identity_version': 3, 'semantic_fingerprint': semantic,
                'config_fingerprint': self.config_fingerprint, 'kind': r.kind, 'rule': r.id,
                'source': o.source, 'epoch': o.epoch, 'track_id': tid, 'zone': r.zone,
                'start': start, 'timestamp': o.timestamp, 'duration_seconds': o.timestamp-start,
                'status': 'confirmed_by_rule', 'evidence': {'anchor':'bottom_center', 'unknown_policy':'reset', **evidence}}

    def process(self, o):
        if not isinstance(o, Observation): raise ValueError('Observation required')
        prior = self._sources.get(o.source)
        if prior and prior[0] == o.epoch and o.timestamp <= prior[1]:
            raise ValueError('timestamps must strictly increase within a source epoch')
        if prior is None and len(self._sources) >= self.config.max_sources:
            raise ValueError('source capacity exceeded')
        if len(o.detections) > self.config.max_tracks:
            raise ValueError('track capacity exceeded')
        transitions, diagnostics = [], []
        if prior and prior[0] != o.epoch:
            self.close_source(o.source); diagnostics.append('source_epoch_changed; state reset')
        self._sources[o.source] = (o.epoch, o.timestamp)
        counts, events, active = [], [], set()

        def transition(rule, tid, reason):
            if self.trace:
                transitions.append({'source':o.source, 'epoch':o.epoch, 'timestamp':o.timestamp,
                                    'rule':rule.id, 'track_id':tid, 'reason':reason})

        for r in self.config.rules:
            eligible = [d for d in o.detections if o.covered and d.label == r.label and d.confidence >= r.confidence]
            selected = [d for d in eligible if inside(d.anchor, self.config.zones[r.zone])]
            if r.kind == 'count':
                counts.append({'rule':r.id, 'source':o.source, 'epoch':o.epoch, 'timestamp':o.timestamp,
                               'count':len(selected) if o.covered else None, 'status':'observed' if o.covered else 'unknown',
                               'track_ids':[d.track_id for d in selected]})
                continue
            if r.kind == 'occupancy':
                key = (o.source, r.id, '@occupancy')
                if o.covered and len(selected) >= r.threshold:
                    active.add(key); state = self._states.get(key)
                    if state is None or o.timestamp-state.last > r.max_gap_seconds+1e-9:
                        state = self._states[key] = _State(o.timestamp, o.timestamp)
                        transition(r, '@occupancy', 'started')
                    state.last = o.timestamp
                    if not state.emitted and o.timestamp-state.start+1e-9 >= r.seconds:
                        events.append(self._event(r,o,'@occupancy',state.start,{'count':len(selected),'threshold':r.threshold}))
                        state.emitted = True; transition(r, '@occupancy', 'confirmed')
                continue
            for d in eligible:
                key = (o.source, r.id, d.track_id); active.add(key)
                state = self._states.get(key)
                if state is None or o.timestamp-state.last > r.max_gap_seconds+1e-9:
                    if state is not None: transition(r, d.track_id, 'observation_gap_reset')
                    state = self._states[key] = _State(o.timestamp, o.timestamp)
                dt = o.timestamp-state.last
                is_inside = inside(d.anchor, self.config.zones[r.zone])
                if r.kind == 'cross':
                    if not is_inside:
                        state.side=0; state.point=None; state.last=o.timestamp
                        continue
                    side = _cross_side(d.anchor, r.line, r.boundary_tolerance)
                    if side and state.side and side != state.side and state.point is not None:
                        direction = 'positive' if side > 0 else 'negative'
                        if r.direction in {'both', direction} and _crosses_segment(state.point, d.anchor, *r.line):
                            events.append(self._event(r,o,d.track_id,state.start,{'direction':direction,'from':state.point,'to':d.anchor}))
                            transition(r,d.track_id,'crossed')
                    if side:
                        state.side, state.point, state.start = side, d.anchor, o.timestamp
                elif r.kind == 'sequence':
                    zones = [z for z in dict.fromkeys(r.sequence_zones) if inside(d.anchor,self.config.zones[z])]
                    if len(zones) > 1:
                        state.step = 0; state.last_zone = None
                        transition(r,d.track_id,'ambiguous_zone_reset')
                    elif zones:
                        z = zones[0]
                        if state.step and o.timestamp-state.start > r.sequence_timeout+1e-9:
                            state.step = 0; state.last_zone = None
                            transition(r,d.track_id,'sequence_timeout_reset')
                        if z != state.last_zone:
                            if z == r.sequence_zones[state.step]:
                                if state.step == 0: state.start = o.timestamp
                                state.step += 1
                                if state.step == len(r.sequence_zones):
                                    events.append(self._event(r,o,d.track_id,state.start,{'sequence':list(r.sequence_zones)}))
                                    state.step = 0; transition(r,d.track_id,'sequence_confirmed')
                            else:
                                state.step = 1 if z == r.sequence_zones[0] else 0
                                state.start = o.timestamp
                            state.last_zone = z
                    else:
                        state.last_zone = None
                else:
                    changed = None
                    if is_inside != state.member:
                        if state.pending_since is None: state.pending_since = o.timestamp
                        required = (r.entry_seconds if is_inside else r.exit_seconds) if r.membership == 'debounced' else 0
                        if o.timestamp-state.pending_since+1e-9 >= required:
                            state.member = is_inside; state.pending_since = None
                            changed = 'entered' if is_inside else 'exited'
                            transition(r,d.track_id,changed)
                            if is_inside:
                                state.start = o.timestamp; state.supported = 0; state.emitted = False
                    else:
                        state.pending_since = None
                    if changed == 'entered' and r.kind == 'enter':
                        events.append(self._event(r,o,d.track_id,state.start,{'membership':r.membership}))
                    if changed == 'exited' and r.kind == 'exit':
                        events.append(self._event(r,o,d.track_id,state.start,{'membership':r.membership}))
                    if r.kind == 'dwell':
                        if state.member and is_inside:
                            if state.raw_inside and changed != 'entered': state.supported += dt
                            if not state.emitted and state.supported+1e-9 >= r.seconds:
                                events.append(self._event(r,o,d.track_id,state.start,{'supported_seconds':state.supported,'membership':r.membership,'confidence':d.confidence,'box':d.box}))
                                state.emitted = True; transition(r,d.track_id,'confirmed')
                        elif not state.member:
                            state.supported = 0; state.emitted = False
                    state.raw_inside = is_inside
                state.last = o.timestamp
        for key in list(self._states):
            if key[0] == o.source and key not in active:
                if self.trace:
                    transitions.append({'source':o.source,'epoch':o.epoch,'timestamp':o.timestamp,'rule':key[1],
                                        'track_id':key[2],'reason':'missing_eligible_evidence_reset'})
                del self._states[key]
        if not o.covered: diagnostics.append('source_not_covered; state reset')
        return Evaluation(tuple(counts), tuple(events), tuple(diagnostics), tuple(transitions))


def config_diff(before, after):
    before, after = parse_scene(before), parse_scene(after)
    if type(before) is not type(after):
        return {'identity_migration': True, 'reset_required': True, 'reason':'schema/identity version changed',
                'before':before.fingerprint, 'after':after.fingerprint}
    a, b = {r.id:r for r in before.rules}, {r.id:r for r in after.rules}
    changed = []
    for name in sorted(set(a) & set(b)):
        fa = before.semantic_fingerprint(a[name]) if isinstance(before, AdvancedScene) else before.fingerprint
        fb = after.semantic_fingerprint(b[name]) if isinstance(after, AdvancedScene) else after.fingerprint
        if fa != fb: changed.append(name)
    return {'identity_migration':False, 'added':sorted(set(b)-set(a)), 'removed':sorted(set(a)-set(b)),
            'changed':changed, 'unchanged':sorted(set(a)&set(b)-set(changed)),
            'reset_required':before.fingerprint != after.fingerprint,
            'before':before.fingerprint,'after':after.fingerprint}


def identity_migration(before_events, after_events):
    """Review-only one-to-one links by episode coordinates, never an alias merge.

    Source epochs must be equal. A different tracker/pipeline epoch makes the
    episode incomparable and is deliberately left unmapped.
    """
    def coordinate(e):
        return (e['source'],e['epoch'],e['rule'],e['track_id'],float(e['start']))
    def index(events):
        result={}
        for event in events:
            key=coordinate(event)
            if key in result: raise ValueError('ambiguous migration episode coordinates')
            result[key]=event
        return result
    a,b=index(before_events),index(after_events)
    return {'schema_version':'identity-migration/1','automatic_aliasing':False,
            'links':[{'before_id':a[k]['id'],'before_version':a[k].get('identity_version'),
                      'after_id':b[k]['id'],'after_version':b[k].get('identity_version'),
                      'source':k[0],'epoch':k[1],'rule':k[2],'track_id':k[3],'start':k[4]}
                     for k in sorted(set(a)&set(b))],
            'unmapped_before':[a[k]['id'] for k in sorted(set(a)-set(b))],
            'unmapped_after':[b[k]['id'] for k in sorted(set(b)-set(a))]}
