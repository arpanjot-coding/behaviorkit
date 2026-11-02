"""Counting and dwell semantics over ordered, normalized observations."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping
from types import MappingProxyType
import hashlib
import json
import math
from typing import Any


def number(value: Any, name: str, low: float = 0, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return result


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError(f"{name} must be a nonempty string of at most 200 characters")
    return value


def keys(data: dict, allowed: set[str], name: str, required=()) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object")
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"{name}: unsupported fields {sorted(extra)}")
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"{name}: missing required fields {sorted(missing)}")


def sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return value


def polygon(value):
    vertices=sequence(value,"polygon")
    points=[]
    for point in vertices:
        if len(sequence(point,"polygon vertex")) != 2:
            raise ValueError("polygon vertices must be x,y pairs")
        points.append(tuple(number(v,"polygon coordinate",0,1) for v in point))
    if len(points)>3 and points[0]==points[-1]:points.pop()
    if not 3 <= len(points) <= 100:
        raise ValueError("zones require 3..100 polygon vertices")
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,p):return abs(cross(a,b,p))<=1e-12 and min(a[0],b[0])-1e-12<=p[0]<=max(a[0],b[0])+1e-12 and min(a[1],b[1])-1e-12<=p[1]<=max(a[1],b[1])+1e-12
    def intersect(a,b,c,d):
        return ((cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b))
    for i,a in enumerate(points):
        b=points[(i+1)%len(points)]
        if a==b:raise ValueError("polygon cannot contain zero-length edges")
        for j in range(i+1,len(points)):
            if j==i+1 or (i==0 and j==len(points)-1):continue
            if intersect(a,b,points[j],points[(j+1)%len(points)]):
                raise ValueError("polygon must be simple and non-self-intersecting")
    area=sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))
    if abs(area)<1e-10:raise ValueError("polygon must have nonzero area")
    return tuple(points)


def inside(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Ray casting, with the polygon boundary included (normalized coordinates)."""
    x, y = point
    result = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= 1e-12 and min(ax, bx) - 1e-12 <= x <= max(ax, bx) + 1e-12 and min(ay, by) - 1e-12 <= y <= max(ay, by) + 1e-12:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            result = not result
    return result


@dataclass(frozen=True)
class Detection:
    track_id: str
    label: str
    confidence: float
    box: tuple[float, float, float, float]

    def __post_init__(self):
        text(self.track_id, "track_id")
        text(self.label, "label")
        number(self.confidence, "confidence", 0, 1)
        if len(sequence(self.box,"box")) != 4:
            raise ValueError("box must contain normalized x1,y1,x2,y2")
        for v in self.box:
            number(v, "box coordinate", 0, 1)
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError("box must have positive width and height")
        object.__setattr__(self,"box",tuple(float(v) for v in self.box))

    @property
    def anchor(self):
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


@dataclass(frozen=True)
class Observation:
    timestamp: float
    detections: tuple[Detection, ...] = ()
    source: str = "camera"
    epoch: str = "0"
    covered: bool = True

    def __post_init__(self):
        number(self.timestamp, "timestamp")
        text(self.source, "source")
        text(self.epoch, "epoch")
        if type(self.covered) is not bool:
            raise ValueError("covered must be a boolean")
        sequence(self.detections,"detections")
        if not all(isinstance(d,Detection) for d in self.detections):
            raise ValueError("detections must contain Detection objects")
        object.__setattr__(self,"detections",tuple(self.detections))
        if not self.covered and self.detections:
            raise ValueError("uncovered observations cannot contain detections")
        if len(self.detections) > 10000:
            raise ValueError("observation exceeds the 10000 detection input limit")
        ids = [d.track_id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("track IDs must be unique within an observation")

    @classmethod
    def from_dict(cls, data):
        keys(data, {"timestamp", "source", "epoch", "covered", "detections"}, "observation", {"timestamp"})
        detections = []
        for d in sequence(data.get("detections", []),"detections"):
            keys(d, {"track_id", "label", "confidence", "box"}, "detection", {"track_id","label","confidence","box"})
            detections.append(Detection(d["track_id"], d["label"], d["confidence"], d["box"]))
        return cls(data["timestamp"], tuple(detections), data.get("source", "camera"), data.get("epoch", "0"), data.get("covered", True))


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str
    zone: str
    label: str = "person"
    confidence: float = 0.3
    seconds: float = 20.0
    max_gap_seconds: float = 0.5

    def __post_init__(self):
        text(self.id,"rule id");text(self.zone,"zone");text(self.label,"label")
        if self.kind not in ("count","dwell"):raise ValueError("rule kind must be count or dwell")
        number(self.confidence,"confidence",0,1)
        number(self.seconds,"seconds",0.001,86400)
        number(self.max_gap_seconds,"max_gap_seconds",0.001,60)


@dataclass(frozen=True)
class SceneConfig:
    zones: dict[str, tuple[tuple[float, float], ...]]
    rules: tuple[Rule, ...]
    max_tracks: int = 200
    max_sources: int = 8

    def __post_init__(self):
        if not isinstance(self.zones,Mapping) or not 1<=len(self.zones)<=100:
            raise ValueError("zones must be an object containing 1..100 polygons")
        owned={text(name,"zone name"):polygon(vertices) for name,vertices in self.zones.items()}
        rules=tuple(sequence(self.rules,"rules"))
        if not 1<=len(rules)<=100 or not all(isinstance(r,Rule) for r in rules):
            raise ValueError("config requires 1..100 Rule objects")
        if len({r.id for r in rules})!=len(rules):raise ValueError("rule IDs must be unique")
        for rule in rules:
            if rule.zone not in owned:raise ValueError(f"unknown zone {rule.zone!r}")
        for name,value,maximum in [("max_tracks",self.max_tracks,10000),("max_sources",self.max_sources,100)]:
            if type(value) is not int or not 1<=value<=maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        object.__setattr__(self,"zones",MappingProxyType(owned))
        object.__setattr__(self,"rules",rules)

    def to_dict(self):
        """Owned, JSON-compatible configuration including effective defaults."""
        rules=[]
        for rule in self.rules:
            item=asdict(rule)
            for key in ('confidence','seconds','max_gap_seconds'):item[key]=float(item[key])
            rules.append(item)
        return {'schema_version':'0.1','zones':{name:[list(p) for p in points] for name,points in self.zones.items()},'rules':rules,'max_tracks':self.max_tracks,'max_sources':self.max_sources}

    @property
    def fingerprint(self):
        payload=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @classmethod
    def from_dict(cls, data):
        keys(data, {"schema_version", "zones", "rules", "max_tracks", "max_sources"}, "config", {"schema_version","zones","rules"})
        if data.get("schema_version") != "0.1":
            raise ValueError('schema_version must be "0.1"')
        rules = []
        for item in sequence(data["rules"],"rules"):
            keys(item, {"id", "kind", "zone", "label", "confidence", "seconds", "max_gap_seconds"}, "rule", {"id","kind","zone"})
            rules.append(Rule(**item))
        return cls(data["zones"],tuple(rules),data.get("max_tracks",200),data.get("max_sources",8))


@dataclass(frozen=True)
class Evaluation:
    counts: tuple[dict, ...]
    events: tuple[dict, ...]
    diagnostics: tuple[str, ...] = ()
    transitions: tuple[dict, ...] = ()


@dataclass
class _Dwell:
    start: float
    last: float
    emitted: bool = False


class BehaviorEngine:
    """One-pass evaluation with bounded active state and no retained frame/event history.

    Bottom-center anchors; polygon boundaries included; no spatial hysteresis in v0.1.
    A missing track, uncovered batch, epoch change, or excessive gap resets dwell.
    """
    def __init__(self, config: SceneConfig, *, trace=False):
        if not isinstance(config,SceneConfig):raise ValueError('config must be a SceneConfig')
        if type(trace) is not bool:raise ValueError('trace must be a boolean')
        self._config = config
        self._config_fingerprint=config.fingerprint
        self.trace=trace
        self._rules={r.id:r for r in config.rules}
        self._sources: dict[str, tuple[str, float]] = {}
        self._dwell: dict[tuple[str, str, str, str], _Dwell] = {}

    @property
    def config(self):
        return self._config

    @property
    def config_fingerprint(self):
        return self._config_fingerprint

    def dwell_states(self, source=None):
        """Copies of current sampled dwell state; no frame/history retention."""
        if source is not None:text(source,'source')
        return tuple(self._state_record(key,state) for key,state in sorted(self._dwell.items()) if source is None or key[1]==source)

    @staticmethod
    def _state_record(key,state):
        return {'rule':key[0],'source':key[1],'epoch':key[2],'track_id':key[3],'start':state.start,'last':state.last,'duration':state.last-state.start,'emitted':state.emitted}

    def close_source(self, source):
        """Release source clocks and dwell. Reopen with a fresh observation epoch."""
        text(source,'source')
        existed=source in self._sources
        transitions=tuple({**self._state_record(k,v),'transition':'reset','reason':'source_closed'} for k,v in sorted(self._dwell.items()) if k[1]==source) if self.trace else ()
        self._dwell={k:v for k,v in self._dwell.items() if k[1]!=source}
        self._sources.pop(source,None)
        return Evaluation((),(),('source_closed; dwell reset',) if existed else (),transitions)

    @property
    def state_size(self):
        return len(self._dwell)

    def process(self, observation: Observation) -> Evaluation:
        if not isinstance(observation,Observation):raise ValueError('observation must be an Observation')
        o = observation
        transitions=[]
        diagnostics=[]
        prior = self._sources.get(o.source)
        if prior and prior[0] == o.epoch and o.timestamp <= prior[1]:
            raise ValueError("timestamps must strictly increase within a source epoch")
        if not prior and len(self._sources) >= self.config.max_sources:
            raise ValueError("source capacity exceeded; start a separate engine")
        if len(o.detections) > self.config.max_tracks:
            raise ValueError("track capacity exceeded; observation was not evaluated")
        if prior and prior[0] != o.epoch:
            if self.trace:
                transitions.extend({**self._state_record(k,v),'transition':'reset','reason':'epoch_changed','timestamp':o.timestamp} for k,v in sorted(self._dwell.items()) if k[1]==o.source)
            self._dwell = {k: v for k,v in self._dwell.items() if k[1] != o.source}
            diagnostics.append('source_epoch_changed; dwell reset')
        self._sources[o.source] = (o.epoch, o.timestamp)
        counts, events, active = [], [], set()
        selection_cache = {}
        for rule in self.config.rules:
            selector = (rule.zone, rule.label, rule.confidence)
            if selector not in selection_cache:
                selection_cache[selector] = [d for d in o.detections if o.covered and d.label == rule.label and d.confidence >= rule.confidence and inside(d.anchor, self.config.zones[rule.zone])]
            selected = selection_cache[selector]
            if rule.kind == "count":
                counts.append({"rule":rule.id, "source":o.source, "epoch":o.epoch, "timestamp":o.timestamp, "count":len(selected) if o.covered else None, "status":"observed" if o.covered else "unknown", "track_ids":[d.track_id for d in selected]})
                continue
            for d in selected:
                key = (rule.id, o.source, o.epoch, d.track_id)
                active.add(key)
                state = self._dwell.get(key)
                if state is None or o.timestamp - state.last > rule.max_gap_seconds + 1e-9:
                    if self.trace and state is not None:transitions.append({**self._state_record(key,state),'transition':'reset','reason':'observation_gap','timestamp':o.timestamp})
                    state = self._dwell[key] = _Dwell(o.timestamp, o.timestamp)
                    if self.trace:transitions.append({**self._state_record(key,state),'transition':'started','reason':'eligible_observation','timestamp':o.timestamp})
                state.last = o.timestamp
                duration = o.timestamp - state.start
                if not state.emitted and duration + 1e-9 >= rule.seconds:
                    state.emitted = True
                    identity=json.dumps(['behaviorkit-event-v2',self.config_fingerprint,rule.id,o.source,o.epoch,d.track_id,float(state.start)],separators=(",",":"),ensure_ascii=False).encode("utf-8")
                    event_id=hashlib.sha256(identity).hexdigest()
                    events.append({"id":event_id, "identity_version":2,"config_fingerprint":self.config_fingerprint,"kind":"person_dwell" if rule.label == "person" else "object_dwell", "rule":rule.id, "source":o.source, "epoch":o.epoch, "track_id":d.track_id, "zone":rule.zone, "start":state.start, "timestamp":o.timestamp, "duration_seconds":duration, "status":"confirmed_by_rule", "evidence":{"anchor":"bottom_center", "confidence":d.confidence, "box":d.box, "max_gap_seconds":rule.max_gap_seconds, "unknown_policy":"reset", "interpretation":"sampled zone membership; no intent inferred"}})
                    if self.trace:transitions.append({**self._state_record(key,state),'transition':'confirmed','reason':'duration_reached','timestamp':o.timestamp,'event_id':event_id})
        if self.trace:
            by_id={d.track_id:d for d in o.detections}
            for key,state in sorted(self._dwell.items()):
                if key[1]!=o.source or key in active:continue
                d=by_id.get(key[3]);rule=self._rules[key[0]]
                reason=('source_not_covered' if not o.covered else 'track_missing' if d is None else 'label_changed' if d.label!=rule.label else 'confidence_below_threshold' if d.confidence<rule.confidence else 'zone_exit')
                transitions.append({**self._state_record(key,state),'transition':'reset','reason':reason,'timestamp':o.timestamp})
        self._dwell = {k:v for k,v in self._dwell.items() if k[1] != o.source or k in active}
        if not o.covered:diagnostics.append('source_not_covered; dwell reset')
        return Evaluation(tuple(counts),tuple(events),tuple(diagnostics),tuple(transitions))
