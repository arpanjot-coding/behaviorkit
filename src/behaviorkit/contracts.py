"""Versioned JSON contracts shared by optional pipelines. No ML imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Protocol

from .core import Detection, Observation, keys, number, text

MAX_LINE_BYTES = 8 * 1024 * 1024


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path, limit=1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'JSON document exceeds {limit} bytes')
    return json.loads(raw)


def read_jsonl(path) -> Iterator[dict]:
    with Path(path).open('rb') as stream:
        line = 0
        while raw := stream.readline(MAX_LINE_BYTES + 1):
            line += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f'line {line} exceeds {MAX_LINE_BYTES} bytes')
            if raw.strip():
                try:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError('record must be an object')
                    yield value
                except (ValueError, TypeError) as error:
                    raise ValueError(f'line {line}: {error}') from error


def json_line(stream, value):
    stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')


@dataclass(frozen=True)
class Capabilities:
    """What a retained stage can actually reproduce, not its model's potential."""
    classes: tuple[str, ...]
    confidence_floor: float
    stage: str = 'postprocessed_candidates'
    pixels: bool = False
    embeddings: bool = False
    sample_fps: float | None = None
    max_detections: int = 10000

    def __post_init__(self):
        if self.stage not in {'postprocessed_candidates', 'tracked_observations'}:
            raise ValueError('unsupported recording stage')
        if not isinstance(self.classes, (list, tuple)) or not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('classes must be unique nonempty class names')
        object.__setattr__(self, 'classes', tuple(text(c, 'class') for c in self.classes))
        number(self.confidence_floor, 'confidence_floor', 0, 1)
        if self.sample_fps is not None:
            number(self.sample_fps, 'sample_fps', .001, 120)
        if type(self.pixels) is not bool or type(self.embeddings) is not bool:
            raise ValueError('pixels and embeddings must be boolean')
        if type(self.max_detections) is not int or not 1 <= self.max_detections <= 10000:
            raise ValueError('max_detections must be in 1..10000')

    def require(self, *, classes, confidence, sample_fps=None, pixels=False, embeddings=False, tracker=False):
        missing = set(classes) - set(self.classes)
        if missing:
            raise ValueError(f'unavailable classes: {sorted(missing)}')
        number(confidence, 'requested confidence', 0, 1)
        if confidence < self.confidence_floor - 1e-12:
            raise ValueError('requested confidence is below retained capture floor')
        if sample_fps is not None and (self.sample_fps is None or sample_fps > self.sample_fps + 1e-9):
            raise ValueError('requested cadence is unavailable; inference rerun required')
        if pixels and not self.pixels or embeddings and not self.embeddings:
            raise ValueError('requested pixels/embeddings were not retained')
        if tracker and self.stage != 'postprocessed_candidates':
            raise ValueError('tracker replay requires candidate detections')

    def to_dict(self):
        value=asdict(self)
        value['classes']=list(self.classes)
        return value


@dataclass(frozen=True)
class EvidenceFrame:
    observation: Observation
    width: int
    height: int
    geometry_revision: str = '1'
    clock: str = 'video_pts'
    observed_utc: float | None = None
    statuses: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.observation, Observation):
            raise ValueError('observation required')
        for value in (self.width, self.height):
            if type(value) is not int or not 1 <= value <= 32768:
                raise ValueError('frame dimensions must be in 1..32768')
        text(self.geometry_revision, 'geometry_revision')
        if self.clock not in {'video_pts', 'monotonic'}:
            raise ValueError('clock must be video_pts or monotonic')
        if self.observed_utc is not None:
            number(self.observed_utc, 'observed_utc', 0)
        ids = {d.track_id for d in self.observation.detections}
        if len(dict(self.statuses)) != len(self.statuses) or any(k not in ids or v not in {'observed', 'tentative', 'predicted'} for k, v in self.statuses):
            raise ValueError('invalid track statuses')
        object.__setattr__(self, 'statuses', tuple(tuple(x) for x in self.statuses))

    def to_dict(self):
        o=self.observation
        observation={'timestamp':o.timestamp,'source':o.source,'epoch':o.epoch,'covered':o.covered,
                     'detections':[{'track_id':d.track_id,'label':d.label,'confidence':d.confidence,'box':list(d.box)} for d in o.detections]}
        return {'schema_version': 'frame/1', 'observation': observation,
                'width': self.width, 'height': self.height, 'geometry_revision': self.geometry_revision,
                'clock': self.clock, 'observed_utc': self.observed_utc, 'statuses': dict(self.statuses)}

    @classmethod
    def from_dict(cls, data):
        keys(data, {'schema_version', 'observation', 'width', 'height', 'geometry_revision', 'clock', 'observed_utc', 'statuses'}, 'frame', {'schema_version', 'observation', 'width', 'height'})
        if data['schema_version'] != 'frame/1':
            raise ValueError('unsupported frame schema')
        return cls(Observation.from_dict(data['observation']), data['width'], data['height'],
                   data.get('geometry_revision', '1'), data.get('clock', 'video_pts'),
                   data.get('observed_utc'), tuple(data.get('statuses', {}).items()))

    def confirmed_observation(self):
        """Predictions and tentative identities cannot become observed evidence."""
        status = dict(self.statuses)
        o = self.observation
        return Observation(o.timestamp, tuple(d for d in o.detections if status.get(d.track_id, 'observed') == 'observed'), o.source, o.epoch, o.covered)


class DetectorAdapter(Protocol):
    """Trusted Python adapters return normalized RGB-derived boxes and timings."""
    capabilities: Capabilities
    provenance: dict

    def predict(self, rgb) -> tuple[tuple[Detection, ...], dict]: ...


class TrackerAdapter(Protocol):
    last_diagnostics: tuple[str, ...]

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Detection, ...]: ...
