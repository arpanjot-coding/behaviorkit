"""Deterministic maximum-cardinality IoU baseline; no motion or appearance model."""
from collections import deque
from dataclasses import dataclass
from .core import Detection, number


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / max(1e-12, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


@dataclass
class _Track:
    box: tuple
    label: str
    last: float
    hits: int


class IoUTracker:
    def __init__(self, threshold=0.3, max_age_seconds=1.0, min_hits=2, max_tracks=200, *, association='max_cardinality', capacity_policy='evict_unmatched'):
        self.threshold = number(threshold, "iou threshold", 0, 1)
        self.max_age = number(max_age_seconds, "max_age_seconds", 0.001, 60)
        if type(min_hits) is not int or min_hits < 1 or type(max_tracks) is not int or max_tracks < 1:
            raise ValueError("min_hits and max_tracks must be positive integers")
        self.min_hits, self.max_tracks = min_hits, max_tracks
        if association not in {'max_cardinality','greedy'}:raise ValueError('association must be max_cardinality or greedy')
        if capacity_policy not in {'evict_unmatched','error'}:raise ValueError('capacity_policy must be evict_unmatched or error')
        self.association,self.capacity_policy=association,capacity_policy
        self.last_diagnostics=()
        self.tracks, self.next_id, self.last_timestamp = {}, 1, None

    def _assign(self,tracks,detections):
        edges={i:[] for i in range(len(detections))}
        for tid,t in sorted(tracks.items()):
            for idx,d in enumerate(detections):
                if t.label!=d.label:continue
                overlap=iou(t.box,d.box)
                if overlap>=self.threshold:edges[idx].append((-overlap,tid))
        edges={idx:sorted(values) for idx,values in edges.items()}
        assigned={}
        if self.association=='greedy':
            used=set()
            for _,tid,idx in sorted((score,tid,idx) for idx,values in edges.items() for score,tid in values):
                if tid not in used and idx not in assigned:assigned[idx]=tid;used.add(tid)
            return assigned
        # Augmenting paths maximize the number of existing identities retained.
        # IoU orders candidates deterministically; total IoU is not optimized.
        track_owner={}
        for root in edges:
            parents={root:None};visited=set();queue=deque([root]);found=False
            while queue and not found:
                current=queue.popleft()
                for _,tid in edges[current]:
                    if tid in visited:continue
                    visited.add(tid)
                    if tid not in track_owner:
                        node=current
                        while node is not None:
                            previous=assigned.get(node)
                            assigned[node]=tid;track_owner[tid]=node
                            tid=previous;node=parents[node]
                        found=True;break
                    child=track_owner[tid]
                    if child not in parents:parents[child]=current;queue.append(child)
        return assigned

    def update(self, detections, timestamp):
        number(timestamp, "timestamp")
        if not isinstance(detections,(tuple,list)) or not all(isinstance(d,Detection) for d in detections):
            raise ValueError('detections must be a list or tuple of Detection objects')
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("tracker timestamps must strictly increase")
        if len(detections) > self.max_tracks:
            raise ValueError("tracker detection capacity exceeded")
        tracks = {k:_Track(v.box,v.label,v.last,v.hits) for k,v in self.tracks.items() if timestamp-v.last <= self.max_age + 1e-9}
        next_id=self.next_id
        assigned=self._assign(tracks,detections)
        needed=len(detections)-len(assigned)
        retire=max(0,len(tracks)+needed-self.max_tracks)
        diagnostics=[]
        if retire:
            if self.capacity_policy=='error':raise ValueError('tracker active capacity exceeded')
            matched=set(assigned.values())
            candidates=sorted((t.last,tid) for tid,t in tracks.items() if tid not in matched)
            for _,tid in candidates[:retire]:
                del tracks[tid];diagnostics.append(f'track_evicted_for_capacity: {tid}')
        output = []
        for idx, d in enumerate(detections):
            tid = assigned.get(idx)
            if tid is None:
                if len(tracks) >= self.max_tracks:
                    raise ValueError("tracker active capacity exceeded")
                tid = next_id; next_id += 1
                tracks[tid] = _Track(d.box, d.label, timestamp, 0)
            t = tracks[tid]; t.box, t.last, t.hits = d.box, timestamp, t.hits + 1
            if t.hits >= self.min_hits:
                output.append(Detection(str(tid), d.label, d.confidence, d.box))
        self.tracks,self.next_id,self.last_timestamp=tracks,next_id,timestamp
        self.last_diagnostics=tuple(diagnostics)
        return tuple(output)
