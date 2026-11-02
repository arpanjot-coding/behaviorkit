"""Optional Apache-2.0 Trackers ByteTrack adapter, pinned to its tested API."""
from .core import Detection, number, text


class ByteTrackAdapter:
    def __init__(self, *, classes=('person',), sample_fps=5, max_tracks=200,
                 max_age_seconds=1, min_hits=2, activation=.5, high_confidence=.4, min_iou=.1):
        from trackers import ByteTrackTracker
        self.classes = tuple(classes)
        if not self.classes or len(set(self.classes))!=len(self.classes) or len(self.classes)>100:
            raise ValueError('ByteTrack requires 1..100 unique declared classes')
        for c in self.classes: text(c,'tracking class')
        number(sample_fps,'sample_fps',.001,120)
        number(max_age_seconds,'max_age_seconds',0,60)
        for name,v in [('activation',activation),('high_confidence',high_confidence),('min_iou',min_iou)]: number(v,name,0,1)
        if type(max_tracks) is not int or not 1<=max_tracks<=10000 or type(min_hits) is not int or min_hits<1:
            raise ValueError('invalid tracking capacity/min_hits')
        self.max_tracks, self.last_timestamp, self.last_diagnostics = max_tracks, None, ()
        self._settings = dict(lost_track_buffer=round(max_age_seconds*30),frame_rate=sample_fps,
                              minimum_consecutive_frames=min_hits,track_activation_threshold=activation,
                              high_conf_det_threshold=high_confidence,minimum_iou_threshold=min_iou)
        # Separate filters make association explicitly class-aware.
        self._trackers = {c:ByteTrackTracker(**self._settings) for c in self.classes}
        self.provenance = {'adapter':'trackers-bytetrack/1','trackers':'2.6.0','license':'Apache-2.0',
                           'settings':self._settings,'classes':self.classes,'max_tracks':max_tracks,
                           'output':'only current detector boxes with confirmed IDs; predictions never emitted'}

    def update(self, detections, timestamp):
        import numpy as np
        import supervision as sv
        number(timestamp,'timestamp')
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:
            raise ValueError('tracker timestamps must strictly increase')
        if not isinstance(detections,(tuple,list)) or not all(isinstance(d,Detection) for d in detections):
            raise ValueError('Detection sequence required')
        if len(detections)>self.max_tracks or any(d.label not in self.classes for d in detections):
            raise ValueError('tracker input capacity/class mismatch')
        output, tentative = [], 0
        for cls,tracker in self._trackers.items():
            selected = [i for i,d in enumerate(detections) if d.label==cls]
            batch = sv.Detections(xyxy=np.array([detections[i].box for i in selected],dtype=np.float32).reshape(-1,4)*1000,
                                  confidence=np.array([detections[i].confidence for i in selected],dtype=np.float32),
                                  data={'candidate_index':np.array(selected,dtype=np.int64)})
            result = tracker.update(batch,timestamp=float(timestamp))
            if result.tracker_id is None or len(result)==0: continue
            for tid,index in zip(result.tracker_id,result.data['candidate_index']):
                if tid<0:
                    tentative += 1; continue
                d = detections[int(index)]
                output.append(Detection(f'{cls}:{int(tid)}',d.label,d.confidence,d.box))
        if sum(len(t.tracks) for t in self._trackers.values())>self.max_tracks:
            for t in self._trackers.values(): t.reset()
            raise RuntimeError('ByteTrack state capacity exceeded; aborted stage, restart with a fresh epoch')
        self.last_timestamp = timestamp
        self.last_diagnostics = (f'tentative_candidates: {tentative}',) if tentative else ()
        return tuple(output)
