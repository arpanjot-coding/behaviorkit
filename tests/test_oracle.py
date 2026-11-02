"""Batch/reference checks independent of the incremental implementation."""
import random
from behaviorkit import BehaviorEngine,SceneConfig,Observation,Detection


def test_randomized_stream_matches_offline_episode_oracle():
    rng=random.Random(4912)
    config=SceneConfig.from_dict({'schema_version':'0.1','zones':{'z':[[.1,.1],[.8,.1],[.8,.8],[.1,.8]]},'rules':[{'id':'n','kind':'count','zone':'z'},{'id':'d','kind':'dwell','zone':'z','seconds':.6,'max_gap_seconds':.3}]})
    engine=BehaviorEngine(config);frames=[];events=[]
    for i in range(400):
        timestamp=i*.2;covered=rng.random()>.03;detections=[];expected_ids=set()
        if covered:
            for track in range(4):
                if rng.random()<.85:
                    in_zone=rng.random()<.9;confidence=.9 if rng.random()<.95 else .1
                    box=(.2,.2,.4,.6) if in_zone else (.85,.2,.95,.95)
                    detections.append(Detection(str(track),'person',confidence,box))
                    if in_zone and confidence>=.3:expected_ids.add(str(track))
        frame=Observation(timestamp,tuple(detections),covered=covered)
        evaluation=engine.process(frame)
        assert evaluation.counts[0]['count']==(len(expected_ids) if covered else None)
        frames.append((timestamp,expected_ids if covered else set()))
        events.extend((e['track_id'],round(e['timestamp'],6)) for e in evaluation.events)
    expected=[]
    # Each track's truth series is grouped in batch; no engine state/timers used.
    for track in map(str,range(4)):
        qualifying=[i for i,(_,ids) in enumerate(frames) if track in ids]
        groups=[]
        for index in qualifying:
            if not groups or index!=groups[-1][-1]+1:groups.append([])
            groups[-1].append(index)
        for group in groups:
            eligible=[i for i in group if frames[i][0]-frames[group[0]][0]>=.6-1e-9]
            if eligible:expected.append((track,round(frames[eligible[0]][0],6)))
    assert sorted(events)==sorted(expected)
