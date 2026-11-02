# Counting and dwell

These rules describe observable evidence. They do not infer intentions, identities across cameras, or physical occupancy behind occlusions.

## Zone coordinates

Zones use normalized image coordinates from 0 to 1. The origin is top-left. Each box is `[x1,y1,x2,y2]`. The rule engine tests its **bottom-center** anchor against the polygon, including the polygon boundary.

The prototype expects a simple, non-self-intersecting polygon. It checks vertex counts, ranges, nonzero area and non-adjacent edge intersections. There is no entry/exit hysteresis yet, so boundary jitter can reset dwell. Place the initial zone away from unstable image boundaries.

## Visible people count

At each covered observation, count current tracks with the selected class, sufficient confidence and an anchor inside the zone. The count is about observed confirmed tracks. It is not a count of unique daily visitors or people hidden from view.

An observation with `covered:false` returns `count:null` with `status:unknown`. It is never treated as an empty scene. A covered batch containing no eligible detections returns zero; detector misses remain possible.

## Dwell interval

A dwell rule binds one track in one source epoch. Its timer starts at the first qualifying observation. It emits once when the supported interval reaches the configured number of seconds, provided successive samples never exceed `max_gap_seconds`.

```json
{
  "id": "person_dwell",
  "kind": "dwell",
  "zone": "waiting_area",
  "label": "person",
  "seconds": 3,
  "max_gap_seconds": 0.5,
  "confidence": 0.3
}
```

A missing track, zone exit, insufficient confidence, uncovered batch, excessive observation gap or source-epoch change resets the interval. Re-entering starts a fresh episode. Leaving a wall-clock timer running does not establish continued presence.

At 5 Hz, fresh samples from 0.0 through 3.0 seconds can support a three-second dwell event. Samples at 0.0 and 3.0 alone cannot: the gap exceeds the policy. Bridging intervals between sufficiently close samples is an explicit sampling assumption, not continuous visual observation.

## Tracker baseline

Video processing uses deterministic maximum-cardinality IoU association, class agreement, two-hit confirmation and one-second expiry. Augmenting paths retain the greatest number of feasible existing matches, with IoU ordering candidates. This does not optimize total IoU or establish real-world identity correctness. Positions come only from actual detections. The tracker does not predict unseen people or recover an identity after a long occlusion.

This small baseline can still switch IDs in crowded scenes. Those switches affect counts and dwell recall. Replace it through timestamped observation input or a future tracker adapter after evaluating a better association model. Python callers can select `association="greedy"` for a controlled baseline comparison.

## Resource bounds

Defaults are 200 detections/tracks per batch, eight sources, 100 rules and 100 zones. More current detections than capacity raises an error. To admit observed detections while preserving the tracker bound, the default tracker evicts oldest unmatched retained tracks and reports their IDs in `last_diagnostics`. Already matched tracks are preserved. Python callers can request `capacity_policy="error"` for strict rejection; rejected updates do not advance state.

The engine retains only active dwell state and source clocks, not frames or event history. Use `close_source(source)` to release a retired source slot, and a new epoch when reopening it. Enable tracing for explicit reset reasons. The strict missing-observation and boundary policies remain unchanged in v0.3; a better matching algorithm does not justify asserting presence during occlusion.
