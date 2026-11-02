# Python API

The core uses Python's standard library. Importing `behaviorkit` does not import NumPy, ONNX Runtime, PyTorch, OpenCV or a decoder.

```python
import json
from pathlib import Path
from behaviorkit import BehaviorEngine, SceneConfig, Detection, Observation

config = SceneConfig.from_dict(json.loads(Path("configs/people.json").read_text()))
engine = BehaviorEngine(config)

for index in range(21):
    batch = Observation(
        timestamp=index / 5,
        detections=(Detection("track-1", "person", 0.95, (0.2, 0.2, 0.4, 0.8)),),
        source="entrance",
        epoch="0",
    )
    result = engine.process(batch)
    for event in result.events:
        print(event)
```

## SceneConfig.from_dict(data)

Construct validated configuration from a versioned dictionary. Required fields are `schema_version`, `zones` and `rules`. Use the supplied JSON example as a starting point. Unsupported operators and unknown fields are errors. Direct `Rule` and `SceneConfig` construction is also validated. Geometry and rules are copied into immutable containers; later changes to caller lists cannot change an active scene. Zone polygons must be simple and have nonzero area.

## Detection(track_id, label, confidence, box)

Track IDs and labels are nonempty strings. Confidence is finite and between zero and one. Boxes are normalized XYXY with positive width and height. IDs must be unique within each observation.

## Observation(timestamp, detections, source, epoch, covered)

Timestamp is finite, nonnegative source time in seconds. Within one source epoch it must strictly increase. Epoch is a string that distinguishes source resets. An uncovered observation must have no detections.

## BehaviorEngine.process(observation)

Returns `Evaluation(counts, events, diagnostics, transitions)`. Events are emitted once per qualifying episode. `engine.state_size` exposes the active dwell-state count. The immutable effective configuration is available through `engine.config`; `engine.config_fingerprint` is its canonical SHA256.

Construct `BehaviorEngine(config, trace=True)` to receive optional structured starts, confirmations and resets in `result.transitions`. Reset reasons distinguish missing tracks, uncovered sources, zone exits, low confidence, changed labels, excessive time gaps and source epochs. Tracing does not change counts or events and retains no history inside the engine.

`engine.dwell_states(source=None)` returns owned copies of current timers with rule, source, epoch, track ID, start, last, duration and emitted status. Editing these returned dictionaries cannot change engine state.

`engine.close_source("entrance")` releases that source's clock, timers and capacity slot. It returns an Evaluation with reset transitions when tracing is enabled. Closing an absent source is harmless. A reopened source must use a fresh epoch; the engine cannot infer camera identity or clock continuity.

## Event identity migration in v0.3

Events contain `identity_version: 2` and `config_fingerprint`. The ID includes the effective configuration, rule, source, epoch, track and episode start. IDs change from v0.2 even for the same saved observations. Do not mix old and new IDs in a deduplication namespace without migrating deliberately. Equivalent JSON formatting and explicit default values produce the same effective fingerprint.

Independent video jobs receive fresh epochs unless `--epoch` is supplied. Exact replay of saved observations and configuration preserves v0.3 event IDs. Event identity is not a claim about unique physical people.

## Custom perception

Use your own detector/tracker to create observations with stable IDs, class names and normalized boxes. The engine does not require the supplied ONNX adapters. Keep the timestamps and gap policy consistent with your actual observation cadence.

Python adapters are trusted application code. Behaviour JSON is data only; the prototype never evaluates generated expressions or downloads executable code while processing frames.
