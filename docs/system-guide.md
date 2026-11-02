# Versioned pipelines and behavior operators

This guide describes the **0.4.0a1 experimental implementation**. Legacy scene `0.1`, observation JSONL and identity version 2 remain supported. Scene `0.2` opts into new operators and identity version 3. New recording directories use `recording/1` and `frame/1`.

The rule core still uses only Python's standard library. There are no ongoing LLM calls, API keys or automatic model downloads. A new operator being implemented does not establish its event accuracy on your cameras.

## Install only the needed components

```sh
python -m pip install -e ".[vision,cpu,custom,resources]"
```

Use `cuda` instead of `cpu` for NVIDIA, or `apple` for native Apple Silicon. Install one ONNX Runtime distribution per environment. ByteTrack is optional:

```sh
python -m pip install -e ".[tracking]"
```

The optional tracker brings NumPy, SciPy, Supervision, OpenCV and their dependencies. The standard IoU tracker needs none of these. The tracker package currently depends on desktop OpenCV, while the vision extra uses headless OpenCV. Keep their versions aligned if both distributions are installed; use a separate environment for tracker comparisons. The [tested recipe](../constraints/cpu-py312.txt) records the local versions, rather than promising every version permitted by dependency ranges.

## Capture once, replay in layers

```sh
behaviorkit capture --video input.mp4 --model d-fine-n --models-dir models --provider cpu --classes person --capture-floor 0.1 --sample-fps 5 --source entrance --epoch pilot-day-1 --output runs/candidates
behaviorkit recording-info --recording runs/candidates
behaviorkit track --recording runs/candidates --output runs/tracks
behaviorkit apply --recording runs/tracks --config configs/people.json --trace --output runs/rules
```

Each output directory must be new. A completed manifest records input, implementation and effective parameter hashes plus artifact sizes and SHA256 digests. Failed stages retain an error manifest and partial evidence. They are never valid cache hits. Integrity verification is independent of the code version that originally produced a recording; cache reuse additionally checks the current stage implementation hash. Hashes detect changes against the manifest, not a malicious author who can replace both data and manifest. They are not signatures.

`capture` retains **postprocessed candidate boxes**, not raw tensors, pixels, masks or embeddings. RTMDet uses classwise NMS at IoU 0.5 with up to 300 boxes per class. D-FINE retains its exported deploy selection; RF-DETR uses its query/class top-K selection. The latter models are NMS-free in this adapter. Each manifest records the exact adapter contract, class set, floor, sampling and model provenance. Do not describe every adapter as post-NMS when no NMS occurred.

The saved floor is an evidence limit. Capturing at 0.3 cannot support a later request for 0.1. Classes removed during capture, higher frame rates, changed image transforms, appearance features and discarded model outputs require inference again. Pixels are not retained merely because the original video exists elsewhere.

The baseline IoU replay has a separate effective association floor of `max(0.3, capture_floor)`. This keeps low-score experimental candidates from changing the default association. Its tracked recording declares that stricter floor. To experiment with a lower floor, save an explicit tracker parameter file:

```json
{"confidence_floor":0.1,"min_hits":2,"max_tracks":200}
```

```sh
behaviorkit track --recording runs/candidates --parameters tracker.json --output runs/low-score-tracks
behaviorkit track --recording runs/candidates --tracker bytetrack --output runs/motion-tracks
```

Changing tracking uses zero ONNX calls. The ByteTrack comparison requires capture floor at most 0.1. Its class-aware filters consume low scores for association, emit only current observed boxes with confirmed identities, and never emit predicted positions as visual confirmation. Timestamped updates use seconds. State overflow aborts the stage; it does not silently retain unlimited identities. The IoU tracker remains the default. Neither adapter proves that a track is a unique physical visitor.

Rule replay uses saved tracks and never calls a detector. Tracker parameters, capture provenance, retained frame bytes, tracking code and source/geometry segments participate in the derived tracking epoch. Changing a model/tracker must not inherit an earlier episode namespace.

## New rules and precise meanings

Validate and try [configs/operators.json](../configs/operators.json):

```sh
behaviorkit validate --config configs/operators.json
behaviorkit apply --recording runs/tracks --config configs/operators.json --trace --output runs/operators
python examples/system_demo.py --output runs/synthetic-operators
```

All boxes use normalized `x1,y1,x2,y2` coordinates in the decoded image. Zone membership uses the bottom-center anchor and includes polygon edges. Timestamp gaps are evaluated in source seconds, separately for each source epoch. Coordinates do not imply metres.

- `count`: number of currently observed, eligible confirmed tracks inside a zone. Uncovered samples yield null, not zero. Debouncing does not inflate visible counts.
- `dwell`: one event after sufficient supported time inside a zone. Strict membership resets on an observed exit. Missing identity, low confidence, unknown coverage and excessive gaps reset all continuity. This establishes sampled support, not continuous visibility between frames.
- `enter`: one event when membership changes from outside to inside. A first confirmed observation inside establishes an entry episode. This is not proof that the physical boundary crossing was filmed.
- `exit`: one event after an observed member becomes outside. Disappearance or unknown coverage resets state without inventing an exit.
- `cross`: a trajectory between two observed anchors crosses a finite line segment in the configured ROI. `positive` means the signed line-side value changes from negative to positive; reverse the line endpoints to reverse that convention. A tolerance dead band holds the last stable side. Reversals can produce separate crossings. Missing evidence resets the previous side; no crossing is interpolated across an unsupported gap.
- `occupancy`: one event when the number of eligible observed tracks stays at or above `threshold` for `seconds`. Falling below the threshold or losing coverage ends the episode.
- `sequence`: the same track enters the ordered `sequence_zones` before `sequence_timeout`. The rule's `zone` is its primary output label; membership is evaluated against the sequence polygons. Unexpected zones restart the sequence if they are the first zone, otherwise clear progress. Overlapping sequence zones are ambiguous and reset progress. Disappearance, low score, gap and unknown coverage also reset it.

For dwell/entry/exit, `membership: "debounced"` opts into `entry_seconds` and `exit_seconds`. Entry is confirmed only after observed inside samples support the entry duration. Exit grace can preserve a still-observed identity briefly outside the polygon. Dwell accumulates time only between consecutive supported inside samples; observed outside/grace time contributes nothing. Unknown or missing tracks never benefit from grace. These choices add confirmation delay and can hide true brief episodes, so compare them on labeled footage before choosing them.

## Inspect why a recorded candidate did not trigger

```sh
behaviorkit inspect --recording runs/candidates --config configs/people.json --source entrance --detail-limit 100 --output runs/candidate-inspection.json
behaviorkit inspect --recording runs/tracks --config configs/people.json --rule person_dwell --start 10 --end 20 --detail-limit 100 --output runs/dwell-inspection.json
```

Inspection has bounded counters and a hard detail limit. Optional source, rule, track and time filters narrow the report. Reasons include missing coverage, no retained candidate, class mismatch, low score, tentative or predicted status, outside anchor, untracked candidate, insufficient supported duration, current confirmation, already-emitted episode and trace reset reasons. Candidate-only inspection never pretends to know a stable track's duration. Inspection does not alter engine state or events.

An absent candidate cannot explain an unseen detector miss. It is not evidence that a real person was physically absent. Unsupported replay capability fails before evaluation instead of inventing a plausible explanation.

## Evaluate event quality

```sh
behaviorkit evaluate --annotations annotations.json --events runs/rules/events.jsonl --counts runs/rules/counts.jsonl --start-tolerance 0.5 --output runs/quality.json
```

Annotations use `evaluation/1`; see the packaged [schema](../src/behaviorkit/schemas/evaluation.schema.json). Provide explicit covered and unknown intervals per source/epoch, truth intervals with rule IDs, optional `eligible_at` confirmation targets, censored boundary episodes, and optional exact timestamp count samples. Camera/day group IDs separate tuning from held-out evaluation; `evaluation.validate_split_groups()` rejects a group shared across splits.

Events match one to one by source/epoch/rule and start tolerance, with optional minimum temporal IoU. Maximum-cardinality matching avoids losing a valid match to a greedy choice. Delay is prediction confirmation minus truth `eligible_at` (default truth end); negative delay is retained. Extra compatible predicted episodes are duplicates and false positives. Repeated transport delivery of the same prediction ID must be deduplicated first; the evaluator rejects repeated IDs.

The report includes TP/FP/FN, precision, recall, false alerts per scored camera-hour, delay samples, duplicates, ignored/censored episodes, count MAE/RMSE/bias, missing/unknown samples and coverage denominators. Gaps between declared windows count as unannotated unknown time, not negative footage. Nothing outside the declared recording extent is scored. Undefined ratios are null. Predictions touching unknown/censored evidence are excluded and counted explicitly.

`evaluation.evaluate_tracks()` reports matched/missed detections, false positives, ID switches and fragments on annotated frames using class-aware IoU matching. Unknown frames break continuity without penalties. These are explicit sampled-frame diagnostics, not HOTA or IDF1.

## Set up and bind each camera

```sh
behaviorkit setup --port 8765
```

Use `--port 0` if the host reserves port 8765. Open the printed loopback URL, load a local reference image, draw and save zones, choose an operator/cadence, validate, and download `scene.json` plus `geometry.json`. Media stays in the browser. The wizard creates one initial rule; use the validated schema/Python API to compose more rules. Sequence and directional-line editing remain configuration operations.

`setup.SceneRouter` binds each known source to its own scene and `Geometry`. Unknown sources fail. Reference width/height describe the image after the declared crop/rotation. Resizing with the same aspect is supported; different crop, rotation, aspect or revision requires revalidation. This metadata declares a transform; it does not secretly apply image transforms. Supply already-transformed matching images.

`router.cutover(source, scene, geometry, timestamp=..., reason=...)` returns an audit record and explicitly resets source state/generation. Persist that returned record if a durable cutover audit is needed; the router keeps only its last 100 records. There is no automatic file watcher or silent hot reload.

## Compare configuration and migrate identities

```sh
behaviorkit config-diff --before old.json --after new.json --output runs/diff.json
behaviorkit compare --recording runs/tracks --before old.json --after new.json --output runs/comparison
```

The full effective configuration fingerprint records provenance. Identity v3 separately hashes a rule's semantics and referenced geometry, then source/epoch/track/episode start. Unrelated rule edits do not rename the unchanged semantic episode during offline replay. Geometry, thresholds, temporal policy and rule changes alter the relevant fingerprint. Resource/pipeline changes require a new source/tracking epoch.

Legacy scene 0.1 keeps v2 identities and strict sampled behavior. Moving to scene 0.2 is explicit; old/new IDs never share a namespace. Comparison writes a review-only `identity-migration.json` linking unambiguous episode coordinates. It never merges deduplication namespaces or asserts that different epochs/tracker IDs identify the same physical person. Rolling back means applying the saved earlier scene to a new stage. Live cutovers always reset explicitly.

## Bring a custom model

Trusted Python adapters implement `DetectorAdapter`: declare `Capabilities` and immutable provenance, then return normalized `Detection` objects plus timing fields from `predict(uint8_RGB)`.

For a fixed-shape box model, `BoxOnnxAdapter` accepts a local ONNX file and `box-onnx/1` JSON contract: SHA256, class names, input/output names, input dimensions, box units, normalization, NMS IoU, detection limit and artifact license provenance. Install the `custom` extra for graph verification. Inputs must be `[1,3,H,W]` float32 RGB; outputs are boxes `[1,N,4]` and probability scores `[1,N,C]`. Classwise NMS is explicit. Self-contained files are required; external tensor data is rejected before session creation. There are no implicit sigmoid/objectness/layout guesses or executable configuration expressions.

```sh
behaviorkit capture --video input.mp4 --custom-model pallet.onnx --contract pallet-contract.json --provider cpu --capture-floor 0.1 --output runs/custom-candidates
```

Missing rule classes and lower unavailable thresholds fail before processing. Several rules consume the same inference pass. Pose/mask/action outputs are not consumed by these operators; do not label an adapter as supporting them merely because a network can emit them.

## Restart archive jobs safely

```sh
behaviorkit batch --videos first.mp4 second.mp4 --config configs/people.json --model d-fine-n --provider cpu --output runs/archive --cancel-file runs/stop
```

The job manifest records each input hash, stable per-file epoch, settings, attempts and result stages. One bad file does not stop the others. Create the cancellation file to stop at a safe processing boundary. Rerun the same command to skip verified completed files; add `--retry-failed` to retry errors. Changed input/settings require a new batch. Complete candidate stages can be reused after a downstream failure.

Partial files restart from frame zero, or reuse a completed perception stage. There is no hidden tracker/rule checkpoint restore. Stable epochs let consumers deduplicate repeated completed episodes. A canceled/error attempt is not a completed result. Use only a complete manifest for downstream scoring/delivery.

## Own, retain and redact artifacts

```sh
behaviorkit store --root runs/evidence-store --add runs/rules/events.jsonl --kind metadata --retention-seconds 604800
behaviorkit store --root runs/evidence-store --max-bytes 100000000 --max-trace-bytes 10000000
behaviorkit store --root runs/evidence-store --max-bytes 100000000 --apply
```

Only files copied into an empty/new owned store are eligible. The store never adopts a run directory, input video or model folder. Cleanup defaults to preview. Pin copies with `--pinned` or protect active owned paths using `--protect`. Root/path checks, ownership metadata, single-link regular files and hash verification reject path escapes, symlink/hardlink substitution and changed artifacts. Use exclusive application ownership of the store; it is not a defense against an attacker concurrently rewriting the filesystem/database.

Pinned/protected evidence can prevent a quota from being met; the report says so. Disk/permission errors propagate. Availability records explain when deleted evidence can no longer support replay. `owned_store.redact_rgb()` makes a separate solid-filled export copy for explicit normalized regions. It neither changes inference input nor promises automatic detection of every sensitive feature.

## Experimental live sources and durable events

`behaviorkit live --sources sources.json --model d-fine-n --provider cuda --seconds 60 --output runs/live --outbox runs/outbox.sqlite3`

A source file has a `sources` list. Each entry contains `id`, `uri`, a scene path relative to the source file, `geometry` reference metadata, optional `sample_fps`, `policy`, queue/timeouts/retry settings and `reconnect_at_eof`. Keep credentials out of Git. Source URIs and decoder exception text are not written to health records.

PyAV/FFmpeg decodes in isolated, killable processes. Every connection has a new epoch. Sources are polled fairly; a decoder that stalls cannot monopolize another decoder's queue. The watchdog terminates a persistently stuck decoder; applications can start a new run after terminal failure. Read/open errors get bounded exponential retries. Clean EOF reconnects by default for network sources and ends local files.

`latest` replaces queued old frames and reports drops. `lossless` applies bounded backpressure inside this adapter; it cannot guarantee network/camera delivery. Dropped or missing intervals still affect rule continuity. Age checks occur before and after inference. Reported age begins at local decoded-frame receipt, not camera capture UTC or network transit. Stale/recovering/failed sources reset tracking/rule continuity. Every later segment receives a new namespace. One shared detector processes frames serially; synchronous native inference can delay all consumers. A decoder watchdog does not make an uninterruptible GPU inference call safe. This is an experimental adapter, not a production availability claim.

Live recordings retain the exact bound tracked observations for rule replay, plus counts, events, health and timings. Replay each saved camera scene with `behaviorkit apply --recording runs/live --saved-bindings --trace --output runs/live-replayed`. The live IoU floor defaults to `max(0.3,capture_floor)`; override it explicitly with `--tracking-floor` and keep rule thresholds compatible. `run_status` distinguishes completed, partial failure and zero-frame failure; manifest `complete` means artifacts finished and passed integrity checks, not that every source was healthy. Relative video PTS cannot establish business hours.

The SQLite `Outbox` stores raw behavior facts independently of notification decisions. It uses stable message IDs, transactional leases, retries and restart recovery. Delivery is **at least once**; consumers deduplicate message IDs. Same event ID with conflicting payload fails. The first sink is durable local JSONL:

```sh
behaviorkit outbox --database runs/outbox.sqlite3 --deliver-jsonl runs/delivered.jsonl
```

`NotificationPolicy` supports explicit UTC minute windows, cooldowns grouped by rule/source/destination and one expiring boolean external fact. Unknown wall-clock/context suppresses notification while retaining the fact. Pass external fact observation and expiry UTC explicitly. Decisions are frozen per event/policy revision; change the policy ID/settings for intentional reevaluation. Suppression never deletes the raw fact. Pending/fact limits fail explicitly and require draining or application-managed archival; arbitrary HTTP/webhook sinks are not included.

## Measure resources and select a cost profile

Add `--monitor-resources` to capture/batch/live after installing `resources`. Logs stream RSS, native thread count, accessible child processes, handles/FDs and optional NVIDIA per-process VRAM queries. Missing VRAM is null, especially where WDDM does not expose it. The sampler has a hard record limit; GPU queries add measurement overhead.

`plan-budget` selects only supplied measured profiles with matching hardware/dataset IDs, artifact digests and explicit quality/latency/RSS budgets. It returns no selection if evidence is missing or budgets fail. It does not invent camera capacity from model FPS.

`profiles.export_precision_candidate()` performs opt-in FP16 or calibrated INT8 export while keeping float32 public I/O. INT8 requires a representative ORT calibration reader. Exports remain `unqualified`; they are not automatically installed in presets. Compare provider execution, positive/negative fixtures, held-out event quality and full pipeline costs before promotion. See [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) and [float16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

Use `scripts/validate_system.py` for the learned-model capture/replay regression. A separate two-frame provider profile avoids retaining profiler history during capture/resource testing. Continuous ORT profiling can grow native memory and is deliberately disabled during ordinary capture and soaks.

## Qualification limits

Generated fixtures and hand-computable rule tests establish the declared software contracts. Controlled learned-model runs on the recorded Intel/NVIDIA machine establish narrow model/pipeline compatibility. They do not establish behavior accuracy on real surveillance footage, all three models on AMD/M1, multi-hour camera availability, calibrated INT8/FP16 accuracy or general theft/intent recognition.

The [pilot register](pilot-requirements.json) and [implementation status](implementation-status.md) keep those external acceptance gates visible. Supply representative positive/negative clips, independent camera/day splits and agreed event/resource budgets before calling a deployment qualified.
