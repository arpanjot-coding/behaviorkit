# BehaviorKit implementation plan

Proposed plan based on the v0.3.0 implementation and recorded evidence, 2 November 2026. Milestones are dependency gates, not delivery-date promises. Start with fixed-camera counting and dwell; expand only when a use case has a measurable acceptance target.

## Objective

Make the library quick to set up, explicit about missing evidence, inexpensive to operate, and reproducible across qualified hardware. Preserve local inference, no ongoing LLM calls, optional heavyweight dependencies, bounded state and separate provenance for code, weights and media.

## Baseline to preserve

- v0.3 freezes run configuration, evaluates file observations incrementally, records hashes, exposes dwell traces/source cleanup and uses bounded maximum-cardinality IoU association.
- Recorded local CPU/CUDA environments each passed 80 tests with one Apple-only skip. All three learned models passed controlled counting/dwell and exact replay on the Intel/NVIDIA host.
- Public CI passed core/package jobs and generated-fixture integration on AMD Windows/Linux runners and an Apple M1 virtual runner. The M1 suite passed 81 tests, including tiny CoreML delegation. This is not all-three-learned-model qualification on AMD/M1.
- The exploratory video lab has no held-out event accuracy guarantee. Missing observations and zone exits still reset strict dwell. There is no live RTSP service or theft classifier.

Sources: [README](../README.md), [recorded reliability](../benchmarks/reliability-v03.json), [CI evidence](../benchmarks/github-actions-v03.json), [existing roadmap](../ROADMAP.md).

## Operating rules

1. Use one canonical library source repository. Keep model files, local runs and private hosting state outside tracked source; generate release/download copies from a tagged revision.
2. Preserve the standard-library rule core. Perception, visualization, media and delivery packages remain optional.
3. Never reinterpret a historical recording silently. Version observation, recording, configuration and event identity contracts explicitly.
4. Treat uncertainty as data. Missing coverage, unsupported capability and observed absence are different outcomes.
5. Evaluate one behavioral change at a time. A new tracker, hysteresis, sampling rate and precision should not arrive as one unexplained accuracy experiment.
6. Agree use-case precision, recall, false-alert, delay and resource budgets before tuning. Do not invent a universal acceptable accuracy threshold.

## M0 Establish requirements and freeze contracts

**Purpose:** prevent incompatible foundations and establish what success means.

Work:

- Freeze a v0.3 golden set of observations, configuration, events, counts and traces, including known failures and identity migration inputs.
- Select initial pilot workflows: visible count and observed dwell first. Record normal lookalikes, camera conditions, consumers and failure costs.
- Define a recording manifest, evidence stages, coordinate units, source clocks/epochs, coverage states and replay capabilities.
- Decide event identity dependencies before implementing configuration updates: full run provenance, rule-semantic revision and tracking/source epoch have different purposes.
- Maintain a claim matrix with separate evidence levels: core correctness, generated-fixture integration, learned-model compatibility, camera-event quality and sustained deployment.

Done when:

- Every advertised use case maps to an event contract, representative data, hardware recipe, measurable budget and documentation example.
- Examples distinguish rule replay, tracker replay and model inference rerun.
- Schema defaults, limits, migrations and failure behavior are specified; legacy v0.3 files remain readable without silent reinterpretation.

Dependency: none. Dataset collection and hardware access can proceed in parallel with later implementation.

## M1 Build evaluation and artifact trust foundations

### Event evaluation

Implement an offline evaluator and a documented annotation format. Specify interval matching, temporal tolerances, duplicate episodes, count sampling, censored events and ambiguous/unknown coverage before scoring.

Include positives, normal lookalikes, empty scenes, occlusion, ID reuse, boundary jitter, source gaps and short entries. Split tuning and evaluation by camera/day or another independent boundary, not adjacent frames.

Report count error, event precision/recall, false alerts per camera-hour, confirmation delay, duplicates and uncertain coverage. Include denominators and enough negative footage to assess the agreed false-alert budget. Link failures to replayable evidence.

Done when hand-calculable fixtures match independent expected scores, held-out thresholds are frozen, and each pilot's quality budget is recorded. Existing annotation/evaluation tools can be integrated without making them core dependencies.

### Approved artifacts and installation recipes

Add an approved artifact catalog with checkpoint digests, publisher URL/revision, exact model/license variant, export recipe and expected contracts. Verify cached/downloaded bytes before invoking checkpoint loaders. A digest recorded after loading is not an authenticity check.

Keep export and runtime environments separate. Maintain a small tested set of platform constraints rather than promising every version combination allowed by broad dependency ranges.

Done when a tampered or mismatched cached checkpoint fails before model/pickle loading, explicit acquisition remains opt-in, and a clean supported recipe reproduces an artifact and smoke result. Record approval provenance; do not imply publisher signatures exist where they do not.

Dependencies: M0. These two workstreams can run in parallel.

## M2 Add provenance-aware recording and layered replay

Current issue: the video runner derives detector labels/confidence from the rules, filters detections, tracks them, and records only tracked observations. Lowering a threshold or changing a tracker later may require evidence that was discarded.

Work:

- Separate detector, tracker and rule orchestration behind small interfaces.
- Preserve default tracked-observation recording; add opt-in candidate-detection recording before tracking.
- Name the capture stage precisely. Post-NMS detections are not raw network tensors. Record capture floor, retained classes, NMS/top-K limits, sampling, timestamps, image dimensions, transforms and hashes.
- Define rule replay from tracks, tracker replay from candidate detections, and inference rerun from video.
- Cache each stage using its input hash, implementation/version and effective parameters. Keep successful completion and integrity checks separate from an existing output folder.
- Publish machine-readable schemas and a capability validator. Reject requests that require absent classes, lower capture thresholds, unsampled frames, pixels or embeddings.

Done when:

- Capture at confidence 0.3 rejects replay requesting unavailable 0.1 evidence rather than returning a plausible-looking zero.
- Eligible box-based tracker/rule replays make zero ONNX calls, verified by a call counter.
- Unchanged saved replay reproduces the expected event/count/trace bytes for the declared implementation and identity version.
- Tracker changes reuse detector records and recompute only tracking/rules.
- Hash changes invalidate affected cache entries; partial/corrupt recordings cannot masquerade as completed ones.
- RAM stays bounded with recording duration. Larger recordings and storage cost are explicit choices.

Do not assume byte-identical model inference across CPU/GPU. Appearance tracking requires compatible image/embedding evidence, not just cached boxes.

Dependencies: M0 contracts; M1 evaluator used for regression comparisons.

## M3 Add diagnostics and camera-specific setup

### Why-not inspection

Extend explanations to candidates that never start an episode. Scope inspection by source, rule, track/candidate and time range. Add reason codes for unavailable capability, below-threshold score, outside-zone anchor, tentative track, missing coverage, insufficient supported duration, reset and already-emitted episode.

Use bounded counters and optional sampled detail. Keep default processing and normal logs small.

Done when every reason has a minimal fixture, inspection does not change events, the exact recording/rule revision is identified, and limits prevent rules multiplied by detections multiplied by duration from becoming unbounded retained traces. A missing recorded observation must not be presented as proof of physical absence or an explanation of an unseen detector miss.

### Scene binding and setup wizard

State is already separated by source, but every source sent to one engine currently uses its shared zones/rules. Add explicit source-to-scene routing and geometry revisions at the pipeline layer. Preserve intentional shared-scene behavior for legacy callers.

Record reference dimensions, rotation, crop and coordinate transforms. Build a local guided setup tool that draws zones, validates settings, checks actual cadence and exports configuration. Orchestrate existing validation/doctor/smoke functions rather than inventing a cloud setup service.

Done when:

- Two cameras can use the same zone name with different polygons and no rule leakage.
- Unknown source binding fails clearly.
- Resizing preserves geometry; crop/rotation changes require a valid transform or revalidation.
- Geometry revisions reset affected state explicitly.
- A developer can configure and validate a scene without hand-editing coordinates.
- Image distances cannot silently become metres. Physical calibration remains an explicit optional capability.

Dependencies: M2 schemas/provenance. Diagnostics and routing can proceed independently; the wizard follows both.

### Custom perception and capability inspection

Define a small trusted Python adapter contract for candidate detections and declare supported classes, coordinate/preprocessing conventions and optional outputs. Keep existing presets as reference adapters. Add one configurable user-supplied box-model path, rather than requiring every user model to impersonate a fixed COCO preset.

Done when a model with custom class names can use count/dwell through the standard pipeline, missing rule capabilities fail before processing, golden inputs verify class maps/coordinates, and multiple rules share one inference pass. Keep adapter dependencies optional and user-supplied Python explicitly trusted; configuration must not execute arbitrary generated expressions. Pose/mask/action support is advertised only when consuming rules and qualified adapters exist.

## M4 Implement safe configuration and identity changes

Keep the full configuration fingerprint for provenance. Introduce a separately versioned rule-semantic fingerprint containing the rule and its referenced geometry, units, thresholds and coverage/gap policy. Define which model/tracker/source changes require a fresh epoch.

Build offline configuration diff, supported what-if replay and rollback before live hot reload. Version any new identity scheme and preserve explicit historical replay/migration paths.

Done when:

- Equivalent formatting/defaults remain stable.
- An unrelated rule edit does not accidentally rename an unchanged semantic episode under the new scheme.
- Changed geometry or temporal meaning cannot inherit incompatible dwell state.
- Old/new identity versions cannot silently share a deduplication namespace.
- Configuration comparisons reject unsupported replay evidence.
- Every future live cutover records its time/reason and resets affected state deliberately.

Resource changes can alter coverage even when rule text is unchanged. Record that provenance and require an epoch/reset where the evidence contract changes.

Dependencies: M0 identity decision, M2 replay and M3 scene revisions. No live reload until cutover tests pass.

## M5 Improve tracking and add measured behavior operators

Compare one established permissively licensed motion-aware tracker with the current IoU baseline. Keep observed, tentative and predicted positions distinguishable. Add named time-based entry/exit hysteresis policies while retaining strict sampled dwell.

Then add operators individually: zone entry/exit episodes, directional crossing, occupancy thresholds sustained for a duration, and restricted zone sequences. Define boundary tolerance, direction, re-entry, deduplication, unknown coverage and episode lifecycle for each.

Done when:

- Held-out event budgets are met for the declared use case and configuration.
- Tracking metrics and identity fragmentation are reported alongside count/dwell error.
- Jitter, true brief entries, reversals, ID reuse, occlusion, variable sampling and epochs have declared outcomes.
- Predicted positions never silently become confirmed visual evidence.
- Each operator has a runnable configuration, Python example, failure cases and measured confirmation-delay tradeoffs.

Do not change defaults solely because a tracker benchmark improves. Do not call tracked episodes unique physical visitors.

Dependencies: M1 evaluation and M2-M4 contracts. Change one factor at a time.

## M6 Qualify hardware and optimize cost

Qualification is an ongoing workstream, beginning with the existing FP32 CPU reference and CUDA evidence. Qualify each selected learned artifact on AMD CPU, M1 CPU and optional M1 CoreML separately. TensorRT remains experimental until independently tested.

For each model/provider recipe, require clean installation, approved hashes, golden preprocessing/output checks, positive and negative inference, recorded-video count/dwell, exact saved replay and provider profiling. Record partial fallback and hardware/OS/runtime versions. Establish tolerances per model/precision rather than treating one 0.005 single-image check as universal.

Only then compare resolution, sampling, ROI, FP16/INT8 and batching profiles against full pipeline costs. Treat the current float32 tensor interfaces deliberately when adding alternate export contracts. A budget planner may select only a measured profile that meets the user's predeclared quality, delay and memory constraints.

Done when each advertised recipe links to actual learned-model and event evidence. Measure warm-up, decode/conversion/transfers, inference, tracking, rules and serialization. Extend Python allocation checks to sustained decoded-video RSS, VRAM, native threads, handles and reload cycles, recording duration and workload. Do not infer camera capacity by multiplying model FPS.

Dependencies: M1 evaluator and approved artifacts; M2 provenance. Model-specific optimization waits for qualification and adequate representative calibration data.

## M7 Add retention and offline archive jobs

### Owned artifact storage

Add metadata retention, trace quotas, optional redaction and a manifest of owned artifacts. Explain when deleted evidence makes replay unavailable. Keep input videos, models and unrelated files outside cleanup ownership.

Done when cleanup cannot escape its store or follow symlinks into unrelated data, preserves active inputs, obeys quotas and reports disk-full/permission failures honestly. Redaction of exported images is distinct from ignoring an inference region.

### Batch processing

Add file/job manifests, progress, cancellation, per-file failure isolation and reuse of completed perception stages. Start with safe skip/retry of completed files, then implement resumable checkpoints only for adapters with an explicit state contract.

Done when restart checks input/model/config fingerprints, processing position, epoch, tracker state and rule state. Resume must not silently reset dwell or duplicate completed episodes. Unsupported resume should fail clearly or reuse a documented earlier stage.

Dependencies: M2 manifests/cache. Full resume also needs M4 identity semantics and supported tracker serialization. Retention safety work can run independently once ownership is specified.

## M8 Introduce live sources and application policies

Wrap a mature ingest backend with source health, bounded timeout/retry, clock quality and explicit epochs. Separate latest-frame from lossless policies; report queue age, drops and stale evidence. Isolate decoder failures from other sources.

Define episode versus message identity, lifecycle, retry and persistence semantics before adding optional sinks and an outbox. Prefer explicit at-least-once delivery plus application deduplication to an unproven exactly-once claim.

Keep behavior facts separate from notification schedules, cooldowns and grouping. Optional external facts such as access schedules or transaction records require timestamp/expiry/unknown policies. Relative video PTS alone does not establish a wall-clock business hour.

Done when disconnect, overload, stale-frame, consumer failure and process-restart tests preserve declared semantics; one stalled source cannot block healthy sources; measured evidence-age/resource budgets are met; suppression never deletes the underlying event record.

Dependencies: M1 quality baseline, M2 clocks/provenance, M4 identity and M7 owned storage. Deliver one bounded ingest adapter and one sink first.

## Later work that requires pilot evidence

Additional framework adapters, pose/segmentation, queue estimates, unattended-object relationships, camera drift/tamper, compact local action models and additional business integrations are separate scoped proposals. Each requires capability declarations, normal lookalikes, artifact terms and target-hardware measurements.

Do not promise general theft or intent recognition. Natural-language setup can eventually compile to the same restricted plans; it must not execute generated code or become a per-frame service dependency.

## First implementation sequence

1. Baseline fixtures, recording/identity specification and the pilot requirements register.
2. Approved checkpoint verification before loading, with tamper/cached-file tests.
3. Event evaluator with hand-checkable count/interval fixtures; begin held-out data labeling.
4. Candidate-detection recording and manifest/capability validation.
5. Rule/tracker replay split, cache invalidation and zero-inference-call tests.
6. Bounded why-not inspection and scene binding; then guided setup.
7. Configuration diff and explicit identity migration.
8. One tracker/hysteresis comparison, followed by one measured new operator.

Suggested first release scope is items 1-5. Do not bundle live RTSP, a dashboard, a new tracker and multiple action models into that release. Hardware qualification and requirements/data collection run alongside this sequence.

## Coverage of the identified gaps

- Event-quality evaluation: M1, then the comparison gates in M5/M6.
- Stronger tracking and zone hysteresis: M5.
- Directional counting, occupancy and sequences: M5 after the evaluator.
- Live camera reliability and delivery: M8.
- Custom models and capability validation: M2 contracts plus M3 adapters.
- Layered replay and reduced repeated GPU work: M2.
- Why-not diagnostics and guided scene setup: M3.
- Precise unknown/unsupported states: M0 contracts, M2 capability checks and M3 explanations.
- Safe configuration changes, stable semantic identities and migration: M4.
- Cost budgets and qualified precision profiles: M6.
- Metadata retention, redaction and artifact ownership: M7.
- Artifact trust and reproducible installation: M1, maintained through release gates.
- Archive jobs, cancellation and resumable work: M7.
- External context, schedules and notification policies: M8, only with explicit clock/expiry semantics.

## Release checklist

- Link each advertised capability to its contract, evidence and supported platform recipe.
- Pass clean wheel/source installation, core and relevant optional-dependency tests.
- Execute documentation examples and preserve golden replay/migration fixtures.
- Complete relevant learned-model runs and provider profiles; report fixture-only evidence separately.
- Score held-out event quality for behavioral changes, with frozen thresholds and denominators.
- Exercise memory, storage and failure bounds appropriate to the changed runtime path.
- Publish known failures, migration notes, artifact hashes, exact revision and dependency recipe.
- Generate GitHub release assets and website documentation from the same reviewed source tag.

## Schedule and ownership

Use one accountable owner per milestone, with independent review of temporal/identity contracts and evaluation scoring. Parallelize artifact trust, evaluator/data preparation and hardware qualification after M0. Serialize schema-dependent implementation and any experiment that changes event meaning.

Estimate release dates after M0/M1 establish data access, hardware availability, scope and measured engineering effort. Dataset rights/annotation, AMD/M1 access, upstream runtime compatibility and pilot accuracy targets can block qualification even when code is complete. Mark that status explicitly rather than replacing evidence with a deadline.
