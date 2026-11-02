# Roadmap and developer requirements

This is a proposed sequence, not a delivery-date commitment. v0.3 supports counting, strict sampled dwell, replay and diagnostics. Features below are planned unless the README explicitly marks them current.

**0.4 alpha update:** the [implementation status](docs/implementation-status.md) records experimental APIs now implemented across M0-M8. The acceptance criteria below still apply: software implementation does not replace held-out event data, hardware qualification or sustained deployment evidence.

The [implementation plan](docs/implementation-plan.md) expands this roadmap into dependency-ordered milestones, concrete first changes, migration requirements and release gates.

## How requirements become supported use cases

Every accepted request should describe:

1. The observable event, its start/end conditions and normal activities that resemble it.
2. Camera/file type, resolution, sampling, viewpoint, lighting and expected occlusion.
3. Available models, classes, tracking, calibration and missing external context.
4. Hardware, OS, installation constraints and offline requirements.
5. Acceptable precision/recall, false alerts per camera-hour, confirmation delay and compute/memory budget.
6. Reproducible configuration and representative rights-cleared labels, including failures.
7. How an application consumes, retries, stores and reviews events.

Start a pilot with several independent developers and camera setups. Record recurring integration work and prioritize requirements supported by examples. Expand the pilot until important differences in tasks and constraints are covered; a fixed participant count does not prove completeness. Public issues identify friction, not market prevalence or willingness to pay.

Maintain a trace: request -> event contract -> capability check -> regression fixture -> hardware recipe -> documentation example -> release evidence. A missing link means the use case is provisional.

## Priority 1: event-quality evaluation

Build an offline runner for labeled counts and event intervals. Integrate established temporal/tracking evaluation tools where suitable.

Acceptance criteria:

- Define matching, temporal tolerance, duplicates, ambiguous coverage and count scoring before tuning.
- Separate tuning and evaluation by camera/day or another leakage-resistant boundary.
- Report precision/recall, false alerts per camera-hour, delay, duplicates, count error and uncertain coverage with sample size.
- Link false/missed events to observations, configuration and evidence intervals.
- Compare changes on the same held-out corpus and preserve known failures.

This is required before claiming a detector or tracker change improves monitoring.

## Priority 2: tracking and zone robustness

Add a stable adapter for an established motion-aware tracker. Distinguish observed, tentative and predicted evidence. Compare against the current IoU baseline.

Add time-based entry/exit hysteresis. Introduce cumulative or tolerated-gap dwell only for demonstrated needs, while preserving strict observed dwell as a named option.

Acceptance criteria:

- Measure fragmentation/tracking quality alongside downstream event error.
- Predicted-only positions cannot silently become confirmed visual observations.
- Declare outcomes for jitter, occlusion, reused IDs, source resets and variable sampling.
- Measure recall/delay tradeoffs from smoothing or gap tolerance.
- Keep histories bounded and explain episode resets.

## Priority 3: deterministic behavior features

Start with directional crossing, zone entry/exit episodes, occupancy thresholds sustained for a duration, and restricted zone sequences.

Examples: an observed track crosses toward an exit; more than N eligible tracks remain for T seconds; a track enters A then B within T seconds.

Acceptance criteria:

- Specify direction, tolerance, deduplication, unknown coverage and episode lifecycle.
- Test reversals, boundary jitter, brief entries and re-entry.
- Validate model/tracking capabilities before processing.
- Provide runnable configuration, a minimal Python example and measured event results.

These predicates do not establish unique physical visitors or intentions.

## Priority 4: live sources and reliable delivery

Wrap a mature ingest backend with source health, timeout/reconnect handling, bounded queues and explicit freshness budgets. Support latest-frame and lossless policies with different contracts.

Acceptance criteria:

- A stalled camera cannot block healthy sources.
- Queue growth is bounded under overload; dropped/stale evidence is observable.
- Disconnects and clock changes produce explicit epochs/resets.
- Measure per-camera evidence age, pipeline stages, RSS, VRAM, threads and handles during sustained runs.
- Define event starts/updates/ends, IDs, retries and idempotence before adding optional MQTT/webhook/SQLite sinks.
- Recovery tests establish what survives process restarts and output failures.

## Priority 5: setup and portability

Add capability inspection, custom model/tracker adapters and validated setup helpers. Keep export toolboxes separate from runtime recipes.

Acceptance criteria:

- Qualify clean environments with real artifacts and record actual hardware.
- Complete physical AMD/M1 learned-model checks; do not infer qualification from wheel availability.
- Verify class maps, preprocessing/output contracts and numerical tolerances on reference inputs.
- Publish target-specific recipes and explicit unsupported combinations.
- A new developer can complete the documented workflow and interpret all outputs.

## Later: relationships and local action models

Candidates include queue waiting episodes, stationary/unattended objects, pickup/carry/place sequences, pose-based actions and camera tamper/drift detection.

Each requires a narrow definition, representative labels, normal lookalikes, checkpoint/data terms and a measured compute budget. Association does not establish ownership; pickup/exit does not establish payment or authorization.

Use geometry and temporal state where sufficient. Add compact local learned adapters when evaluation shows value. Optional natural-language setup can compile into a restricted validated plan; it must not execute generated code or introduce ongoing LLM calls into the frame loop.

## Release gate for an advertised feature

- A developer need and observable contract are recorded.
- Unsupported capabilities fail clearly; missing evidence stays explicit.
- Relevant unit, integration, replay and held-out event tests pass.
- Hardware recipes and resource bounds are demonstrated.
- Docs, migration notes and failure examples match actual behavior.
- CI and physical learned-model evidence are reported separately.
- Claims stay within evaluated conditions.
