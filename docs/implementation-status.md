# Implementation and qualification status

**0.4.0a1 is an experimental implementation across M0-M8.** This status separates available code from external acceptance gates. It does not declare every milestone qualified for production. The [system guide](system-guide.md) specifies the commands and API contracts; the [original plan](implementation-plan.md) retains the complete acceptance criteria.

## Available software

**M0: contracts and baseline.** Packaged frame/recording/annotation JSON schemas, native semantic validation, explicit coverage/clocks/coordinates, capability declarations and a v0.3 wheel-generated golden fixture. The [pilot register](pilot-requirements.json) keeps workflow conditions and unset budgets visible.

**M1: evaluation and artifact checks.** One-to-one event matching, count and sampled identity diagnostics, false-alert/delay/duplicate/coverage denominators, split-group leakage checks, approved observed checkpoint digests and verification before checkpoint imports/loaders. Cached tamper tests and pinned D-FINE/RF-DETR exporter checks fail closed. Digests are locally approved observed bytes, not publisher signatures or a general checkpoint license guarantee.

**M2: layered evidence.** Immutable candidate, track and rule stages; implementation/input/parameter hashes; bounded JSONL readers; complete/error manifests and integrity verification. Capture/association floors remain separate. Tracker and rule replay make zero model calls. Missing classes, lower discarded scores, unsupported cadence/pixels/embeddings and incomplete/corrupt evidence fail explicitly.

**M3: setup and inspection.** Bounded why-not counters/detail, candidate-only and tracked inspection, per-source scene/geometry routing, explicit crop/rotation/reference metadata, local polygon editor/validation/download, trusted adapter protocols and a configurable non-COCO ONNX box adapter. The wizard starts one rule; sequence/line composition uses config/Python. Rich pose/mask/action adapters are not implemented.

**M4: configuration and identity.** Versioned semantic identity v3 in opt-in scene 0.2; full provenance remains separate. Offline diff/compare, review-only episode migration links, immutable replay, rollback by reapplying an older scene and explicit reset/audit cutovers. Legacy scene 0.1 keeps v2 IDs. There is no silent hot reload.

**M5: tracking and operators.** Optional pinned Apache-2.0 ByteTrack adapter, strict and observed debounced membership, entry/exit, finite directional crossing, sustained occupancy and restricted zone sequences. Predictions are excluded from observed evidence. Each operator has deterministic fixtures; [the complete synthetic example](../examples/system_demo.py) emits all six event kinds and visible counts. Held-out event improvements over IoU are not yet established.

**M6: measurement and optimization gates.** Learned-model pipeline validation and separate provider probes, stage timing, bounded native-process/resource logs, explicit measured-profile selection and unqualified FP16/calibrated INT8 export candidates. CPU and CUDA short controlled cases pass on the recorded host. Precision conversion tests establish valid exported graphs/float I/O on generated models, not learned-model accuracy or a speedup.

**M7: owned storage and archive jobs.** Owned copies with verified cleanup, quotas/pins/protected inputs, explicit unavailable evidence, export-only redaction, per-file job manifests, cancellation/failure isolation, verified stage reuse and retries. Partial files restart at frame zero or a completed perception stage. Unsupported mid-file tracker/rule restoration is not attempted.

**M8: experimental ingest and delivery.** PyAV decoder process isolation, bounded latest/backpressure queues, open/read timeout and retry limits, watchdog termination, fresh epochs, fair polling and stale checks before/after serial inference. SQLite facts and leased at-least-once messages survive restart; JSONL is the initial sink. Notification schedules/cooldowns/expiring boolean context remain separate from raw behavior facts. Relative PTS never invents UTC context.

## Recorded checks

The [reliability report](../benchmarks/reliability-v04.json) links exact results and evidence scopes. Local Intel Windows suite: **124 passed, one CoreML-only skip**, including optional ByteTrack, resource monitoring, precision-export graph checks and schema validation. The prior baseline was 80 passed/one skip. New package/CI results are recorded separately when completed.

The three learned models each pass the new candidate/track/rule path on Intel CPU and NVIDIA CUDA. Each short case has 16 sampled frames, maximum count one, an empty-scene count and one observed one-second dwell confirmation at 1.2 seconds. Saved replay is exact and makes zero further inference calls. Separate two-frame profiles show requested provider execution. These repeated-still positive/negative tests do not cover moving-camera surveillance accuracy.

The longer CPU D-FINE check processes 1,001 samples from 200.1 seconds of source video in about 36 seconds of shared-host wall time. It produces the expected 65 dwell episodes and exact replay. After warm-up, RSS is approximately 213-218 MB in the recorded short run. Per-process VRAM is unavailable on this host's driver path. This is not a multi-hour soak or a camera-capacity guarantee.

Fault checks cover downstream batch write failure/reuse, sink failure/retry, expired/exclusive leases, process restart, corrupt inputs/stages, unknown and predicted evidence, real local HTTP EOF/reconnect/timeouts, queue drops and a deliberately stalled native decoder alongside a healthy source. They do not establish a particular real RTSP camera's availability.

Browser QA exercises local image selection, polygon drawing, server validation, scene/geometry downloads and mobile overflow. The local wizard works with `--port 0` when a fixed port is reserved by the host OS.

## Known failures retained

A heavily compressed version of the controlled image caused an extra RTMDet person detection. That false positive remains visible in the [failure record](../benchmarks/known-failures-v04.json). It is not removed by a library bug fix.

Low-score candidates initially changed IoU association and fragmented D-FINE's true-person identity. Replaying the same candidates with only the independent association floor raised from 0.1 to 0.3 restores its dwell episode without new inference. A regression test now protects that separation. Capture at a lower floor remains available for explicit tracker experiments; it is not a free accuracy improvement.

An early resource experiment left ORT profiling enabled for every frame, retaining a large native event history. Normal capture already disables it. The validation harness now uses a separate short provider probe, and the resource report distinguishes that from steady capture. Do not use continuous profiling as ordinary production telemetry.

## External gates still open

- Representative rights-cleared camera footage, human event labels, independent camera/day splits and agreed quality/resource budgets. Null pilot targets are not passes.
- All three learned artifacts on physical AMD CPU and M1 CPU/CoreML, with model-specific event and execution evidence. Earlier generated-fixture CI is retained separately.
- Held-out event-quality comparisons for motion tracking, debouncing, new operators, resolution/sampling choices and calibrated learned-model FP16/INT8 exports.
- Multi-hour real-camera/decode/inference/storage/reload tests, latency/freshness budgets, disk-full operations and deployment-specific recovery policies. Decoder isolation cannot interrupt a synchronous native inference hang.
- Appearance re-identification, camera drift detection, physical calibration, rich pose/action models and general theft/intent recognition. These are later scoped features, not hidden capabilities of this alpha.

No PyPI publication has occurred. Use the verified source or wheel for this revision and keep experimental data separate from deployment claims.
