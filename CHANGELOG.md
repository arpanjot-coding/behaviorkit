# Changelog

## 0.3.0

- Freeze one validated configuration snapshot per video job and save it with hashes. Changes to the original file cannot change active detection or evaluation rules.
- Evaluate and flush counts/events during video processing, while recording observations for deterministic replay. Include decode/sampling, conversion, rules and serialization stage timings.
- Add effective configuration fingerprints and event identity version 2. Independent file jobs use fresh epochs; `--epoch` supports deliberate caller-managed ingestion. Saved observations preserve their epoch on replay.
- Add optional `--trace`, public `dwell_states()`, structured reset reasons and explicit `close_source()` cleanup. Tracing retains no engine history and does not change events.
- Replace greedy association with deterministic maximum-cardinality IoU matching. Retire oldest unmatched retained tracks at capacity with diagnostics; keep strict rejection available to Python callers.
- Validate model tensor metadata and runtime output dimensions, alignment, finite values and probability ranges before accepting detections.
- Add regression coverage for the configuration race, independent matching oracle, event namespaces, capacity, cleanup, explanations and exact real-video replay.

### Migration and limits

Event IDs differ from 0.2. Keep identity versions separate in downstream deduplication. Replaying 0.2 observations through 0.3 produces 0.3 identities. Strict dwell resets remain the default; motion prediction, re-identification, hysteresis, RTSP recovery and theft/action classification are still absent. Physical AMD/M1 qualification remains pending. Existing historical v0.2 evidence is retained with its version.
