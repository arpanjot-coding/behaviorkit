# Changelog

## 0.4.0a1 (experimental development)

- Add pre-load approved checkpoint verification, cached tamper rejection and pinned upstream export checks.
- Add versioned frame/recording contracts, integrity manifests, implementation-aware caches and candidate/track/rule replay. Preserve capture and association confidence floors separately.
- Add offline event/count and sampled tracking evaluation with explicit coverage, one-to-one matching and denominators.
- Add opt-in scene 0.2 with semantic event identity v3, observed debouncing, entry/exit, directional crossing, sustained occupancy and restricted zone sequences. Preserve legacy scene 0.1 / v2 golden output bytes.
- Add bounded why-not inspection, camera/geometry routing, explicit cutover audits, offline config comparisons and review-only migration links.
- Add configurable non-COCO box ONNX adapters, optional Apache-2.0 ByteTrack, and a loopback-only polygon setup wizard.
- Add immutable stage retries, archive job cancellation/reuse, owned artifact quotas/pins/redaction, and measured-profile budget selection.
- Add experimental isolated PyAV live decoders, latest/backpressure queues, stale-frame checks, source health and fresh segment epochs.
- Add durable SQLite facts and leased at-least-once messages with JSONL delivery, notification policy/context expiry, resource sampling and explicit unqualified FP16/INT8 export candidates.
- Add contract/fault/integration tests, clean package checks, a new learned-model validation script and detailed usage/qualification documentation.

### Qualification limits

Implemented software does not complete all pilot gates. Held-out camera-event accuracy, all-three-model AMD/M1 qualification, production camera uptime and calibrated precision variants remain pending. The known compressed-image false positive is retained in evidence. There is no general theft or intent classifier and no PyPI publication.

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
