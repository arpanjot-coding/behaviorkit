# Command reference

All commands below are implemented. Every command also supports `--help`; the installed entry point is `behaviorkit`. You can use `python -m behaviorkit` if the executable is not on PATH.

## demo

```sh
behaviorkit demo --output runs/demo
```

Creates synthetic observations, a scene config, count samples, one dwell event and a summary. No vision dependencies or network access are required.

## validate

```sh
behaviorkit validate --config configs/people.json
```

Checks version, required fields, simple nonzero-area polygons, rule kinds, labels, finite numeric ranges, unique rule IDs and resource limits. Only `count` and `dwell` are accepted. It does not inspect a model or certify camera visibility.

## replay

```sh
behaviorkit replay --observations runs/video/observations.jsonl --config runs/video/config.snapshot.json --trace --output runs/replay
```

Evaluates one JSON observation per line. Timestamps must strictly increase within each source epoch. Each line is limited to 8 MiB. A malformed batch stops evaluation with its line number; outputs written before that error can be partial. Configuration is limited to 1 MiB. Observation hashing is streamed.

Check `summary.json`: `status` is `running`, `ok`, or `error`. Failed runs replace stale successful summaries and mark outputs partial. Input paths cannot overlap output files. Use a fresh output directory per experiment.

## run

```sh
behaviorkit run --video sample.mp4 --model d-fine-n --models-dir models --config configs/people.json --provider cuda --sample-fps 5 --source entrance --max-frames 300 --output runs/video
```

- `--model`: `rtmdet-tiny`, `d-fine-n`, or `rf-detr-nano`.
- `--provider`: `auto` (default), `cpu`, `cuda`, optional `tensorrt`, or experimental Apple Silicon `coreml`.
- `--threads`: ONNX intra-op CPU threads, 0..1024, default 0. Zero uses runtime defaults for CPU and one thread for GPU/CoreML paths. Tune against your workload.
- `--sample-fps`: requested sampling rate in (0,120], default 5.
- `--max-frames`: maximum sampled frames, default 0 for the complete file.
- `--source`: source name in event and count records.
- `--epoch`: optional caller-managed source epoch. The default is a unique run ID, preventing independent files from sharing a default episode namespace.
- `--trace`: record dwell starts, confirmations and reset reasons in `transitions.jsonl`. Also available for replay; disabled by default.
- `--output`: output folder; files with the same names are overwritten.

`auto` prefers advertised CUDA, otherwise CPU; it records any CUDA initialization fallback. CoreML is opt-in. Explicit provider initialization failure is an error. Summaries record requested and selected providers.

The decoder requires presentation timestamps. A dwell rule's maximum gap must be at least the requested sample period. Actual source gaps can still reset the timer. This legacy local-file command does not ingest RTSP; the separate experimental `live` command has explicit source health/retry contracts.

Rules are evaluated as sampled frames are processed, and count/event records are flushed during the run. The runner reads the configuration once, preserves it in `config.snapshot.json`, and uses that in-memory scene throughout the job. Editing the original configuration does not change an active run. Summaries include the original config hash, effective config fingerprint, saved snapshot hash, video hash, source epoch and observation-stream hash.

`diagnostics.jsonl` records engine and tracker messages. A retained unmatched track may be evicted to admit a currently observed detection; its retirement is reported rather than silently exceeding the state limit. More current detections than configured capacity still fail explicitly.

Per-frame stage timings include decode/sampling, RGB conversion, detection, tracking, rules and serialization. `sample_pipeline_ms` measures processing work between selected frames; it is not live-camera evidence age. Check the final summary before consuming a completed run: failures can leave partial output.

## doctor and models

```sh
behaviorkit doctor --provider cuda
behaviorkit models
```

Inspect CPU vendor/architecture and device/provider availability and the supported model contracts. Availability does not verify inference. `models` does not download weights.

## benchmark

```sh
behaviorkit benchmark engine --tracks 100 --batches 5000 --output runs/engine.json
behaviorkit benchmark models --models rtmdet-tiny d-fine-n rf-detr-nano --models-dir models --provider cuda --iterations 50 --warmup 10 --output runs/models.json
```

The engine benchmark excludes perception. The model benchmark uses seeded synthetic RGB input and does not estimate accuracy. Errors are recorded per model; any failed model returns a nonzero exit code.

## Exit codes

- `0`: command completed; selected benchmark models passed execution checks.
- `1`: failed model benchmark, partial/failed live or batch run, or canceled archive job.
- `2`: invalid configuration, missing artifacts/dependencies, input or runtime error.

## Versioned system commands (0.4 alpha)

The [system guide](system-guide.md) contains runnable examples and the evidence/identity contracts for `capture`, `recording-info`, `track`, `apply`, `evaluate`, `inspect`, `config-diff`, `compare`, `setup`, `batch`, `store`, `live`, `outbox` and `plan-budget`.

Use `behaviorkit COMMAND --help` for all arguments. New stage directories and report outputs must not already exist. Legacy `run`/`replay` keep their existing output behavior. Legacy configurations remain schema 0.1; opt into schema 0.2 for new behavior operators. A complete recording manifest means artifact integrity/completion; check `run_status` and per-source health for live success.
