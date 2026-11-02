# Benchmarks you can reproduce

The website reads actual JSON artifacts, including their version, environment and failure status. Shared-host timings are observations, not a default-model recommendation or idle-device ranking.

## CPU and NVIDIA measurements

```sh
behaviorkit benchmark models --models rtmdet-tiny d-fine-n rf-detr-nano --provider cpu --threads 4 --models-dir models --iterations 50 --warmup 10 --output runs/cpu.json
behaviorkit benchmark models --models rtmdet-tiny d-fine-n rf-detr-nano --provider cuda --threads 4 --models-dir models --iterations 50 --warmup 10 --output runs/cuda.json
```

Bundled v0.3 measurements use 50 timed iterations and ten warm-ups on the recorded Intel/RTX host, four ONNX intra-op CPU threads, and one OpenCV thread. The input is the same seeded 1280x720 synthetic RGB image, transformed using each model's native contract. Session creation is reported separately.

Timed sessions have profiling disabled. Preprocessing, synchronous `session.run` including CPU input/output transfers, and postprocessing are timed. A separate profiled session verifies kernel events for the selected provider. Some CUDA graph operations may still execute on CPU. CoreML events establish delegation, not which physical Apple compute unit was used.

**Latency is not accuracy.** These measurements exclude video decoding, tracking, rules, multi-camera scheduling and notification delivery. They do not verify TensorRT, AMD hardware or M1 execution. Historical v0.1 JSON remains in the source archive; the tables here use v0.3 artifacts.

## Engine overhead and state lifetime

```sh
behaviorkit benchmark engine --tracks 100 --batches 5000 --output runs/engine.json
python scripts/soak_replay.py --batches 20000 --tracks 10 --output runs/soak.json
```

The engine benchmark uses 20 rules and excludes inference, tracking, input construction and serialization. The soak test replaces all IDs each batch, checking active-state bounds and retained Python allocations. It is not a multi-hour RSS/VRAM test.

## Correctness evidence

See [compatibility and tests](compatibility.md) for the 80-pass suite, independent randomized event oracle, positive video regressions on all three learned models, and exact hardware verification limits. Those controlled fixtures establish integration, not surveillance accuracy.

## Next pilot accuracy protocol

Collect rights-cleared, camera-specific video and annotate visible counts plus dwell episodes. Separate cameras/days across tuning and held-out sets. Record confidence thresholds, sampling cadence, source gaps and tracker parameters.

- Counts: mean absolute error, exact-count rate and unknown coverage fraction.
- Dwell: event precision/recall, false alerts per camera-hour, duplicate rate and threshold-confirmation delay.
- Tracking: ID switches and fragmentation around occlusions.
- Deployment: sustained throughput, p95 pipeline delay, RSS, VRAM and measured whole-system power.

No labeled-camera accuracy or multi-hour deployment claim is made by this prototype.
