# Recorded evidence

Version 0.2 reports: `compatibility-validation.json`, `cpu-model-benchmark.json`, `cuda-v02-benchmark.json`, `positive-cpu-validation.json`, `positive-cuda-validation.json`, `engine-v02-benchmark.json`, `soak-v02.json`, and `apple-arm64-dependencies.json`. JUnit suites and actual Intel host metadata are under `validation/`.

Positive regression uses a repeated still-person image and blank frames; it is not a motion/occlusion accuracy dataset. Model benchmarks use seeded synthetic RGB input and a separate execution-provider profile; timings exclude decode/tracking/rules and are from a shared desktop. Raw profiler files are local scratch and excluded from the downloadable source ZIP.

Historical `model-benchmark.json`, `engine-benchmark.json`, `doctor.json` and `video-smoke/` came from v0.1. Physical AMD, M1/CoreML and TensorRT learned-model execution remain pending. A successful dependency resolution or configured CI matrix is not hardware validation. See `docs/compatibility.md` for reproduction and limits.
