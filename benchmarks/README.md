# Recorded evidence

Current v0.3 entry point: `reliability-v03.json`. It links to `positive-cpu-v03.json`, `positive-cuda-v03.json`, `cpu-v03-benchmark.json`, `cuda-v03-benchmark.json`, `engine-v03-benchmark.json`, `soak-v03.json`, and the two v0.3 JUnit suites under `validation/`. `github-documentation-checks.json` records README examples and local Markdown link checks.

The source hashes identify the checked runtime code. Rebuilding package metadata or README content changes the wheel hash without changing those runtime hashes. Local recorded results and current GitHub Actions status are distinct evidence.

Version 0.2 reports: `compatibility-validation.json`, `cpu-model-benchmark.json`, `cuda-v02-benchmark.json`, `positive-cpu-validation.json`, `positive-cuda-validation.json`, `engine-v02-benchmark.json`, `soak-v02.json`, and `apple-arm64-dependencies.json`. JUnit suites and actual Intel host metadata are under `validation/`.

Positive regression uses a repeated still-person image and blank frames; it is not a motion/occlusion accuracy dataset. Model benchmarks use seeded synthetic RGB input and a separate execution-provider profile; timings exclude decode/tracking/rules and are from a shared desktop. Raw profiler files are local scratch and excluded from the downloadable source ZIP.

Historical `model-benchmark.json`, `engine-benchmark.json`, `doctor.json` and `video-smoke/` came from v0.1. Physical AMD, M1/CoreML and TensorRT learned-model execution remain pending. A successful dependency resolution or configured CI matrix is not hardware validation. See `docs/compatibility.md` for reproduction and limits.
