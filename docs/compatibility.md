# Compatibility and test evidence

Version **0.4.0a1** adds experimental layered replay, evaluation, temporal operators, setup, archive/retention and live delivery. Read [current implementation status](implementation-status.md) and the [system guide](system-guide.md). There are no ongoing LLM calls, cloud inference services or runtime model downloads. This remains feasibility software.

The historical v0.3 platform baseline below is retained as version-specific evidence. Current validation reports are linked from implementation status. A previously passed platform suite does not automatically validate new APIs or every learned model.

## What was actually tested

- **Intel Windows CPU: verified.** Core Ultra 7 270K Plus, Python 3.12, CPU-only ONNX Runtime 1.31.1. Full v0.3 suite: 80 passed, one Apple hardware test skipped. Each shortlisted learned model ran positive counting/dwell regression, exact event/count/trace replay and a separately profiled CPU latency benchmark.
- **NVIDIA Windows: verified.** RTX 5060 Ti. The CUDA environment also passed 80 tests with one Apple hardware test skipped. All three shortlisted learned models passed positive counting/dwell regression and exact replay; CUDA kernel events were observed in separate benchmark profiles.
- **AMD x86-64 CPU: fixture integration verified in CI; learned-model qualification pending.** The full generated-fixture suite passed on AMD EPYC Windows and Linux GitHub runners. All three actual learned-model regressions and capacity measurements on AMD remain pending.
- **M1 / Apple Silicon: fixture integration verified on a GitHub M1 virtual runner; learned-model qualification pending.** The arm64 macOS suite and tiny CoreML delegation test passed. This does not qualify the three learned model graphs. CPU remains the baseline; CoreML is experimental and opt-in.
- **TensorRT: experimental, unverified.** CUDA success does not establish TensorRT compatibility.

The [current v0.3 reliability report](../benchmarks/reliability-v03.json) links to exact suite results, positive regression reports, profiling and state checks. The [v0.2 validation report](../benchmarks/compatibility-validation.json) remains historical evidence. Python's Windows architecture name `AMD64` is not a CPU-vendor claim.

## Correctness checks and fixes

The expanded suite covers an independent randomized counting/dwell oracle; source/epoch resets and missing coverage; polygon and configuration validation; tracker failure rollback; model contracts; artifact hashes; input/output path collisions; partial-output failure status; provider fallback reporting; and real CPU ONNX inference through decoded video, tracking and rules.

The audit exposed malformed-input exceptions, mutable scene geometry, incomplete direct-constructor validation, tracker state changes after rejected updates, stale success summaries, input truncation risk from overlapping output paths, and full-file observation hashing. These cases now have regression tests and fixes.

The v0.3 audit fixes a configuration file race between perception and rule replay, configuration-independent event IDs, avoidable greedy-association fragmentation, retained-track capacity failures and unreleased source slots. New tests compare small matching problems against exhaustive assignment, verify events are written before end-of-file, inspect precise reset reasons, reject malformed runtime tensors and compare saved video results with exact replay. These checks do not establish general camera accuracy.

A separate 20,000-batch test continually replaces ten track IDs and checks bounded retained Python state. The current report is [soak-v03.json](../benchmarks/soak-v03.json). This is Python allocation evidence, not a multi-hour decoded-video/RSS/VRAM measurement. The historical v0.2 soak recorded 5,088 bytes of retained traced allocation growth after warm-up.

## Positive learned-model regression

Each of RTMDet-Tiny, D-FINE-N and RF-DETR-Nano processed a 5.1-second controlled clip: one public-domain NASA still-person image repeated for 4.2 seconds, then blank frames. At 5 Hz, each produced 26 observations, maximum visible count one, final count zero and exactly one 3-second dwell event confirmed at 3.2 seconds after tracker confirmation. CPU/CUDA static-image boxes and confidence agreed within the configured 0.005 absolute tolerance.

The fixture source and image SHA256 are in the reports. [scikit-image documents the NASA image's public-domain status](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut). This is a positive integration regression. It does not test actual motion, occlusion, crowds, lighting variation or surveillance event accuracy. A tiny signal-driven ONNX graph in the automated suite separately verifies known count transitions; it is explicitly not a trained detector.

## Run on an AMD or M1 host

From a fresh extracted checkout, install `.[vision,cpu,dev]` on AMD or `.[vision,apple,dev]` on native Apple Silicon. Then:

```sh
# AMD host:
python scripts/validate_platform.py --require-vendor AMD --output runs/amd-validation
# M1 host:
python scripts/validate_platform.py --require-vendor Apple --require-architecture arm64 --output runs/m1-validation
```

The validator refuses a mismatched vendor/architecture and records the actual host and JUnit results. On a native Mac with CoreML available, the conditional hardware test requires a real CoreML profile event for the tiny test graph. That test alone does not establish compatibility of all three learned model graphs.

After preparing the real model artifacts, run:

```sh
behaviorkit benchmark models --provider cpu --threads 4 --models-dir models --output runs/cpu-benchmark.json
python scripts/validate_single_person.py --help
```

The second script accepts an explicit local rights-cleared image; it never downloads one automatically. Run with `--provider coreml` on the M1 only after the CPU baseline passes. Record all model failures rather than treating availability as acceleration proof.

The initial public GitHub Actions run passed all six jobs, including Windows/Linux AMD and arm64 macOS fixture integration, Python 3.10/3.12 core tests and package build/install. [Recorded CI evidence](../benchmarks/github-actions-v03.json) identifies the exact source commit, runner hardware and test counts. The tiny CoreML graph delegated successfully on the M1 virtual runner. These checks do not qualify all three learned models. Hosted runner labels are documented by [GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners); actual vendor/architecture is recorded on execution.

## Remaining feasibility gates

Complete the real learned-model regressions and deployment measurements on AMD and M1 hardware. Then measure counts and dwell precision/recall on labeled fixed-camera clips with motion, occlusion, entries/exits and source gaps. Finally run sustained decoded-video memory, power and latency tests on the deployment hardware before adding more behaviours.
