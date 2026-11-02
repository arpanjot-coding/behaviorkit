# BehaviorKit

[![Tests](https://github.com/arpanjot-coding/behaviorkit/actions/workflows/tests.yml/badge.svg)](https://github.com/arpanjot-coding/behaviorkit/actions/workflows/tests.yml)

**Local video observations, explainable behavior rules, reproducible events.**

BehaviorKit is a Python library for fixed-camera visible-object counting and observed dwell detection. Use the small rule engine with your own detector and tracker, or run the supplied ONNX video pipeline on CPU or NVIDIA CUDA.

The runtime makes **no LLM calls**, requires no API key, and performs no automatic model downloads. Models, tracking and rules run on your hardware. Importing the core loads only Python's standard library.

**Status: v0.3.0 feasibility prototype.** Counting and strict sampled dwell work today. This is not a production camera service or theft classifier. The distribution is `behaviorkit-prototype` and is **not published to PyPI**. BehaviorKit is a working title; the unrelated [IUB-RHouse/BehaviorKit](https://github.com/IUB-RHouse/BehaviorKit) robotics project is not affiliated with this repository.

[Quickstart](#quickstart) · [Hardware](#choose-a-vision-runtime) · [Python API](#python-api) · [Documentation](#documentation) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

## What works today

- Count observed eligible confirmed tracks inside normalized polygons.
- Confirm that one track has remained in a zone for a configured sampled dwell interval.
- Apply multiple rules to shared observations without repeating model inference for every rule.
- Save observations and frozen configuration, then reproduce count, event and optional trace records exactly.
- Inspect dwell starts, confirmations and reset reasons through CLI output and a public Python API.
- Run supported ONNX contracts on CPU or NVIDIA CUDA; verify execution using separate provider profiles.
- Consume observations from your own perception system without installing vision dependencies.

Examples include visible queue-area counts, waiting-zone dwell, and evaluation of rule changes on recorded video. A tracker ID is an input assumption, not a verified identity across people, clips or cameras.

## Quickstart

Python 3.12 is recommended for vision. The dependency-free core supports Python 3.10+; recorded execution evidence uses Python 3.12.

```sh
git clone https://github.com/arpanjot-coding/behaviorkit.git
cd behaviorkit
python -m venv .venv
```

Activate your environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# Linux or macOS
source .venv/bin/activate
```

Install the core and run an example without a model or camera:

```sh
python -m pip install -e .
behaviorkit --version
behaviorkit demo --output runs/demo
behaviorkit validate --config configs/people.json
behaviorkit replay --observations runs/demo/observations.jsonl --config runs/demo/config.json --trace --output runs/replayed
```

Expected: version `0.3.0`, **41 count samples**, and **one dwell event at source time 3 seconds**. Check `summary.json` for `status: "ok"`. This synthetic example tests rules, not computer-vision accuracy.

Install this checkout or a verified wheel from this repository's releases. There is no `pip install behaviorkit` instruction for this project; similarly named PyPI packages may be unrelated.

## Choose a vision runtime

Install one ONNX Runtime distribution per environment. CPU and GPU distributions share an import namespace. Do not combine `cuda` with `cpu` or `apple`.

### Intel or AMD CPU

```sh
python -m pip install -e ".[vision,cpu]"
behaviorkit doctor --provider cpu
```

Intel x86-64 Windows execution has been tested. The portable CPU path is implemented for AMD x86-64, but recorded v0.3 learned-model evidence does not include a physical AMD host. The architecture string `AMD64` means x86-64, not an AMD CPU vendor.

### NVIDIA GPU

```sh
python -m pip install -e ".[vision,cuda]"
behaviorkit doctor --provider cuda
```

Recorded GPU tests used an RTX 5060 Ti on Windows. A compatible NVIDIA driver, CUDA, cuDNN and ONNX Runtime remain necessary. [requirements-runtime-tested.txt](requirements-runtime-tested.txt) records the checked Windows environment, not a universal lock file. Follow the publisher's [CUDA compatibility matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

### Apple Silicon

Use native arm64 Python 3.12 on macOS 14+ for the dependency recipe checked here:

```sh
python -m pip install -e ".[vision,apple]"
python -c "import platform; print(platform.machine())"
behaviorkit doctor --provider cpu
```

The architecture should be `arm64`. CPU is the baseline; CoreML is experimental and opt-in. Dependency wheels resolved for this recipe, but actual M1 execution of all three learned models remains unverified. CoreML delegation does not identify the Apple compute unit used.

`--provider auto` prefers advertised CUDA, otherwise CPU, and records CUDA initialization fallback. Explicit provider initialization failures are errors. TensorRT support is experimental and unverified. `doctor` reports availability, not actual model execution; use a profiled benchmark after preparing artifacts.

## Prepare models

The supplied video runner supports these exact contracts:

- **RTMDet-Tiny:** MMDetection COCO export, 640-pixel input, classwise NMS.
- **D-FINE-N:** selected COCO checkpoint, 640-pixel input, deployed postprocessor.
- **RF-DETR-Nano:** 384-pixel input, sparse COCO class mapping and raw-logit decoding.

Weights are excluded from Git, wheels and source archives. [Model preparation](docs/model-contracts.md) documents publisher sources, preprocessing, class IDs, export dependencies and parity limits.

Use a separate export environment: export toolboxes can install a conflicting CPU runtime. Some upstream checkpoint loaders use pickle, so use trusted publisher checkpoints. The inference path loads local ONNX files and verified manifests without importing PyTorch.

```text
models/
  rtmdet-tiny/
    model.onnx
    manifest.json
  d-fine-n/
    model.onnx
    manifest.json
  rf-detr-nano/
    model.onnx
    manifest.json
```

Runtime checks cover artifact hashes, tensor metadata, dimensions, alignment and finite output values. Manifest integrity does not certify arbitrary weights or their licenses.

## Process video and inspect results

After installing one vision runtime and preparing artifacts:

```sh
behaviorkit run --video sample.mp4 --model rtmdet-tiny --models-dir models --config configs/people.json --provider cpu --threads 4 --sample-fps 5 --source entrance --trace --output runs/video
```

For NVIDIA, use `--provider cuda`. Input must be a local video with valid presentation timestamps. Rules evaluate during processing and result records are flushed. Configuration is read once and frozen. Use a new output directory per experiment; named result files are overwritten when a directory is reused.

Output files:

- `observations.jsonl`: exact timestamped observations evaluated by rules.
- `config.snapshot.json`: frozen configuration for replay.
- `counts.jsonl`: count samples, eligible IDs and coverage status.
- `events.jsonl`: confirmed dwell events with evidence.
- `transitions.jsonl`: optional starts, confirmations and reset reasons.
- `diagnostics.jsonl`: coverage and tracker-capacity messages.
- `stage_timings.jsonl`: decode/sampling, conversion, perception, tracking, rules and serialization.
- `summary.json`: completion/error status, hashes, provider, manifest and epoch.

Check the final summary: errors can leave partial output. Processing time is not live-camera evidence age. Replay with the saved configuration:

```sh
behaviorkit replay --observations runs/video/observations.jsonl --config runs/video/config.snapshot.json --trace --output runs/video-replayed
```

Counts, events and traces match for the same observations, effective configuration and library version.

## Configure a behavior

The schema currently accepts `count` and `dwell`. Unsupported operators and fields are errors.

```json
{
  "schema_version": "0.1",
  "zones": {"waiting_area": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.95], [0.1, 0.95]]},
  "rules": [
    {"id": "visible_people", "kind": "count", "zone": "waiting_area"},
    {"id": "long_wait", "kind": "dwell", "zone": "waiting_area", "seconds": 3, "max_gap_seconds": 0.5}
  ],
  "max_tracks": 200,
  "max_sources": 8
}
```

- Boxes are normalized XYXY; zone membership uses their bottom-center anchor. Polygon boundaries are included.
- Counts mean observed eligible confirmed tracks, not guaranteed occupancy or unique visitors.
- Dwell binds a track, rule, source and epoch and confirms once per qualifying episode.
- Missing tracks, zone exits, low confidence, uncovered observations, excessive gaps and changed epochs reset strict dwell.
- `covered: false` yields an unknown count (`null`). A covered batch with no eligible detections yields zero; detector misses remain possible.
- Gap limits must accommodate actual sampling intervals. Tracking does not establish presence behind occlusion.
- State is bounded. Oldest unmatched retained tracks may be evicted with diagnostics; excess current detections fail explicitly.

Read [behavior semantics](docs/behaviours.md) before interpreting events.

## Python API

This standard-library example produces one dwell event:

```python
from behaviorkit import BehaviorEngine, Detection, Observation, SceneConfig

scene = SceneConfig.from_dict({
    "schema_version": "0.1",
    "zones": {"all": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    "rules": [
        {"id": "count", "kind": "count", "zone": "all"},
        {"id": "wait", "kind": "dwell", "zone": "all", "seconds": 3},
    ],
})
engine = BehaviorEngine(scene, trace=True)
person = Detection("track-1", "person", 0.95, (0.2, 0.2, 0.4, 0.8))
events = []
for index in range(21):
    result = engine.process(Observation(index / 5, (person,), "entrance", "session-1"))
    events.extend(result.events)
assert len(events) == 1
assert events[0]["timestamp"] == 3.0
print(engine.dwell_states("entrance"))
engine.close_source("entrance")
```

`dwell_states()` returns copies. `close_source()` releases clocks, timers and a source slot; reopen with a fresh epoch. Serialize calls to an engine instance and manage application concurrency explicitly.

Custom perception systems can emit observations with stable IDs, labels, confidence and normalized boxes. The ONNX runner is optional. No generated expressions are evaluated. See the [API reference](docs/python-api.md) and [example](examples/python_replay.py).

## Event identity and migration

Version 0.3 events use `identity_version: 2`. IDs include the effective configuration fingerprint, rule, source, epoch, track and episode start. They differ from v0.2 IDs.

Each file job gets a fresh epoch. Saved observations preserve identities on replay. `--epoch` is for deliberately managed ingestion; reusing epochs across unrelated sources undermines deduplication. Keep identity versions separate during migration. Current IDs do not provide durable exactly-once delivery.

## Tests and evidence

Recorded v0.3 Intel Windows evidence includes:

- **80 passing tests and one Apple-only skip** in each CPU-only and CUDA environment.
- Counting, dwell and exact count/event/trace replay for all three learned models on CPU and CUDA using a controlled repeated-still-person clip.
- Twelve exploratory model/video runs on four excerpts from three sources, with exact replay.
- A 20,000-batch allocation/state test with bounded active dwell state.
- Separate provider profiles for model latency measurements.

GitHub Actions additionally passed all six initial jobs: Python 3.10/3.12 core tests, package build/install, and fixture integration on AMD EPYC Windows/Linux runners and an Apple M1 virtual macOS runner. The M1 run passed the tiny CoreML delegation test. See [recorded CI evidence](benchmarks/github-actions-v03.json); all-three-learned-model qualification on AMD/M1 remains pending.

Read [reliability-v03.json](benchmarks/reliability-v03.json), [compatibility](docs/compatibility.md) and the [video protocol](docs/video-lab.md). Recorded local evidence is separate from current GitHub Actions status. A passed fixture does not qualify every model or camera on that hardware.

The exploratory clips show normal handling, occluded shopping and fictional historical theft. **No theft/action classifier exists.** Sparse single-reviewer count checks expose errors; they do not establish modern CCTV event accuracy. Source media and posters are not bundled; provenance and terms are recorded separately.

```sh
python -m pip install -e ".[vision,cpu,dev]"
python -m pytest
```

Tests use generated ONNX fixtures and do not download learned weights. Actual-model qualification requires separately prepared artifacts.

```sh
behaviorkit benchmark engine --tracks 100 --batches 5000 --output runs/engine.json
behaviorkit benchmark models --models rtmdet-tiny d-fine-n rf-detr-nano --models-dir models --provider cpu --threads 4 --iterations 50 --warmup 10 --output runs/models.json
```

Model latency uses seeded synthetic RGB, native preprocessing and synchronized inference. It excludes video decode, tracking and rules. Shared-host timing is not an accuracy score, universal model ranking or camera-capacity promise. `run` provides fuller pipeline stage timings.

## Documentation

- [Quickstart](docs/quickstart.md) and [hardware installation](docs/installation.md)
- [CLI commands and exit codes](docs/commands.md)
- [Counting and dwell](docs/behaviours.md) and [Python API](docs/python-api.md)
- [Model preparation](docs/model-contracts.md) and [benchmarks](docs/benchmarks.md)
- [Compatibility](docs/compatibility.md) and [video evidence](docs/video-lab.md)
- [Troubleshooting](docs/troubleshooting.md) and [commercial use](docs/commercial-use.md)
- [Release notes](CHANGELOG.md)

The [hosted visual demo](https://behaviorkit-prototype-docs.auralcouch1.chatgpt.site/) currently has restricted access. Public Markdown documentation is available in this repository. The hosted introduction, navigation and getting-started guide have twelve language variants; detailed references and the lab use English.

Full visual-site rebuilding requires separately prepared media/model evidence. After following the video protocol:

```sh
python scripts/build_docs.py
python scripts/serve_docs.py --port 8080
```

## Planned features and developer needs

Next priorities: event-quality evaluation, a production tracker adapter, zone hysteresis, directional crossing, occupancy thresholds and bounded live sources. Custom adapters, temporal sequences and narrow local action models follow demonstrated use cases.

These are **planned**, not available commands. [ROADMAP.md](ROADMAP.md) defines prerequisites, acceptance criteria and a feedback process. Each advertised use case needs an event contract, representative labeled examples, tested hardware and accuracy/latency/resource budgets. No library can promise to cover every developer need in advance.

[Request a feature](https://github.com/arpanjot-coding/behaviorkit/issues/new?template=feature_request.yml) or [report a bug](https://github.com/arpanjot-coding/behaviorkit/issues/new?template=bug_report.yml). Describe observable events and normal lookalikes, rather than only a label such as suspicious behavior. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Current limits

- No RTSP recovery, multi-camera scheduler, durable delivery or restart-safe outbox.
- No motion/appearance tracker, re-identification, spatial hysteresis or camera-drift guard.
- No pickup, concealment, ownership, payment, intent or theft classifier.
- No held-out event accuracy certification or multi-hour decoded-video RSS/VRAM qualification.
- Physical AMD and M1 learned-model checks remain pending; CoreML and TensorRT are experimental.

## License

Original code, tests and docs are [Apache-2.0](LICENSE); preserve [NOTICE](NOTICE). No Ultralytics/AGPL runtime dependency is required.

Exact checkpoints, datasets, runtime binaries, codecs, GPU packages and demonstration media retain their own terms. Related model sizes may use different licenses. ONNX conversion does not grant new rights. See [commercial-use notes](docs/commercial-use.md) before distributing third-party artifacts.
