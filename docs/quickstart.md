# Your first explained event

Start with replay. It exercises the same counting and dwell engine used by video processing, without downloading a model or connecting a cloud service.

## Install from the checkout

Download and extract the source archive, open a terminal in its root, and use Python 3.12. This prototype is not on PyPI: install the local directory, not a similarly named package.

```sh
python -m pip install -e .
behaviorkit --version
behaviorkit demo --output runs/demo
```

Expect 41 count samples and one dwell event. Person 1 stays for the three-second threshold. Person 2 leaves earlier and produces no dwell event. The data is synthetic and labeled as such.

Alternatively, download the [v0.3.0 wheel](https://github.com/arpanjot-coding/behaviorkit/releases/download/v0.3.0/behaviorkit_prototype-0.3.0-py3-none-any.whl) and install the local file:

```sh
python -m pip install ./behaviorkit_prototype-0.3.0-py3-none-any.whl
```

The wheel contains the dependency-free core, not model weights. The source checkout includes model-preparation scripts and examples. Choose the vision/runtime extras in the installation guide for video inference.

## Inspect the evidence

Open `runs/demo/events.jsonl` for the confirmed rule event, `counts.jsonl` for per-observation visible counts, and `summary.json` for replay hashes. Every event records its rule, zone, track, source epoch, supported interval and gap policy.

```sh
behaviorkit replay --observations runs/demo/observations.jsonl --config runs/demo/config.json --output runs/replayed
```

Replaying the same observations and configuration produces identical event and count files. Summary output paths can differ; processing speed is not part of event identity.

## Use your own video

Choose the [CPU, NVIDIA or Apple installation](installation.md), prepare a model artifact, and edit `configs/people.json` for the camera's normalized zone coordinates.

```sh
behaviorkit run --video sample.mp4 --model rtmdet-tiny --models-dir models --config configs/people.json --provider auto --sample-fps 5 --output runs/video
```

The video runner evaluates counts and events as it processes sampled frames, and writes observations, stage timings and a frozen `config.snapshot.json` for replay. Add `--trace` for explanations. Check `summary.json` for `status: "ok"` and the selected provider. Use `--provider cpu --threads 4` to select CPU explicitly. This release processes local video files. It does not run a persistent camera server or send notifications.

```sh
behaviorkit replay --observations runs/video/observations.jsonl --config runs/video/config.snapshot.json --trace --output runs/video-replayed
```

## What comes next

Read [counting and dwell semantics](behaviours.md) before interpreting results. Run the [model benchmark](benchmarks.md) on your machine. Use labeled, rights-cleared camera footage to establish event accuracy; a successful demo or high FPS does not establish reliable monitoring.
