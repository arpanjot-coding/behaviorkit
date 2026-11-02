# The video lab: actions, evidence and limits

Open the [interactive video lab](https://behaviorkit-prototype-docs.auralcouch1.chatgpt.site/demo.html). It contains four excerpts from three online sources, run through RTMDet-Tiny, D-FINE-N and RF-DETR-Nano on the actual NVIDIA host. No model or LLM runs in the browser. The player synchronizes the original clip with locally computed detections, confirmed tracks, rule state and event records.

## What it tests

The modern product-handling clip shows an ordinary person handling goods. The shopping-basket clip adds substantial foreground occlusion. Two excerpts from Edwin S. Porter's public-domain film *The Kleptomaniac* (1905) show fictional taking and concealment in a crowded shop. Historical footage is a domain-shift stress case, not evidence of modern CCTV accuracy. The film excerpts are not independent source videos.

The modern Pexels clips depict normal shopping/product handling and are never labeled criminal. The actor portrayals in the film are clearly marked fictional. Media credits, rights links, transformations and SHA256 hashes accompany the demo. Source media retain their own terms; they are not Apache-2.0 library assets.

## Fixed test settings

- All three models use their documented native preprocessing contracts and the same prepared clip bytes.
- 5 Hz inference, person confidence threshold 0.30, full-image zone.
- Maximum-cardinality IoU tracker, two-hit confirmation, one-second internal expiry.
- Dwell threshold three seconds; maximum supported sample gap 0.5 seconds.
- A missing eligible track resets dwell immediately, even while the tracker retains an internal candidate.
- Settings are fixed across cases. No per-clip confidence tuning or hand-edited detections.

Twelve runs completed. A separate ONNX profile verified CUDA execution for each model. Each run's saved observations were independently replayed through the public rule-evaluation path; counts and events matched exactly. That establishes reproducibility, not detector or action accuracy.

## Observed results

All three models matched six out of six selected count snapshots in the clear normal product-handling clip. Each maintained one confirmed identity and emitted one dwell event. Normal handling therefore satisfies a dwell rule without constituting a theft prediction.

All three matched the two scorable basket snapshots, but the occluded portions exposed false detections, missed people and identity fragmentation. Two ambiguous snapshots were deliberately left unscored. No model supported a three-second dwell event in this clip.

On the fictional bread-taking scene, RTMDet-Tiny matched three of six checked counts; D-FINE-N and RF-DETR-Nano each matched two of six. RTMDet-Tiny emitted no dwell event; the other two emitted one each. These differences are visible in the player. The crowded film scene has no exact manual count labels because overlapping background figures were ambiguous; its many tracks/events must not be interpreted as validated occupancy or theft alarms.

The complete automated suite, including the new evidence exporter, passed **80 tests with one Apple hardware test skipped**, in both the CPU-only and CUDA environments on the Intel Windows host. The tests verify configuration, temporal semantics, regression cases, decoded-video/ONNX transport, replay equality, annotation abstentions and source-file protection. Passing software tests does not override the observed detection failures in these clips.

## How to read the player

Select a case and model. Pause, drag the timeline, or step between inference samples. Switch between detector boxes and confirmed tracks. Blue boxes carry temporary tracker IDs; the small bottom-center point is the anchor used for zone membership. Amber boxes are filtered detector results before association, not unprocessed network tensors.

### Playback and inspection controls

- **Jump by:** enter any interval from 0.1 to 60 seconds, including decimals, then use the backward/forward buttons. The player clamps jumps to the clip's beginning/end and preserves whether playback was running.
- **Sample arrows:** pause and move to the previous/next saved inference sample. The 5 Hz samples are 0.2 seconds apart; these are not individual 15 fps video frames.
- **Start / End:** pause at the clip boundary. Play restarts a completed clip.
- **Speed and loop:** use 0.25×, 0.5×, 1×, 1.5× or 2× playback, and optionally repeat the clip.
- **Keyboard:** focus the video area. Space toggles play/pause, left/right arrows jump by the selected interval, Shift+arrows step inference samples, and Home/End go to clip boundaries. Editing an input does not trigger video shortcuts.

The three timeline lanes distinguish actual dwell events, actual track-timer resets and approximate human reference intervals. Click a marker to pause at that time. The scene guide provides readable human notes with the same navigation. Human annotations do not become model predictions.

Select a track chip to highlight its box and inspect confidence, the normalized anchor, confirmation status and the current supported dwell span. The progress bar shows the three-second rule threshold, capped at three visually; the episode text retains the actual elapsed duration. A selected track that disappears stays explicitly absent instead of silently transferring its timer to another ID. Auto selection follows the longest current span.

The model comparison cards show each model's whole-clip count spot checks, dwell event count, temporary ID count and reset records. Selecting a card switches the evidence at the current video time. Temporary IDs and reset records are diagnostic counts, not unique real people or verified physical exits.

Overlays and rule values are withheld while a seek is still being decoded. Switching scenes/models clears prior sample data and download links until the requested evidence arrives. Older asynchronous responses cannot replace a newer selection. Failed evidence loads offer a retry.

### Player verification

Browser checks exercised custom jumps, sample stepping and boundary clamping across all 12 model/scene combinations. Additional checks covered both jump directions during playback, all five speeds, looping, invalid intervals, keyboard shortcuts, track evidence, timeline markers, model changes at the same video time and layouts at 320, 390, 768 and 1024 pixels. Simulated failures covered initial-report retry, model-data retry, clearing stale results/downloads and ignoring late responses after a scene/model switch. These verify the player, not additional behaviour-recognition accuracy.

Download the [player check results](../benchmarks/video-lab-player-qa.json) and [loading/failure check results](../benchmarks/video-lab-loading-qa.json).

The inspector shows raw box count, confirmed track count, current rule count and the longest active dwell span. Event entries jump to the exact confirming sample. Reset messages indicate loss of supporting track evidence; they do not prove that the person physically departed. Human annotations are shown in a separate colored panel and are never mixed into model output.

The first sample can show a detected person and zero confirmed tracks: the tracker requires two matches. The file runner marks decoded frames covered; it does not infer whether a shelf blocks a person. A missing detection can consequently produce zero and a dwell reset. The demo calls out that visibility limitation rather than inventing a continuous identity through it.

## What the library cannot conclude

**Version 0.3 has no theft, pickup, concealment, payment or ownership classifier.** A count or dwell event cannot supply those missing facts. The video lab keeps the action result explicitly unsupported. It does not report a false theft precision/recall score for a classifier that does not exist.

Complex behaviour needs evaluated evidence for person–object relationships and temporal sequences, alongside normal lookalikes. For theft-related applications, authorization/payment context may remain unavailable from video alone. The next implementation should target an observable sequence such as pickup–carry–exit, with human review and measured false alerts, before assigning broader meanings.

## Reproduce locally

The source ZIP includes the source manifest, manual annotations, preparation/export scripts and precomputed evidence. Video assets are excluded from the source ZIP and wheel; retrieve the originals from their listed publishers under the recorded media terms. Put the source files in a local folder with the filenames in `examples/video-lab/sources.json`. Prepare the model artifacts separately.

```sh
python scripts/prepare_video_demo.py --sources-dir path/to/local/source-videos
python scripts/build_video_demo.py --provider cuda --threads 4 --models-dir models
```

For a CPU evaluation, use `--provider cpu`. The builder exports per-frame data, observations, rule configuration, counts and events under `examples/video-lab/results/`. Replayed counts/events must equal the original evaluation. The saved timing values are build-host measurements, not current browser speed or a controlled model ranking.

Run one prepared video directly:

```sh
behaviorkit run --video examples/video-lab/media/bread.mp4 --model rtmdet-tiny --models-dir models --config examples/video-lab/config.json --provider cuda --threads 4 --sample-fps 5 --source bread --output runs/bread
```

Build the static documentation after results are present:

```sh
python scripts/build_docs.py
python scripts/serve_docs.py --port 8080
```

Open `/demo.html`. Download [all run summaries](../examples/video-lab/results/report.json), [the source manifest](../examples/video-lab/sources.json) and [the human annotations](../examples/video-lab/annotations.json) directly from the demo. Frame evidence and JSONL records are linked per model/case.

## Next evaluation gate

Use a larger rights-cleared dataset of staged observable actions and normal lookalikes. Annotate continuous person/object identities, intervals, occlusions and action boundaries. Separate actors, scenes and days across tuning and held-out evaluation. Measure count error, identity switches, action-event precision/recall, duplicate alerts and false alerts per camera-hour. The current sparse single-reviewer spot checks are an exploratory starting point.

## Media sources

- Gustavo Fring, [Woman in a Store Placing Products Respectively](https://www.pexels.com/video/woman-in-a-store-placing-products-respectively-5103990/), under the [Pexels License](https://www.pexels.com/license/).
- Kampus Production, [A Woman Carrying a Shopping Basket](https://www.pexels.com/video/a-woman-carrying-a-shopping-basket-8420585/), under the same license.
- Edwin S. Porter / Edison Manufacturing Company, [The Kleptomaniac (1905), Wikimedia Commons file and public-domain statement](https://commons.wikimedia.org/wiki/File:The_Kleptomaniac_(1905)_-_yt.webm). Silent, resized excerpts; source offsets are recorded in the manifest.

The local preview server implements HTTP byte ranges so the browser can seek through the MP4 files. The ordinary Python `http.server` does not provide the same seeking behavior.
