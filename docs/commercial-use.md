# Commercial-friendly by design

Original prototype code, tests and documentation are Apache-2.0. There is no runtime Ultralytics dependency, no LLM service, no API key and no network fallback during inference. Core replay uses only the Python standard library.

## Separate the license layers

The original library license does not replace the terms for model weights, their training/pretraining data, GPU runtimes, video codecs or footage. Preserve the source checkpoint URL, artifact hashes and upstream notices. The manifest records a license basis, not blanket clearance.

The shortlist uses Apache-oriented paths: RTMDet-Tiny from MMDetection, D-FINE-N's COCO checkpoint, and the Apache RF-DETR Nano variant. Some D-FINE pretrained variants and some RF-DETR sizes have additional or different conditions. Review exact artifacts before distributing a commercial application.

## What the archive includes

Source code, tests, example observations, configuration, documentation and measured JSON reports. Model checkpoints and exported weights are excluded from the source archive and wheel. A separate preparation step retrieves the selected publisher artifacts.

NVIDIA libraries and FFmpeg-enabled decoder binaries retain their own redistribution terms. Avoid bundling an entire export environment just because it was convenient for a benchmark.

## Retention

Core processing retains active state only. The CLI explicitly writes counts, events, observations and timing metadata to its chosen output directory. Video files and their credentials are not automatically uploaded. The offline runner does not retain frame images or create evidence clips.

Deployment-specific privacy and monitoring obligations belong to the application using the library. This prototype does not add face recognition or cross-camera identity.
