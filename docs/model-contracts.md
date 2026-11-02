# Prepare the shortlisted models

Model preparation is a build-time step. Runtime loads local ONNX files and verified manifests. Checkpoints are excluded from source archives and wheels. Use trusted publisher sources and preserve their terms and provenance.

## Separate export environment

Recommended: use Python 3.12 and an export-only virtual environment. Install the prototype there for its manifest definitions, then install CPU PyTorch and the export dependencies. Export packages may install the CPU ONNX Runtime distribution; do not use that environment as the final CUDA runtime.

```sh
python -m pip install -e .
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install "rfdetr[onnx]" "mmcv-lite==2.1.0" "mmdet==3.3.0" timm scipy onnxscript tensorboard faster-coco-eval calflops loguru
git clone https://github.com/Peterande/D-FINE.git upstream-dfine
python scripts/prepare_models.py --models rtmdet-tiny --models-dir models
python scripts/prepare_models.py --models d-fine-n --models-dir models --dfine-source upstream-dfine
python scripts/prepare_models.py --models rf-detr-nano --models-dir models
```

The script downloads only the selected, explicit publisher checkpoint URL. It enables pickle loading for those trusted upstream checkpoints where required. Never substitute an untrusted checkpoint. The runtime itself does not import PyTorch or any export toolbox.

## Artifact folder

Each preset uses `models/<name>/model.onnx` and `manifest.json`. The manifest records hashes, source checkpoint URL, class/decoding contract, precision, export metadata and parity status. Runtime rejects missing manifests, mismatched names/contracts and changed hashes. Version 0.3 also validates named input/output metadata, tensor rank, fixed dimensions and data types. Runtime results must have aligned query counts, finite values and valid probability ranges. Malformed outputs fail the run instead of silently becoming an observed zero.

## RTMDet-Tiny

The export uses MMDetection 3.3.0's COCO checkpoint and actual point priors. It exports decoded XYXY boxes and sigmoid class scores before classwise NMS. No compiled MMCV operations are needed for this network's export; an import-only placeholder fails if an unavailable operation is accidentally invoked.

Input: BGR, aspect-preserving resize to 640, right/bottom 114 padding, channel means `[103.53,116.28,123.675]`, standard deviations `[57.375,57.12,58.395]`, float32 NCHW. Person class is 0. Output names are `boxes` and `scores`.

## D-FINE-N

Use the official `dfine_n_coco.pth` COCO checkpoint, not an Objects365-derived variant by accident. Export includes the deployed postprocessor. Input is RGB resized to 640, divided by255, float32 NCHW; `orig_target_sizes` is int64 `[[original_width,original_height]]`. Outputs are `labels`, `boxes`, `scores`; boxes are already in original-image pixels and person class is 0.

## RF-DETR-Nano

Native input is 384: RGB, half-pixel bilinear resize, divided by255 and ImageNet normalized. Outputs `dets` are normalized CXCYWH; `labels` are raw logits. Decode sigmoid with global top-K query/class pairs. Official COCO checkpoints use sparse COCO class slots: person is 1, and the final slot 90 is foreground. Do not assume per-query argmax or a removable final background slot.

The script uses the official exporter and records that independently checked tensor parity is unavailable unless explicitly added. Export compatibility and real execution status appear in the measured benchmark files.

## Commercial terms

MMDetection and D-FINE source are Apache-2.0. RF-DETR Nano is documented under Apache-2.0; some other sizes use different terms. Exact checkpoints, training/pretraining provenance, datasets and binary dependencies require separate review. Exporting to ONNX does not grant new rights.
