# Contributing

Start with an issue describing the developer problem, expected observable event and smallest reproduction. Use [ROADMAP.md](ROADMAP.md) for feature proposals. Evaluation, hardware qualification, examples and documentation are valuable contributions.

## Setup

Use Python 3.12 and a dedicated environment:

```sh
python -m venv .venv
# Activate it using your shell's command.
python -m pip install -e ".[vision,cpu,dev]"
python -m pytest
```

Use a separate CUDA environment for GPU checks. Do not install CPU and GPU ONNX Runtime distributions together. Automated tests use generated fixtures and do not download learned weights.

## Useful reports

Include the version, OS, actual hardware, requested/selected provider, command or Python example, configuration, expected result and observed result. Prefer minimal timestamped observations for rule failures. Include a source-time or event reference where possible.

Remove credentials and personal data. Submit only artifacts you are entitled to share. Model hashes and publisher URLs often identify an artifact without uploading weights or footage.

## Implementation expectations

- Keep the standard-library core independent of ML packages.
- Preserve source time, coverage, epochs, resource bounds and deterministic replay.
- Add meaningful regression coverage for behavior changes and event-changing edge cases.
- Share inference across rules; measure actual bottlenecks before optimizing.
- Distinguish observations, tracking assumptions and semantic claims.
- Update docs and CHANGELOG for public behavior or identity changes.
- Record actual hardware when adding platform evidence; simulated provider names are not qualification.

Run checks appropriate to the change. Model/export changes also need reference contract checks and real-model execution. Accuracy changes need held-out clips and normal lookalikes. Passing CI alone does not establish camera accuracy.

## Pull requests

Explain the problem, resulting behavior, validation and material limits. Keep PRs reviewable. Exclude model weights, generated builds, local environments, hosting credentials and private configuration. Preserve provenance and notices.

## Hardware and documentation

Follow [compatibility.md](docs/compatibility.md) and `scripts/validate_platform.py`. Publish actual host identity, reports and failures. Learned-model support additionally needs the controlled model regression and provider profile. CoreML fixture delegation does not prove all shortlisted graphs work.

Markdown docs are readable on GitHub. Rebuilding the visual site requires separately prepared evidence/media; follow [video-lab.md](docs/video-lab.md).
