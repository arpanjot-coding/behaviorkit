# Troubleshooting

## The command is not found

Activate the environment containing the editable install, or use `python -m behaviorkit --help`. Install from the checkout root with `python -m pip install -e .`; the prototype is not published to PyPI.

## CUDA is available but the model fails

Run `behaviorkit doctor --provider cuda`. Check driver/runtime compatibility and confirm you installed `onnxruntime-gpu`, not both runtime distributions. Real GPU execution is verified by the model benchmark's kernel profile, not by `doctor` alone.

## Missing model or manifest

The runtime expects `models/<preset>/model.onnx` and `manifest.json`. Run the preparation script in an export environment or copy a verified artifact folder. A random ONNX file with the same tensor shapes does not establish the expected preprocessing or class map.

## SHA256 mismatch

Stop and identify the changed artifact. Regenerate the manifest through the preparation script after a deliberate export. Do not edit the expected hash merely to bypass a provenance check.

## Dwell never fires

Check that the track remains inside the zone, satisfies confidence, and retains one ID. Source gaps larger than `max_gap_seconds` reset the timer. At 2 Hz, a gap policy below 0.5 seconds cannot support the requested cadence. Boundary jitter and the simple IoU tracker can reset episodes.

## Counts flicker or IDs switch

The video path uses a small IoU baseline without appearance re-identification. Increase sample frequency, improve the camera view, adjust the zone or use your own tracked observations. Measure changes against labeled events rather than assuming smoothing improves correctness.

## Source timestamps fail

The offline runner requires presentation timestamps that strictly increase. Normalize the input file with a deliberate timestamp policy, or provide timestamped observation replay. The prototype does not silently replace source time with processing wall-clock time.

## Output files look incomplete

Commands can write partial outputs before an input/runtime error. Check the exit code, terminal message and `summary.json`. Only `status: "ok"` denotes success. `running` can mean interruption; `error` marks failure and partial outputs. Use a fresh output directory for each experiment and preserve its configuration and model manifests.


## CPU is slower than expected

Benchmark `--provider cpu --threads 4`, then vary thread count. Zero lets ONNX Runtime choose CPU threads; more threads are not always faster. `AMD64` is an architecture label even on Intel Windows computers.

## CoreML is unavailable or reports no delegation

Use native arm64 Python, the macOS 14+ recipe and the `apple` extra in a clean environment. `doctor --provider coreml` checks availability; `benchmark models --provider coreml` checks the exact graph. Unsupported operators can execute on CPU. Prefer CPU until your exported models pass an actual M1 test. CoreML, TensorRT and physical AMD results are not verified by Intel/NVIDIA tests.
