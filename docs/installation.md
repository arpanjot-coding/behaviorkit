# Install on your hardware

Download and extract the source checkout. This prototype is **not published to PyPI**. Run installation commands from the extracted directory. Python 3.12 is the recommended vision runtime; the dependency-free core supports Python 3.10+.

## Create an environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

## Intel or AMD CPU

```sh
python -m pip install -e ".[vision,cpu]"
behaviorkit doctor --provider cpu
```

This installs CPU ONNX Runtime without CUDA dependencies. Intel Windows execution has been verified. The portable CPU adapter is implemented for AMD x86-64; physical AMD validation is pending. The architecture string `AMD64` means x86-64 and does not identify the CPU vendor.

## NVIDIA GPU

```sh
python -m pip install -e ".[vision,cuda]"
behaviorkit doctor --provider cuda
```

Tested here on Windows with an RTX 5060 Ti, Python 3.12 and ONNX Runtime GPU. Other GPUs and Linux require a smoke test. A system NVIDIA driver remains necessary. Match CUDA, cuDNN and ONNX Runtime using the publisher's [CUDA provider matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). CUDA DLL preloading occurs only for CUDA/TensorRT execution.

## Apple M1 and other Apple Silicon

Use **native arm64 Python 3.12 on macOS 14+** for the dependency recipe checked in this release:

```sh
python -m pip install -e ".[vision,apple]"
python -c "import platform; print(platform.machine())"
behaviorkit doctor --provider cpu
```

The architecture must be `arm64`, not an Intel Python running under Rosetta. The `apple` extra installs CPU ONNX Runtime, whose official macOS distribution also includes CoreML. Wheel availability for this recipe was checked; no physical M1 was available here. See [compatibility status](compatibility.md).

CPU is the default baseline on Macs. To experiment with CoreML:

```sh
behaviorkit doctor --provider coreml
behaviorkit benchmark models --models rtmdet-tiny --models-dir models --provider coreml --output runs/coreml.json
```

The adapter requests MLProgram, static input shapes and all compute units, with CPU fallback for unsupported operators. [CoreML's documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) describes operator limits and compute-unit options. Availability is not proof of graph delegation. A CoreML profile event does not identify the physical GPU or Neural Engine used. Benchmark the exact exported model. Full shortlisted-model CoreML execution remains unverified.

## Verify actual execution

After [preparing model artifacts](model-contracts.md):

```sh
behaviorkit benchmark models --models rtmdet-tiny --provider cpu --threads 4 --models-dir models --iterations 50 --warmup 10 --output runs/cpu-smoke.json
```

Replace `cpu` with `cuda` for NVIDIA. `doctor` inspects availability only. The benchmark checks a separate ONNX Runtime kernel profile. Explicit provider requests fail if initialization fails; `auto` can select CPU after failed CUDA initialization and records that selection.

## Avoid conflicting installations

`onnxruntime` and `onnxruntime-gpu` share Python modules. Install one distribution per environment: do not combine `cuda` with `cpu` or `apple`. Export tools may pull in CPU ONNX Runtime; keep model export and runtime environments separate.

Optional `--provider tensorrt` requests FP16 and engine caching. Matching TensorRT libraries are a separate prerequisite. This path remains experimental; CUDA success does not verify TensorRT.
