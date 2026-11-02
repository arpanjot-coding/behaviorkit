"""Hardware selection without importing a GPU SDK or an ML framework."""
from dataclasses import dataclass
import platform

PROVIDERS={"cpu":"CPUExecutionProvider","cuda":"CUDAExecutionProvider","tensorrt":"TensorrtExecutionProvider","coreml":"CoreMLExecutionProvider"}
CHOICES=("auto",*PROVIDERS)


@dataclass(frozen=True)
class ProviderSelection:
    requested: str
    selected: str
    ort_name: str
    chain: tuple
    reason: str


def select_provider(requested,available,*,system=None,machine=None):
    if requested not in CHOICES:
        raise ValueError(f"provider must be one of {', '.join(CHOICES)}")
    system=system or platform.system();machine=machine or platform.machine()
    selected=requested
    reason="Explicit provider request; initialization failure is an error."
    if requested=="auto":
        selected="cuda" if "CUDAExecutionProvider" in available else "cpu"
        reason="Auto prefers an advertised CUDA provider; otherwise uses portable CPU. CoreML is opt-in."
    if selected=="coreml" and (system!="Darwin" or machine.lower() not in {"arm64","aarch64"}):
        raise ValueError("coreml requires native arm64 Python on Apple Silicon macOS; use --provider cpu on this host")
    ort_name=PROVIDERS[selected]
    if ort_name not in available:
        extra="apple" if selected=="coreml" else ("cpu" if selected=="cpu" else "cuda")
        raise RuntimeError(f"Requested {ort_name} is unavailable. Install .[vision,{extra}] in a separate compatible environment. Available: {list(available)}")
    if selected=="tensorrt":
        if "CUDAExecutionProvider" not in available:raise RuntimeError("TensorRT path requires CUDA for unsupported nodes")
        chain=(ort_name,"CUDAExecutionProvider","CPUExecutionProvider")
    elif selected=="coreml":
        chain=((ort_name,{"ModelFormat":"MLProgram","MLComputeUnits":"ALL","RequireStaticInputShapes":"1","EnableOnSubgraphs":"0"}),"CPUExecutionProvider")
    else:chain=(ort_name,) if selected=="cpu" else (ort_name,"CPUExecutionProvider")
    return ProviderSelection(requested,selected,ort_name,chain,reason)
