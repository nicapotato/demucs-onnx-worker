"""Map an ONNX Runtime provider list to the short device id the GUI shows.

The GUI stores this in a 15-character field (``device[16]`` including the NUL).
"""

from __future__ import annotations

_NAMES = {
    "CoreMLExecutionProvider": "coreml",
    "DmlExecutionProvider": "dml",
    "CUDAExecutionProvider": "cuda",
    "CPUExecutionProvider": "cpu",
    "TensorrtExecutionProvider": "tensorrt",
}


def device_name(providers: list[str]) -> str:
    """Prefer the first GPU provider. CPU is the fallback the GUI already knows."""
    for provider in providers:
        if provider == "CPUExecutionProvider":
            continue
        if provider in _NAMES:
            return _NAMES[provider]
        raw = provider.replace("ExecutionProvider", "").lower()
        if raw:
            return raw[:15]
    return "cpu"
