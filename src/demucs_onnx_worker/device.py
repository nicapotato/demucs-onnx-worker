"""Map an ONNX Runtime provider list to the short device id the GUI shows.

The GUI stores this in a 15-character field (``device[16]`` including the NUL).
``gpu_name`` is the human adapter name sent alongside it (``gpu`` field).
"""

from __future__ import annotations

import shutil
import subprocess
import sys

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


def _dxgi_adapter0_name() -> str | None:
    """Description of DXGI adapter 0: the adapter ORT DirectML uses at device_id 0."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class DXGI_ADAPTER_DESC(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", ctypes.c_uint),
            ("DeviceId", ctypes.c_uint),
            ("SubSysId", ctypes.c_uint),
            ("Revision", ctypes.c_uint),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", ctypes.c_int64),
        ]

    # IID_IDXGIFactory {7b7166ec-21c7-44ae-b21a-c9ae321ae369}
    tail = (ctypes.c_ubyte * 8)(0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69)
    iid = GUID(0x7B7166EC, 0x21C7, 0x44AE, tail)
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        return None

    def vcall(obj: ctypes.c_void_p, index: int, restype, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])

    # IUnknown: 0 QueryInterface, 1 AddRef, 2 Release. IDXGIObject: 3-6.
    # IDXGIFactory::EnumAdapters = 7. IDXGIAdapter::GetDesc = 8.
    try:
        adapter = ctypes.c_void_p()
        enum = vcall(factory, 7, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        if enum(factory, 0, ctypes.byref(adapter)) != 0:
            return None
        desc = DXGI_ADAPTER_DESC()
        get_desc = vcall(adapter, 8, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC))
        ok = get_desc(adapter, ctypes.byref(desc)) == 0
        vcall(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        vcall(factory, 2, ctypes.c_ulong)(factory)
    name = desc.Description.strip() if ok else ""
    return name or None


def _nvidia_smi_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = subprocess.run(
        [exe, "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    name = out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    return name or None


def gpu_name(device: str) -> str | None:
    """Adapter name for a GPU device id, or None for CPU / when it cannot be read."""
    try:
        if device == "dml":
            return _dxgi_adapter0_name()
        if device in ("cuda", "tensorrt"):
            return _nvidia_smi_name()
    except Exception:  # noqa: BLE001 - a missing name must never fail separation
        return None
    return None
