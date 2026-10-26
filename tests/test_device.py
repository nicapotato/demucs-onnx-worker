from __future__ import annotations

import sys

import pytest

from demucs_onnx_worker import device


def test_device_name_prefers_gpu_provider() -> None:
    assert device.device_name(["DmlExecutionProvider", "CPUExecutionProvider"]) == "dml"
    assert device.device_name(["CPUExecutionProvider"]) == "cpu"


def test_cpu_has_no_gpu_name() -> None:
    assert device.gpu_name("cpu") is None


@pytest.mark.skipif(sys.platform == "win32", reason="DXGI is real on Windows")
def test_dml_name_is_none_off_windows() -> None:
    assert device.gpu_name("dml") is None


def test_cuda_name_from_nvidia_smi(monkeypatch) -> None:
    class Done:
        stdout = "NVIDIA GeForce RTX 3080\n"

    monkeypatch.setattr(device.shutil, "which", lambda _exe: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(device.subprocess, "run", lambda *_a, **_k: Done())
    assert device.gpu_name("cuda") == "NVIDIA GeForce RTX 3080"


def test_gpu_name_errors_are_swallowed(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError("no driver")

    monkeypatch.setattr(device.shutil, "which", lambda _exe: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(device.subprocess, "run", boom)
    assert device.gpu_name("cuda") is None
