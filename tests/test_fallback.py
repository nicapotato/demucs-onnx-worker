from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pytest

from demucs_onnx_worker import events, separate
from demucs_onnx_worker.separate import N_CHANNELS, N_SAMPLES, SOURCES


class FakeSession:
    def __init__(self, providers: list[str], fail: bool) -> None:
        self._providers = providers
        self._fail = fail
        self.calls = 0

    def get_providers(self) -> list[str]:
        return self._providers

    def run(self, _outputs, feeds):
        self.calls += 1
        if self._fail:
            raise RuntimeError(
                "Non-zero status code returned while running DmlFusedNode_0_0 node."
                ' Name:"DmlFusedNode_0_0"'
            )
        batch = feeds["mix"]
        return [np.zeros((1, len(SOURCES), N_CHANNELS, batch.shape[-1]), dtype=np.float32)]


@pytest.fixture
def fake_audio(monkeypatch):
    mix = np.zeros((N_CHANNELS, N_SAMPLES + 1000), dtype=np.float32)
    monkeypatch.setattr(separate, "load_audio", lambda _p, target_sr: (mix, target_sr))
    written: list[str] = []
    monkeypatch.setattr(
        separate, "write_audio", lambda path, _a, _sr, bitrate_kbps: written.append(path.name)
    )
    return written


def test_dml_session_options() -> None:
    opts = separate.session_options(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert opts.enable_mem_pattern is False
    assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert separate.session_options(separate.CPU_ONLY).enable_mem_pattern is True


def test_gpu_run_failure_retries_on_cpu(tmp_path, fake_audio, capsys) -> None:
    gpu = FakeSession(["DmlExecutionProvider", "CPUExecutionProvider"], fail=True)
    cpu = FakeSession(["CPUExecutionProvider"], fail=False)
    events.set_enabled(True)
    try:
        used = separate.separate_file(
            gpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
            cpu_fallback=lambda: cpu,
        )
    finally:
        events.set_enabled(False)
    assert used is cpu
    assert cpu.calls > 0
    assert sorted(fake_audio) == sorted(f"{s}.wav" for s in SOURCES)
    captured = capsys.readouterr()
    lines = [json.loads(ln) for ln in captured.out.splitlines() if ln]
    fallback = [ln for ln in lines if ln.get("stage") == "fallback_cpu"]
    assert fallback and fallback[0]["device"] == "cpu"
    assert lines[-1]["event"] == "done"
    assert "DmlFusedNode" in captured.err


def test_cpu_run_failure_is_raised(tmp_path, fake_audio) -> None:
    cpu = FakeSession(["CPUExecutionProvider"], fail=True)
    with pytest.raises(RuntimeError, match="DmlFusedNode"):
        separate.separate_file(
            cpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
            cpu_fallback=lambda: cpu,
        )
