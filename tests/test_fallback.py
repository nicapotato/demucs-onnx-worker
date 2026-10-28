from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pytest

from demucs_onnx_worker import cli, dml_state, events, separate
from demucs_onnx_worker.separate import N_CHANNELS, N_SAMPLES, SOURCES

DML = ["DmlExecutionProvider", "CPUExecutionProvider"]
GPU = "NVIDIA GeForce RTX 3080"
DML_ERROR = (
    "Non-zero status code returned while running DmlFusedNode_0_0 node."
    " Name:'DmlFusedNode_0_0' Status Message: DmlGraphFusionHelper.cpp(1041) 80070057"
)


class FakeSession:
    def __init__(self, providers: list[str], fail: bool, label: str = "") -> None:
        self._providers = providers
        self._fail = fail
        self.label = label
        self.calls = 0

    def get_providers(self) -> list[str]:
        return self._providers

    def get_outputs(self) -> list:
        return []

    def run(self, _outputs, feeds):
        self.calls += 1
        if self._fail:
            raise RuntimeError(DML_ERROR)
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
    opts = separate.session_options(DML)
    assert opts.enable_mem_pattern is False
    assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    with pytest.raises(RuntimeError):
        opts.get_session_config_entry(separate.DML_DISABLE_FUSION)
    unfused = separate.session_options(DML, dml_fusion=False)
    assert unfused.get_session_config_entry(separate.DML_DISABLE_FUSION) == "1"
    assert separate.session_options(separate.CPU_ONLY).enable_mem_pattern is True


def test_failure_hook_replaces_session(tmp_path, fake_audio) -> None:
    gpu = FakeSession(DML, fail=True)
    cpu = FakeSession(separate.CPU_ONLY, fail=False)
    used = separate.separate_file(
        gpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
        on_failure=lambda _s, _e: cpu,
    )
    assert used is cpu
    assert sorted(fake_audio) == sorted(f"{s}.wav" for s in SOURCES)


class BeepSession(FakeSession):
    """Runs without error but returns a full-scale sine in every stem."""

    def run(self, _outputs, feeds):
        self.calls += 1
        n = feeds["mix"].shape[-1]
        tone = 0.9 * np.sin(2 * np.pi * 1000 * np.arange(n) / 44100).astype(np.float32)
        return [np.broadcast_to(tone, (1, len(SOURCES), N_CHANNELS, n)).copy()]


def test_check_stems_accepts_a_separation() -> None:
    rng = np.random.default_rng(0)
    parts = rng.normal(0, 0.1, (len(SOURCES), N_CHANNELS, 4096)).astype(np.float32)
    mix = parts.sum(axis=0)
    noisy = parts + rng.normal(0, 0.02, parts.shape).astype(np.float32)
    separate.check_stems(noisy, mix, 0)
    separate.check_stems(np.zeros_like(parts), np.zeros_like(mix), 0)


def test_check_stems_rejects_garbage() -> None:
    mix = np.full((N_CHANNELS, 4096), 0.1, dtype=np.float32)
    bad = np.zeros((len(SOURCES), N_CHANNELS, 4096), dtype=np.float32)
    bad[0, 0, 7] = np.nan
    with pytest.raises(separate.BadGpuOutput, match="NaN"):
        separate.check_stems(bad, mix, 0)
    beep = np.full((len(SOURCES), N_CHANNELS, 4096), 0.9, dtype=np.float32)
    with pytest.raises(separate.BadGpuOutput, match="do not add up"):
        separate.check_stems(beep, mix, 2)
    with pytest.raises(separate.BadGpuOutput):
        separate.check_stems(beep, np.zeros_like(mix), 0)


def test_gpu_beep_falls_back(tmp_path, fake_audio) -> None:
    gpu = BeepSession(DML, fail=False)
    cpu = FakeSession(separate.CPU_ONLY, fail=False)
    seen: list[Exception] = []

    def on_failure(_s, exc):
        seen.append(exc)
        return cpu

    used = separate.separate_file(
        gpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
        on_failure=on_failure,
    )
    assert used is cpu and gpu.calls == 1
    assert isinstance(seen[0], separate.BadGpuOutput)


def test_cpu_output_is_not_checked(tmp_path, fake_audio) -> None:
    cpu = BeepSession(separate.CPU_ONLY, fail=False)
    separate.separate_file(
        cpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
        on_failure=lambda _s, _e: None,
    )
    assert len(fake_audio) == len(SOURCES)


def test_failure_without_replacement_is_raised(tmp_path, fake_audio) -> None:
    cpu = FakeSession(separate.CPU_ONLY, fail=True)
    with pytest.raises(RuntimeError, match="DmlFusedNode"):
        separate.separate_file(
            cpu, tmp_path / "a.wav", tmp_path, "a", mp3=False, bitrate_kbps=128,
            on_failure=lambda _s, _e: None,
        )


@pytest.fixture
def fake_cli(tmp_path, monkeypatch, fake_audio):
    """cli._run with fake sessions: split DML fails, split unfused fails, CPU works."""
    model = tmp_path / "htdemucs_6s_fp16weights.onnx"
    model.write_bytes(b"x" * 10)
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setenv("DEMUCS_ONNX_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv(dml_state.ENV, raising=False)
    monkeypatch.setattr(cli, "find_model", lambda _d, _w: model)
    monkeypatch.setattr(cli, "resolve_providers", lambda _p: list(DML))
    monkeypatch.setattr(cli, "gpu_name", lambda d: GPU if d == "dml" else None)
    opened: list[str] = []

    def fake_open(path, providers, *, dml_fusion=True, strict=False):
        prefix = "split_" if path.name.startswith(("head", "tail")) else ""
        if providers == separate.CPU_ONLY:
            opened.append(prefix + "cpu")
            return FakeSession(separate.CPU_ONLY, fail=False, label="cpu"), "cpu"
        label = prefix + ("fused" if dml_fusion else "unfused")
        opened.append(label)
        return FakeSession(DML, fail=True, label=label), "dml"

    from demucs_onnx_worker import split_model

    monkeypatch.setattr(
        split_model, "ensure_split", lambda _m, d: (d / "head.onnx", d / "tail.onnx")
    )
    monkeypatch.setattr(separate, "open_session", fake_open)
    argv = [str(wav), "-o", str(tmp_path / "out"), "--track-name", "song", "--progress-json"]
    return argv, opened


def _events(out: str) -> list[dict]:
    return [json.loads(ln) for ln in out.splitlines() if ln.startswith("{")]


def test_dml_ladder_then_remembered(fake_cli, capsys) -> None:
    argv, opened = fake_cli
    try:
        assert cli.main(argv) == 0
    finally:
        events.set_enabled(False)
    assert opened == ["split_fused", "split_cpu", "split_unfused", "split_cpu", "cpu"]
    captured = capsys.readouterr()
    stages = [e.get("stage") for e in _events(captured.out) if e["event"] == "status"]
    assert stages == ["loading_model", "model_ready", "retry_gpu", "fallback_cpu"]
    fallback = [e for e in _events(captured.out) if e.get("stage") == "fallback_cpu"][0]
    assert "80070057" in fallback["reason"]
    assert "DmlFusedNode" in captured.err
    state = json.loads(dml_state.state_path().read_text(encoding="utf-8"))
    assert [v["mode"] for v in state.values()] == ["cpu"]

    opened.clear()
    try:
        assert cli.main(argv) == 0
    finally:
        events.set_enabled(False)
    assert opened == ["cpu"]
    remembered = [e for e in _events(capsys.readouterr().out) if e.get("stage") == "fallback_cpu"]
    assert remembered and "earlier run" in remembered[0]["reason"]


def test_retry_env_ignores_remembered(fake_cli, monkeypatch, capsys) -> None:
    argv, opened = fake_cli
    key = dml_state.state_key(ort.__version__, GPU, cli.find_model(None, None))
    dml_state.remember(key, "cpu", "old")
    monkeypatch.setenv(dml_state.ENV, "retry")
    try:
        assert cli.main(argv) == 0
    finally:
        events.set_enabled(False)
    assert opened[0] == "split_fused"


def test_device_gpu_errors_instead_of_cpu(fake_cli, capsys) -> None:
    argv, opened = fake_cli
    try:
        assert cli.main([*argv, "--device", "gpu"]) == 1
    finally:
        events.set_enabled(False)
    assert opened == ["split_fused", "split_cpu", "split_unfused", "split_cpu"]
    evs = _events(capsys.readouterr().out)
    assert "fallback_cpu" not in [e.get("stage") for e in evs]
    err = [e for e in evs if e["event"] == "error"][0]
    assert err["kind"] == "gpu_failed"
    assert err["message"].startswith(f"GPU inference failed (DirectML on {GPU}). Switch")
    assert "80070057" in err["message"]
    state = json.loads(dml_state.state_path().read_text(encoding="utf-8"))
    assert [v["mode"] for v in state.values()] == ["split_unfused"]


def test_device_gpu_ignores_remembered_cpu(fake_cli, capsys) -> None:
    argv, opened = fake_cli
    key = dml_state.state_key(ort.__version__, GPU, cli.find_model(None, None))
    dml_state.remember(key, "cpu", "old")
    try:
        assert cli.main([*argv, "--device", "gpu"]) == 1
    finally:
        events.set_enabled(False)
    assert opened[0] == "split_fused"
    assert "cpu" not in opened


def test_device_gpu_without_gpu_provider(fake_cli, monkeypatch, capsys) -> None:
    argv, opened = fake_cli
    monkeypatch.setattr(cli, "resolve_providers", lambda _p: list(separate.CPU_ONLY))
    try:
        assert cli.main([*argv, "--device", "gpu"]) == 1
    finally:
        events.set_enabled(False)
    assert opened == []
    err = [e for e in _events(capsys.readouterr().out) if e["event"] == "error"][0]
    assert err["kind"] == "gpu_failed" and "No GPU" in err["message"]


def test_device_cpu_skips_gpu(fake_cli, capsys) -> None:
    argv, opened = fake_cli
    try:
        assert cli.main([*argv, "--device", "cpu"]) == 0
    finally:
        events.set_enabled(False)
    assert opened == ["cpu"]
    assert not dml_state.state_path().exists()
    ready = [e for e in _events(capsys.readouterr().out) if e.get("stage") == "model_ready"][0]
    assert ready["device"] == "cpu"


def test_env_forces_whole_graph_mode(fake_cli, monkeypatch) -> None:
    argv, opened = fake_cli
    monkeypatch.setenv(dml_state.ENV, "fused")
    try:
        assert cli.main(argv) == 0
    finally:
        events.set_enabled(False)
    assert opened == ["fused", "unfused", "cpu"]


def test_split_failure_runs_whole_graph(fake_cli, monkeypatch) -> None:
    argv, opened = fake_cli
    from demucs_onnx_worker import split_model

    def broken(_m, _d):
        raise ImportError("No module named 'onnx'")

    monkeypatch.setattr(split_model, "ensure_split", broken)
    try:
        assert cli.main(argv) == 0
    finally:
        events.set_enabled(False)
    assert opened == ["fused", "unfused", "cpu"]


def test_old_state_key_is_ignored(tmp_path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    assert dml_state.state_key("1.24.4", GPU, model).startswith(f"v{dml_state.KEY_VERSION}|")


def test_remembered_mode_expires(tmp_path) -> None:
    path = tmp_path / "s.json"
    dml_state.remember("k", "split_unfused", "r", path=path)
    assert dml_state.start_mode("k", path=path) == ("split_unfused", "remembered")
    later = json.loads(path.read_text())["k"]["at"] + dml_state.RETRY_AFTER_S + 1
    assert dml_state.start_mode("k", path=path, now=later) == ("split", "default")
    assert dml_state.next_mode("split") == "split_unfused"
    assert dml_state.next_mode("split_unfused") == "cpu"
    assert dml_state.next_mode("fused") == "unfused"
    assert dml_state.next_mode("cpu") == "cpu"
