from pathlib import Path

from demucs_onnx_worker.model_path import find_model


def test_weights_flag_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.onnx"
    explicit.write_bytes(b"not a real model")
    assert find_model(tmp_path, str(explicit)) == explicit


def test_fp16_file_in_cache_dir(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    model = cache / "htdemucs_6s_fp16weights.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.setenv("DEMUCS_ONNX_CACHE", str(cache))
    monkeypatch.delenv("DEMUCS_MLX_CACHE", raising=False)
    monkeypatch.delenv("DEMUCS_CUDA_CACHE", raising=False)
    monkeypatch.delenv("DEMUCS_TORCH_CACHE", raising=False)
    monkeypatch.delenv("DEMUCS_ONNX_MODEL", raising=False)
    assert find_model(tmp_path / "exe") == model


def test_mlx_cache_env_used_by_the_gui(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "gui-models"
    cache.mkdir()
    model = cache / "htdemucs_6s.onnx"
    model.write_bytes(b"onnx")
    monkeypatch.delenv("DEMUCS_ONNX_CACHE", raising=False)
    monkeypatch.setenv("DEMUCS_MLX_CACHE", str(cache))
    monkeypatch.delenv("DEMUCS_ONNX_MODEL", raising=False)
    assert find_model(tmp_path / "missing") == model


def test_missing_model_raises(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "DEMUCS_ONNX_CACHE",
        "DEMUCS_MLX_CACHE",
        "DEMUCS_CUDA_CACHE",
        "DEMUCS_TORCH_CACHE",
        "DEMUCS_ONNX_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        find_model(tmp_path / "exe")
    except FileNotFoundError as exc:
        assert "htdemucs_6s" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
