import pytest

from demucs_onnx_worker.cli import VERSION, build_parser, main


def test_version_matches_package() -> None:
    assert VERSION == "0.1.1"


def test_gui_argv_parses() -> None:
    args = build_parser().parse_args(
        [
            "song.wav",
            "-n",
            "htdemucs_6s",
            "-o",
            "out",
            "--track-name",
            "song",
            "-b",
            "8",
            "--progress-json",
            "--prefetch-tracks",
            "0",
            "--mp3",
        ]
    )
    assert args.name == "htdemucs_6s"
    assert args.out == "out"
    assert args.track_name == "song"
    assert args.batch_size == 8
    assert args.progress_json is True
    assert args.prefetch_tracks == 0
    assert args.mp3 is True
    assert args.mp3_bitrate == 128
    assert args.providers == "auto"
    assert args.overlap == 0.25


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(SystemExit):
        from demucs_onnx_worker.cli import _validate

        args = build_parser().parse_args(["song.wav", "-n", "htdemucs_ft"])
        _validate(args)


def test_list_models(capsys) -> None:
    assert main(["--list-models"]) == 0
    assert capsys.readouterr().out.strip() == "htdemucs_6s"


def test_missing_model_emits_error_json(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "demucs_onnx_worker.cli.executable_dir",
        lambda _argv0: tmp_path / "exe",
    )
    for key in (
        "DEMUCS_ONNX_CACHE",
        "DEMUCS_MLX_CACHE",
        "DEMUCS_CUDA_CACHE",
        "DEMUCS_TORCH_CACHE",
        "DEMUCS_ONNX_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"not audio")
    code = main(
        [
            str(wav),
            "-o",
            str(tmp_path / "out"),
            "--track-name",
            "song",
            "--progress-json",
            "--providers",
            "cpu",
        ]
    )
    assert code == 1
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"event"' in line
    assert "error" in line
