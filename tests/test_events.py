from __future__ import annotations

import json

from demucs_onnx_worker import events


def test_separating_line_matches_gui_contract(capsys) -> None:
    events.set_enabled(True)
    try:
        events.emit("separating", track="song.wav", done=1, total=4, pct=25.0)
        events.emit(
            "status",
            stage="model_ready",
            model="htdemucs_6s",
            device="cpu",
            sources=["drums", "bass", "other", "vocals", "guitar", "piano"],
        )
        events.emit("error", message='bad "wav"')
    finally:
        events.set_enabled(False)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 3
    for line in lines:
        assert '"event"' in line
        assert events.validate(json.loads(line)) == []
    assert json.loads(lines[0])["pct"] == 25.0
    assert json.loads(lines[1])["device"] == "cpu"
    assert '\\"' in lines[2]
