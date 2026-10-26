from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

pytest.importorskip("onnx")

from demucs_onnx_worker import split_model  # noqa: E402
from demucs_onnx_worker.separate import CPU_ONLY, N_CHANNELS, N_SAMPLES  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "models" / "htdemucs_6s_fp16weights.onnx"


@pytest.mark.skipif(not MODEL.is_file(), reason="model not downloaded")
def test_split_matches_full_graph(tmp_path) -> None:
    head, tail = split_model.ensure_split(MODEL, tmp_path)
    assert split_model.ensure_split(MODEL, tmp_path) == (head, tail)
    session = split_model.SplitSession(
        ort.InferenceSession(str(head), providers=CPU_ONLY),
        ort.InferenceSession(str(tail), providers=CPU_ONLY),
    )
    full = ort.InferenceSession(str(MODEL), providers=CPU_ONLY)
    rng = np.random.default_rng(0)
    mix = (rng.standard_normal((1, N_CHANNELS, N_SAMPLES)) * 0.1).astype(np.float32)
    want = full.run(["stems"], {"mix": mix})[0]
    got = session.run(["stems"], {"mix": mix})[0]
    assert session.get_providers() == CPU_ONLY
    np.testing.assert_allclose(got, want, atol=1e-5)
