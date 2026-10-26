from demucs_onnx_worker.chunking import chunk_count, chunk_span, transition_window
from demucs_onnx_worker.device import device_name


def test_device_prefers_gpu_and_fits_gui_field() -> None:
    assert device_name(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == "coreml"
    assert device_name(["DmlExecutionProvider", "CPUExecutionProvider"]) == "dml"
    assert device_name(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda"
    assert device_name(["CPUExecutionProvider"]) == "cpu"
    assert len(device_name(["SomeVeryLongCustomExecutionProvider"])) <= 15


def test_quarter_overlap_matches_demucs_onnx_segment() -> None:
    from demucs_onnx.inference import N_SAMPLES, SAMPLE_RATE

    assert SAMPLE_RATE == 44100
    assert N_SAMPLES == 343980
    window = transition_window(N_SAMPLES, 0.25)
    assert window.shape == (N_SAMPLES,)
    assert float(window[0]) == 0.0
    assert float(window[-1]) == 0.0
    assert float(window[N_SAMPLES // 2]) == 1.0

    twenty_seconds = 20 * SAMPLE_RATE
    assert chunk_count(twenty_seconds, N_SAMPLES, 0.25) == 4
    start, end = chunk_span(0, twenty_seconds, N_SAMPLES, 0.25)
    assert (start, end) == (0, N_SAMPLES)
