from __future__ import annotations

from demucs_onnx_worker import errors

DML_RAW = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running "
    "DmlFusedNode_0_0 node. Name:'DmlFusedNode_0_0' Status Message: "
    "E:\\_work\\1\\s\\onnxruntime\\core\\providers\\dml\\DmlExecutionProvider\\src\\"
    "DmlGraphFusionHelper.cpp(1041)\\onnxruntime_pybind11_state.pyd!00007FFAA11FC07A:"
    " (caller: 00007FFAA11FB2B9) Exception(3) tid(5b4c) 887A0006 \x1b[0m"
    "Le GPU ne r\u00e9pond plus\r\n"
)


def test_one_line_is_printable_ascii() -> None:
    line = errors.one_line(DML_RAW)
    assert "\r" not in line and "\n" not in line and "\x1b" not in line
    assert line.isascii()
    assert "\\x1b" in line
    assert "r\\xe9pond" in line
    assert line.startswith("[ONNXRuntimeError]")


def test_hresult_found_and_named() -> None:
    code, meaning = errors.hresult(errors.one_line(DML_RAW))
    assert code == "887A0006"
    assert "DEVICE_HUNG" in meaning


def test_pointer_addresses_are_not_hresults() -> None:
    assert errors.hresult("pyd!00007FFAA11FC07A: (caller: 00007FFAA11FB2B9) tid(5b4c)") is None


def test_unknown_failure_code_is_reported() -> None:
    assert errors.hresult("failed with 887B0001") == ("887B0001", "")


def test_describe_puts_hresult_first() -> None:
    text = errors.describe(RuntimeError(DML_RAW))
    assert text.startswith("HRESULT 887A0006 (DXGI_ERROR_DEVICE_HUNG")
    assert "DmlFusedNode_0_0" in text


def test_printable_lines_keep_indentation() -> None:
    lines = errors.printable_lines('Traceback:\n  File "x.py"\r\nboom\x1b\n')
    assert lines == ["Traceback:", '  File "x.py"', "boom\\x1b"]
