"""Make ONNX Runtime errors readable in a GUI console.

DirectML failures carry CR/LF, sometimes ESC or localized (non-ASCII) Windows
text, and the HRESULT that explains them sits at the end. Printed raw, a CR
rewinds the console line and hides everything before it.
"""

from __future__ import annotations

import re

# HRESULTs DirectML / D3D12 report from a failed DmlFusedNode.
HRESULTS: dict[str, str] = {
    "887A0005": "DXGI_ERROR_DEVICE_REMOVED: the GPU was reset or removed (driver crash or TDR)",
    "887A0006": "DXGI_ERROR_DEVICE_HUNG: the GPU stopped responding (TDR timeout or bad command)",
    "887A0007": "DXGI_ERROR_DEVICE_RESET: the GPU was reset by another app or the driver",
    "887A0020": "DXGI_ERROR_DRIVER_INTERNAL_ERROR: the display driver failed",
    "80070057": "E_INVALIDARG: DirectML rejected an operator or tensor shape in the model",
    "8007000E": "E_OUTOFMEMORY: not enough GPU memory",
    "887E0003": "DML: the graph uses an unsupported operator or data type",
    "80004005": "E_FAIL: unspecified DirectML failure",
}

_HR = re.compile(r"(?<![0-9A-Fa-f])(?:0x)?([0-9A-Fa-f]{8})(?![0-9A-Fa-f])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def one_line(text: str) -> str:
    """Single printable ASCII line: whitespace controls become spaces, other
    control and non-ASCII characters become backslash escapes."""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    return re.sub(r" {2,}", " ", text).strip()


def printable_lines(text: str) -> list[str]:
    """Split on newlines and make each line safe to print (tracebacks)."""
    out = []
    for ln in re.split(r"\r\n|\r|\n", text.rstrip()):
        ln = _CONTROL.sub(lambda m: " " if m.group() == "\t" else f"\\x{ord(m.group()):02x}", ln)
        out.append(ln.encode("ascii", "backslashreplace").decode("ascii").rstrip())
    return out


def hresult(text: str) -> tuple[str, str] | None:
    """First known HRESULT in ``text`` as ``(code, meaning)``; else the last
    8-hex-digit code that looks like a failure (high bit set), meaning ``""``."""
    codes = [m.group(1).upper() for m in _HR.finditer(text)]
    for code in codes:
        if code in HRESULTS:
            return code, HRESULTS[code]
    failures = [c for c in codes if c[0] in "89ABCDEF" and not c.startswith("0000")]
    return (failures[-1], "") if failures else None


CPU_HINT = "Switch the stems device to CPU in the app and run again."


class GpuFailed(RuntimeError):
    """``--device gpu`` could not run on the GPU. The message already leads with
    the advice and carries the described cause."""


def gpu_failed_message(providers: list[str], stage: str, cause: BaseException,
                       gpu: str | None = None) -> str:
    where = providers[0].replace("ExecutionProvider", "")
    where = {"Dml": "DirectML"}.get(where, where)
    if gpu:
        where += f" on {gpu}"
    return f"GPU {stage} failed ({where}). {CPU_HINT} Cause: {describe(cause)}"


def describe(exc: BaseException) -> str:
    """One printable line, HRESULT first so a truncating GUI still shows it."""
    text = one_line(str(exc)) or exc.__class__.__name__
    if isinstance(exc, GpuFailed):
        return text
    hr = hresult(text)
    if hr is None:
        return text
    code, meaning = hr
    return f"HRESULT {code}{' (' + meaning + ')' if meaning else ''}: {text}"
