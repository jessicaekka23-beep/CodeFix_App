"""
CodePilot Local - Compiler / error-checking module.

Python has no separate "compile" step, so this module does what a compiler front-end
does and reports problems with their locations:

  1. check_python()  - byte-compiles the source (never executes it) to find syntax errors,
                       then runs pyflakes (if installed) for undefined names, unused imports, ...
  2. run_python()    - optional: runs the code in a throw-away directory (Docker when
                       available) with a timeout, to surface runtime tracebacks.
  3. render_report() - turns diagnostics into compiler-style text with a caret under
                       the offending column. The same text is fed to the agent.

Every diagnostic is a dict: {severity, kind, line, col, end_line, end_col, message, text}.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

try:                                    # optional: richer static analysis
    from pyflakes.checker import Checker as _PyflakesChecker
except ImportError:                     # pragma: no cover - exercised only without pyflakes
    _PyflakesChecker = None

MAX_OUTPUT_CHARS = 4000

# pyflakes message classes that are worth flagging but do not stop the program from running
_WARNING_PREFIXES = ("Unused", "Redefined", "ImportShadowed", "ImportStar", "Imports")


def _diag(severity, kind, line, col, message, text="", end_line=None, end_col=None) -> dict:
    return {"severity": severity, "kind": kind, "line": line, "col": col,
            "end_line": end_line or line, "end_col": end_col,
            "message": message, "text": text}


# ── 1. Static check ──────────────────────────────────────

def check_python(source: str, filename: str = "<snippet>") -> dict:
    """Compile `source` without executing it.

    Returns:
        {"ok": bool, "errors": int, "warnings": int, "diagnostics": [...], "linter": bool}
        `ok` is True when there are no errors (warnings alone do not fail the build).
    """
    diagnostics = []
    tree = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            compile(source, filename, "exec", dont_inherit=True)
        except SyntaxError as exc:
            diagnostics.append(_diag(
                "error", type(exc).__name__, exc.lineno or 1, exc.offset,
                exc.msg or str(exc), (exc.text or "").rstrip("\n"),
                getattr(exc, "end_lineno", None), getattr(exc, "end_offset", None)))
        except (ValueError, MemoryError, RecursionError) as exc:    # e.g. null bytes
            diagnostics.append(_diag("error", type(exc).__name__, 1, None, str(exc)))
    for w in caught:                                                # e.g. invalid escape sequence
        if issubclass(w.category, SyntaxWarning):
            diagnostics.append(_diag("warning", "SyntaxWarning", w.lineno or 1, None, str(w.message)))

    has_syntax_error = any(d["severity"] == "error" for d in diagnostics)
    if not has_syntax_error and _PyflakesChecker is not None:
        import ast
        try:
            tree = ast.parse(source, filename)
            for msg in sorted(_PyflakesChecker(tree, filename).messages,
                              key=lambda m: (m.lineno, m.col)):
                kind = type(msg).__name__
                severity = "warning" if kind.startswith(_WARNING_PREFIXES) else "error"
                diagnostics.append(_diag(severity, kind, msg.lineno, msg.col + 1,
                                         msg.message % msg.message_args))
        except Exception:               # the linter must never break the compile step
            pass

    lines = source.splitlines()
    for d in diagnostics:               # attach the source line when the exception did not
        if not d["text"] and 1 <= d["line"] <= len(lines):
            d["text"] = lines[d["line"] - 1]

    errors = sum(1 for d in diagnostics if d["severity"] == "error")
    return {"ok": errors == 0, "errors": errors,
            "warnings": len(diagnostics) - errors,
            "diagnostics": diagnostics, "linter": _PyflakesChecker is not None}


# ── 2. Run (optional) ────────────────────────────────────

_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?')


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else "...(truncated)\n" + text[-MAX_OUTPUT_CHARS:]


def _parse_traceback(stderr: str, script_name: str):
    """Extract the runtime error from a traceback: (kind, message, line) or None."""
    if "Traceback (most recent call last)" not in stderr:
        return None
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    kind, _, message = last.partition(":")
    line_no = None
    for m in _FRAME_RE.finditer(stderr):            # deepest frame that belongs to the user's file
        if os.path.basename(m.group("file")) == script_name:
            line_no = int(m.group("line"))
    return kind.strip() or "RuntimeError", message.strip(), line_no


def _docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=8).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _clean_env() -> dict:
    """Pass only what Python needs, so pasted code cannot read API keys from the environment."""
    keep = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TEMP", "TMP")
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_python(source: str, timeout: int = 10, use_docker: bool = True,
               script_name: str = "main.py") -> dict:
    """Run `source` in a disposable directory and capture the outcome.

    Docker (no network, 256 MB, read-only mount) is used when available; otherwise a plain
    subprocess in a temp dir with a timeout. The subprocess mode is NOT a security sandbox.
    """
    tmp = tempfile.mkdtemp(prefix="codepilot_run_")
    started = time.time()
    try:
        os.chmod(tmp, 0o755)
        path = os.path.join(tmp, script_name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(source)
        os.chmod(path, 0o644)

        if use_docker and _docker_ready():
            mode = "docker"
            cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "256m", "--cpus", "1.0",
                   "--read-only", "--tmpfs", "/tmp:rw,noexec,size=50m", "-v", f"{tmp}:/work:ro",
                   "-w", "/work", "--user", "nobody", "python:3.12-slim",
                   "python", "-I", script_name]
        else:
            mode = "subprocess (limited isolation)"
            cmd = [sys.executable, "-I", script_name]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp,
                                  stdin=subprocess.DEVNULL, env=_clean_env(),
                                  encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            return {"ok": False, "timed_out": True, "returncode": None, "mode": mode,
                    "stdout": _clip(out), "stderr": "", "diagnostics": [
                        _diag("error", "Timeout", 1, None,
                              f"Execution exceeded {timeout}s (infinite loop or blocking call?)")],
                    "duration": round(time.time() - started, 2)}

        diagnostics = []
        parsed = _parse_traceback(proc.stderr, script_name)
        if parsed:
            kind, message, line_no = parsed
            diagnostics.append(_diag("error", kind, line_no or 1, None, message,
                                     source.splitlines()[line_no - 1] if line_no and line_no <= len(source.splitlines()) else ""))
        elif proc.returncode != 0:
            diagnostics.append(_diag("error", "ExitCode", 1, None, f"Process exited with code {proc.returncode}"))
        return {"ok": proc.returncode == 0, "timed_out": False, "returncode": proc.returncode,
                "mode": mode, "stdout": _clip(proc.stdout), "stderr": _clip(proc.stderr),
                "diagnostics": diagnostics, "duration": round(time.time() - started, 2)}
    except Exception as exc:
        return {"ok": False, "timed_out": False, "returncode": None, "mode": "none",
                "stdout": "", "stderr": "", "diagnostics": [_diag("error", "RunnerError", 1, None, str(exc))],
                "duration": round(time.time() - started, 2)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── 3. Report ────────────────────────────────────────────

def render_report(diagnostics: list, filename: str = "<snippet>") -> str:
    """Compiler-style text, e.g.

        main.py:3:11: error: SyntaxError: '(' was never closed
            3 | print("hi"
              |       ^
    """
    if not diagnostics:
        return f"{filename}: no problems found"
    out = []
    for d in diagnostics:
        col = f":{d['col']}" if d.get("col") else ""
        out.append(f"{filename}:{d['line']}{col}: {d['severity']}: {d['kind']}: {d['message']}")
        text = d.get("text", "")
        if text:
            out.append(f"  {d['line']:>4} | {text}")
            if d.get("col") and d["col"] >= 1:
                width = 1
                if d.get("end_col") and d.get("end_line") == d["line"] and d["end_col"] > d["col"]:
                    width = d["end_col"] - d["col"]
                out.append(f"  {'':>4} | {' ' * (d['col'] - 1)}{'^' * width}")
    return "\n".join(out)
