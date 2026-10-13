"""Tests for the compiler module and the compile_check tool."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compiler import check_python, render_report, run_python
from tools import compile_check, execute_tool, propose_patch


def test_clean_code_compiles():
    res = check_python("def add(a, b):\n    return a + b\n")
    assert res["ok"] and res["errors"] == 0 and res["diagnostics"] == []


def test_syntax_error_reports_line_and_column():
    res = check_python('x = 1\nprint("hi"\n')
    assert not res["ok"]
    d = res["diagnostics"][0]
    assert d["severity"] == "error" and d["kind"] == "SyntaxError"
    assert d["line"] in (2, 3)           # exact line differs slightly between Python versions
    assert "never closed" in d["message"] or "expected" in d["message"].lower()


def test_indentation_error_detected():
    res = check_python("def f():\nreturn 1\n")
    assert not res["ok"]
    assert res["diagnostics"][0]["kind"] == "IndentationError"


def test_undefined_name_is_error_and_unused_import_is_warning():
    res = check_python("import os\nprint(missing_name)\n")
    if not res["linter"]:                # pyflakes not installed
        return
    severities = {d["kind"]: d["severity"] for d in res["diagnostics"]}
    assert severities["UndefinedName"] == "error"
    assert severities["UnusedImport"] == "warning"
    assert res["errors"] == 1 and res["warnings"] == 1


def test_warnings_alone_do_not_fail_the_build():
    res = check_python("import os\n")
    if res["linter"]:
        assert res["ok"] and res["warnings"] == 1


def test_render_report_has_caret():
    res = check_python("x = (1 + \n")
    report = render_report(res["diagnostics"], "bad.py")
    assert report.startswith("bad.py:") and "error" in report and "^" in report


def test_run_python_captures_traceback():
    res = run_python("a = 1\nb = 0\nprint(a / b)\n", timeout=10, use_docker=False)
    assert not res["ok"]
    d = res["diagnostics"][0]
    assert d["kind"] == "ZeroDivisionError" and d["line"] == 3


def test_run_python_success_and_stdout():
    res = run_python("print('hello')\n", timeout=10, use_docker=False)
    assert res["ok"] and res["stdout"].strip() == "hello"


def test_run_python_timeout():
    res = run_python("while True:\n    pass\n", timeout=1, use_docker=False)
    assert res["timed_out"] and not res["ok"]


def test_run_python_does_not_leak_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    res = run_python("import os\nprint(os.environ.get('GROQ_API_KEY'))\n", use_docker=False)
    assert res["stdout"].strip() == "None"


def test_compile_check_tool(tmp_path):
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("def f(:\n    pass\n")
    assert compile_check(str(tmp_path), "good.py")["compiles"] is True
    bad = compile_check(str(tmp_path), "bad.py")
    assert bad["success"] and bad["compiles"] is False and "bad.py:1" in bad["report"]
    assert not compile_check(str(tmp_path), "../escape.py")["success"]
    assert not compile_check(str(tmp_path), "notes.txt")["success"]
    assert execute_tool("compile_check", {"filepath": "bad.py"}, str(tmp_path))["compiles"] is False


def test_propose_patch_flags_broken_syntax(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    ok = propose_patch(str(tmp_path), "m.py", old_text="return 1", new_text="return 2")
    assert ok["syntax"]["ok"] is True
    broken = propose_patch(str(tmp_path), "m.py", old_text="return 1", new_text="return (1")
    assert broken["syntax"]["ok"] is False and broken["syntax"]["message"]
