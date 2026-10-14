"""Tests for tools module."""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    apply_patch,
    execute_tool,
    get_file_metadata,
    list_files,
    propose_patch,
    read_file,
    search_code,
    validate_path,
)


@pytest.fixture
def workspace(tmp_path):
    """Fixture providing a temporary workspace with sample code files."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    # Create subfolder and files
    src_dir = ws / "src"
    src_dir.mkdir()

    hello_py = src_dir / "hello.py"
    hello_py.write_text(
        "import os\n"
        "from typing import List\n\n"
        "class Greeter:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        "    def greet(self) -> str:\n"
        "        return f'Hello, {self.name}'\n\n"
        "def main():\n"
        "    greeter = Greeter('World')\n"
        "    print(greeter.greet())\n",
        encoding="utf-8"
    )

    notes_md = ws / "notes.md"
    notes_md.write_text(
        "# Project Notes\n\n"
        "This is a sample project for testing CodePilot.\n"
        "Contains function definitions and classes.\n",
        encoding="utf-8"
    )

    bin_file = ws / "test.png"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    return ws


# ── 1. validate_path ─────────────────────────────────────

def test_validate_path_valid(workspace):
    """Test validate_path with valid relative path."""
    valid, msg, resolved = validate_path("src/hello.py", str(workspace))
    assert valid is True
    assert resolved is not None
    assert resolved.exists()
    assert resolved.name == "hello.py"


def test_validate_path_traversal(workspace):
    """Test validate_path rejects path traversal attempts."""
    valid, msg, resolved = validate_path("../../etc/passwd", str(workspace))
    assert valid is False
    assert "traversal" in msg.lower() or "suspicious" in msg.lower()
    assert resolved is None


def test_validate_path_rejects_sibling_sharing_the_name_prefix(workspace):
    """Regression: a string startswith() check let '../workspace-evil/x.py' through,
    because '...\\workspace-evil' shares the '...\\workspace' prefix but is outside it."""
    evil = workspace.parent / (workspace.name + "-evil")
    evil.mkdir()
    (evil / "x.py").write_text("print('pwned')", encoding="utf-8")

    valid, msg, resolved = validate_path(f"../{evil.name}/x.py", str(workspace))
    assert valid is False
    assert resolved is None
    assert "traversal" in msg.lower()


def test_validate_path_allows_dotdot_that_stays_inside(workspace):
    """'..' is only a problem when it escapes: src/../notes.md is still in the root."""
    valid, _msg, resolved = validate_path("src/../notes.md", str(workspace))
    assert valid is True
    assert resolved is not None and resolved.name == "notes.md"


def test_validate_path_empty(workspace):
    """Test validate_path rejects empty paths."""
    valid, msg, resolved = validate_path("", str(workspace))
    assert valid is False
    assert "empty" in msg.lower()

    valid, msg, resolved = validate_path("src/hello.py", "")
    assert valid is False
    assert "empty" in msg.lower()


# ── 2. list_files ────────────────────────────────────────

def test_list_files_success(workspace):
    """Test listing files in workspace directory."""
    result = list_files(str(workspace), directory=".")
    assert result["success"] is True
    file_names = [f["name"] for f in result["files"]]
    assert "hello.py" in file_names
    assert "notes.md" in file_names


def test_list_files_extension_filtering(workspace):
    """Test filtering by file extensions."""
    result = list_files(str(workspace), directory=".", extensions=[".py"])
    assert result["success"] is True
    assert all(f["extension"] == ".py" for f in result["files"])
    assert any(f["name"] == "hello.py" for f in result["files"])
    assert not any(f["name"] == "notes.md" for f in result["files"])


@pytest.mark.parametrize("directory", ["", None, "/", "./", "  "])
def test_list_files_treats_empty_directory_as_workspace_root(workspace, directory):
    """Regression: models send "" / null for "everything"; that used to fail with 'Empty path or workspace root'."""
    result = list_files(str(workspace), directory=directory)
    assert result["success"] is True and result["files"]


def test_execute_tool_list_files_with_empty_or_null_directory(workspace):
    for args in ({"directory": ""}, {"directory": None}, {}):
        assert execute_tool("list_files", args, str(workspace))["success"] is True


def test_list_files_accepts_extension_as_plain_string(workspace):
    result = list_files(str(workspace), directory=".", extensions=".py")
    assert result["success"] is True and all(f["extension"] == ".py" for f in result["files"])


def test_list_files_nonexistent_directory(workspace):
    """Test list_files with nonexistent directory."""
    result = list_files(str(workspace), directory="nonexistent_folder")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ── 3. read_file ─────────────────────────────────────────

def test_read_file_success(workspace):
    """Test reading file contents with line numbering."""
    result = read_file(str(workspace), "src/hello.py")
    assert result["success"] is True
    assert result["total_lines"] > 0
    assert "class Greeter:" in result["content"]
    assert "1 |" in result["content"]


def test_read_file_line_range(workspace):
    """Test reading specific line range."""
    result = read_file(str(workspace), "src/hello.py", start_line=4, end_line=6)
    assert result["success"] is True
    assert result["shown_lines"] == (4, 6)
    assert "class Greeter:" in result["content"]
    assert "return f'Hello" not in result["content"]


def test_read_file_binary_rejection(workspace):
    """Test rejecting reading of binary files."""
    result = read_file(str(workspace), "test.png")
    assert result["success"] is False
    assert "binary" in result["error"].lower()


def test_read_file_nonexistent(workspace):
    """Test reading nonexistent file."""
    result = read_file(str(workspace), "does_not_exist.py")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ── 4. search_code ───────────────────────────────────────

def test_search_code_matches(workspace):
    """Test searching for patterns across workspace files."""
    result = search_code(str(workspace), "Greeter")
    assert result["success"] is True
    assert len(result["matches"]) > 0
    match_files = [m["file"] for m in result["matches"]]
    assert any("hello.py" in f for f in match_files)


def test_search_code_empty_pattern(workspace):
    """Test search_code with empty pattern."""
    result = search_code(str(workspace), "")
    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_search_code_case_sensitivity(workspace):
    """Test case-sensitive search behavior."""
    insensitive_result = search_code(str(workspace), "greeter", case_sensitive=False)
    assert insensitive_result["success"] is True
    assert len(insensitive_result["matches"]) > 0

    sensitive_result = search_code(str(workspace), "GREETER_ALL_CAPS_NONE", case_sensitive=True)
    assert sensitive_result["success"] is True
    assert len(sensitive_result["matches"]) == 0


# ── 5. get_file_metadata ─────────────────────────────────

def test_get_file_metadata_python_file(workspace):
    """Test extracting metadata for a Python file with functions and classes."""
    result = get_file_metadata(str(workspace), "src/hello.py")
    assert result["success"] is True
    meta = result["metadata"]
    assert meta["name"] == "hello.py"
    assert meta["extension"] == ".py"
    assert meta["total_lines"] > 0

    class_names = [c["name"] for c in meta["classes"]]
    assert "Greeter" in class_names

    func_names = [f["name"] for f in meta["functions"]]
    assert "__init__" in func_names
    assert "greet" in func_names
    assert "main" in func_names

    assert any("os" in imp for imp in meta["imports"])


# ── 6. propose_patch ─────────────────────────────────────

def test_propose_patch(workspace):
    """Test generating unified diff and counting additions/deletions."""
    original = "line 1\nline 2\nline 3\n"
    modified = "line 1\nline 2 modified\nline 3\nline 4 added\n"

    result = propose_patch(str(workspace), "src/hello.py", original, modified)
    assert result["success"] is True
    assert result["filepath"] == "src/hello.py"
    assert "--- a/src/hello.py" in result["diff"]
    assert "+++ b/src/hello.py" in result["diff"]
    assert result["stats"]["additions"] >= 2
    assert result["stats"]["deletions"] >= 1


# ── 7. apply_patch ───────────────────────────────────────

def test_apply_patch_rejected_when_not_approved(workspace):
    """Test apply_patch rejection when approved=False."""
    result = apply_patch(str(workspace), "src/hello.py", "new content", approved=False)
    assert result["success"] is False
    assert "not approved" in result["error"].lower()
    assert result["backup_path"] is None


def test_apply_patch_success_when_approved(workspace):
    """Test apply_patch creates backup and applies changes when approved=True."""
    target_file = workspace / "src" / "hello.py"
    original_text = target_file.read_text(encoding="utf-8")
    new_text = "# Modified content\nprint('patched!')\n"

    result = apply_patch(str(workspace), "src/hello.py", new_text, approved=True)
    assert result["success"] is True
    assert result["backup_path"] is not None
    assert Path(result["backup_path"]).exists()
    assert Path(result["backup_path"]).read_text(encoding="utf-8") == original_text
    assert target_file.read_text(encoding="utf-8") == new_text


# ── 8. execute_tool ──────────────────────────────────────

def test_execute_tool_unknown(workspace):
    """Test execute_tool handles unknown tool names."""
    result = execute_tool("nonexistent_tool", {}, str(workspace))
    assert result["success"] is False
    assert "unknown tool" in result["error"].lower()


def test_execute_tool_valid(workspace):
    """Test execute_tool dispatches and returns expected valid tool result."""
    result = execute_tool("list_files", {"directory": "."}, str(workspace))
    assert result["success"] is True
    assert "files" in result
    assert isinstance(result["files"], list)
