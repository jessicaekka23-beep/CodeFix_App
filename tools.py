"""
CodePilot Local — Tools Module

Provides validated workspace tools for the agentic loop:
- list_files: List files in the workspace
- read_file: Read file contents with line numbers
- search_code: Search for patterns in code files
- search_knowledge_base: Query the RAG vector store
- get_file_metadata: Get file size, modified time, etc.
- run_tests: Execute tests in isolated environment
- compile_check: Compile a Python file and report syntax errors / undefined names
- propose_patch: Generate a unified diff patch
- apply_patch: Apply an approved patch with backup
- save_debugging_note: Save a note to the knowledge base

All path operations are validated against the workspace root
to prevent path traversal attacks.
"""

import os
import re
import json
import shutil
import difflib
import hashlib
import logging
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from compiler import check_python, render_report
from config import (
    AUDIT_LOG_FILE, TEST_EXECUTION_TIMEOUT,
    EXCLUDED_DIRS, SUPPORTED_EXTENSIONS
)

logger = logging.getLogger(__name__)

# ── Audit Logging ────────────────────────────────────────

def _audit_log(action: str, details: dict) -> None:
    """Append an entry to the audit log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        **details
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


# ── Path Validation ──────────────────────────────────────

def validate_path(filepath: str, workspace_root: str) -> tuple[bool, str, Optional[Path]]:
    """
    Validate that a file path is within the workspace root.

    Returns:
        (is_valid, message, resolved_path_or_None)
    """
    if not filepath or not workspace_root:
        return False, "Empty path or workspace root", None

    try:
        workspace = Path(workspace_root).resolve()
        target = (workspace / filepath).resolve()
    except Exception as e:
        return False, f"Path validation error: {e}", None

    # Containment is a path-operation check, not a string-prefix check: a sibling
    # directory named like "workspace-evil" shares the string prefix but is outside
    # the root. relative_to() compares real path components and raises if it escapes.
    try:
        target.relative_to(workspace)
    except ValueError:
        return False, f"Path traversal detected: {filepath}", None

    # Check for shell metacharacters. ".." is allowed here only because the path is
    # already proven to stay inside the workspace above.
    suspicious = ["~", "$", "%", "|", ";", "&", "`"]
    for pattern in suspicious:
        if pattern in filepath:
            return False, f"Suspicious pattern in path: {pattern}", None

    return True, "Valid path", target


def _should_exclude_dir(dir_name: str) -> bool:
    """Check if a directory should be excluded from file operations."""
    return dir_name in EXCLUDED_DIRS or dir_name.endswith(".egg-info")


# ── Tool: list_files ─────────────────────────────────────

def list_files(workspace_root: str, directory: str = ".",
               extensions: list[str] = None,
               max_depth: int = 5) -> dict:
    """
    List files in the workspace directory.

    Args:
        workspace_root: Absolute path to workspace root
        directory: Relative directory within workspace (default ".")
        extensions: Filter by file extensions (e.g., [".py", ".md"])
        max_depth: Maximum directory depth to traverse

    Returns:
        dict with keys: success, files (list of {path, name, size, extension}), error
    """
    # Models often send "", null, "/" or "./" to mean "the whole workspace"; only a missing key hits the default.
    directory = (directory or "").strip()
    if directory in ("", "/", "\\", "./", ".\\"):
        directory = "."
    if isinstance(extensions, str):                  # e.g. ".py" instead of [".py"]
        extensions = [extensions]

    valid, msg, base_path = validate_path(directory, workspace_root)
    if not valid:
        _audit_log("list_files", {"directory": directory, "error": msg})
        return {"success": False, "files": [], "error": msg}

    if not base_path.exists():
        return {"success": False, "files": [], "error": f"Directory not found: {directory}"}

    if not base_path.is_dir():
        return {"success": False, "files": [], "error": f"Not a directory: {directory}"}

    # Normalize extensions so that e.g. ['py'] or ['.py'] both match correctly
    normalized_exts = None
    if extensions:
        normalized_exts = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
            if ext
        ]

    files = []
    workspace = Path(workspace_root).resolve()

    try:
        for root, dirs, filenames in os.walk(str(base_path)):
            # Calculate depth
            depth = len(Path(root).relative_to(base_path).parts)
            if depth > max_depth:
                dirs.clear()
                continue

            # Filter excluded directories
            dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

            for fname in sorted(filenames):
                fpath = Path(root) / fname

                # Apply extension filter
                if normalized_exts and fpath.suffix.lower() not in normalized_exts:
                    continue

                try:
                    rel_path = fpath.relative_to(workspace)
                    stat = fpath.stat()
                    files.append({
                        "path": str(rel_path).replace("\\", "/"),
                        "name": fname,
                        "size": stat.st_size,
                        "extension": fpath.suffix.lower(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    continue

    except Exception as e:
        return {"success": False, "files": [], "error": str(e)}

    _audit_log("list_files", {"directory": directory, "count": len(files)})
    return {"success": True, "files": files, "error": None}


# ── Tool: read_file ──────────────────────────────────────

def read_file(workspace_root: str, filepath: str,
              start_line: int = None, end_line: int = None,
              max_lines: int = 200) -> dict:
    """
    Read file contents with optional line range.

    Args:
        workspace_root: Absolute path to workspace root
        filepath: Relative path within workspace
        start_line: Starting line number (1-based, inclusive)
        end_line: Ending line number (1-based, inclusive)
        max_lines: Maximum lines to return (default 200)

    Returns:
        dict with keys: success, content, total_lines, shown_lines, error
    """
    valid, msg, full_path = validate_path(filepath, workspace_root)
    if not valid:
        _audit_log("read_file", {"filepath": filepath, "error": msg})
        return {"success": False, "content": "", "total_lines": 0,
                "shown_lines": (0, 0), "error": msg}

    if not full_path.exists():
        return {"success": False, "content": "", "total_lines": 0,
                "shown_lines": (0, 0), "error": f"File not found: {filepath}"}

    if not full_path.is_file():
        return {"success": False, "content": "", "total_lines": 0,
                "shown_lines": (0, 0), "error": f"Not a file: {filepath}"}

    # Reject binary files
    binary_extensions = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
                         ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
                         ".pdf", ".zip", ".tar", ".gz", ".whl"}
    if full_path.suffix.lower() in binary_extensions:
        return {"success": False, "content": "", "total_lines": 0,
                "shown_lines": (0, 0), "error": f"Binary file: {filepath}"}

    try:
        # Try UTF-8 first, then latin-1
        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = full_path.read_text(encoding="latin-1")

        lines = content.splitlines()
        total = len(lines)

        # Apply line range
        s = max(1, start_line or 1) - 1  # Convert to 0-based
        e = min(total, end_line or total)

        # Enforce max_lines
        if e - s > max_lines:
            e = s + max_lines

        selected = lines[s:e]

        # Add line numbers
        numbered = "\n".join(
            f"{i+s+1:4d} | {line}" for i, line in enumerate(selected)
        )

        _audit_log("read_file", {"filepath": filepath, "lines": f"{s+1}-{e}"})
        return {
            "success": True,
            "content": numbered,
            "total_lines": total,
            "shown_lines": (s + 1, e),
            "error": None
        }

    except Exception as e:
        return {"success": False, "content": "", "total_lines": 0,
                "shown_lines": (0, 0), "error": str(e)}


# ── Tool: search_code ────────────────────────────────────

def search_code(workspace_root: str, pattern: str,
                file_extensions: list[str] = None,
                max_results: int = 20,
                case_sensitive: bool = False) -> dict:
    """
    Search for a pattern in workspace code files.

    Args:
        workspace_root: Absolute path to workspace root
        pattern: Regex pattern or plain text to search for
        file_extensions: Filter by extensions (default: all text files)
        max_results: Maximum number of matches to return
        case_sensitive: Whether search is case-sensitive

    Returns:
        dict with keys: success, matches (list of {file, line, content, line_number}), error
    """
    if not pattern:
        return {"success": False, "matches": [], "error": "Empty search pattern"}

    workspace = Path(workspace_root).resolve()
    if not workspace.exists():
        return {"success": False, "matches": [], "error": "Workspace not found"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error:
        # Fall back to literal search
        compiled = re.compile(re.escape(pattern), flags)

    if file_extensions:
        exts = [
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in file_extensions
            if e
        ]
    else:
        exts = [e for e in SUPPORTED_EXTENSIONS if e != ".json"]
    matches = []

    try:
        for root, dirs, filenames in os.walk(str(workspace)):
            dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

            for fname in filenames:
                fpath = Path(root) / fname
                if fpath.suffix.lower() not in exts:
                    continue

                try:
                    try:
                        content = fpath.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        content = fpath.read_text(encoding="latin-1")

                    for line_num, line in enumerate(content.splitlines(), 1):
                        if compiled.search(line):
                            rel = fpath.relative_to(workspace)
                            matches.append({
                                "file": str(rel).replace("\\", "/"),
                                "line_number": line_num,
                                "content": line.strip()[:200]
                            })
                            if len(matches) >= max_results:
                                break

                except Exception:
                    continue

                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

    except Exception as e:
        return {"success": False, "matches": [], "error": str(e)}

    _audit_log("search_code", {"pattern": pattern, "matches": len(matches)})
    return {"success": True, "matches": matches, "error": None}


# ── Tool: search_knowledge_base ──────────────────────────

def search_knowledge_base(query: str, workspace_root: str = None,
                          collection: str = "workspace_code",
                          top_k: int = 5) -> dict:
    """
    Search the RAG knowledge base for relevant code/documentation.

    Args:
        query: Natural language query
        workspace_root: Workspace path (for context)
        collection: Collection to search ("workspace_code" or "reference_docs")
        top_k: Number of results to return

    Returns:
        dict with keys: success, results (list of {text, source, chunk_id, distance}), error
    """
    if not query:
        return {"success": False, "results": [], "error": "Empty query"}

    try:
        from vector_store import VectorStore
        vs = VectorStore()
        results = vs.search(query, collection_name=collection, top_k=top_k)

        formatted = []
        for r in results:
            formatted.append({
                "text": r.get("text", ""),
                "source": r.get("metadata", {}).get("source", "unknown"),
                "chunk_id": r.get("metadata", {}).get("chunk_id", "unknown"),
                "start_line": r.get("metadata", {}).get("start_line", 0),
                "end_line": r.get("metadata", {}).get("end_line", 0),
                "distance": r.get("distance", 1.0)
            })

        _audit_log("search_knowledge_base", {"query": query[:100], "results": len(formatted)})
        return {"success": True, "results": formatted, "error": None}

    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return {"success": False, "results": [], "error": str(e)}


# ── Tool: get_file_metadata ──────────────────────────────

def get_file_metadata(workspace_root: str, filepath: str) -> dict:
    """
    Get metadata for a file in the workspace.

    Returns:
        dict with keys: success, metadata (name, path, size, lines, extension,
                         modified, functions, classes), error
    """
    valid, msg, full_path = validate_path(filepath, workspace_root)
    if not valid:
        return {"success": False, "metadata": {}, "error": msg}

    if not full_path.exists() or not full_path.is_file():
        return {"success": False, "metadata": {}, "error": f"File not found: {filepath}"}

    try:
        stat = full_path.stat()
        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = full_path.read_text(encoding="latin-1")

        lines = content.splitlines()

        # Extract Python symbols if .py file
        functions = []
        classes = []
        imports = []
        if full_path.suffix == ".py":
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("def "):
                    match = re.match(r"def\s+(\w+)", stripped)
                    if match:
                        functions.append({"name": match.group(1), "line": i})
                elif stripped.startswith("class "):
                    match = re.match(r"class\s+(\w+)", stripped)
                    if match:
                        classes.append({"name": match.group(1), "line": i})
                elif stripped.startswith(("import ", "from ")):
                    imports.append(stripped)

        metadata = {
            "name": full_path.name,
            "path": filepath,
            "size_bytes": stat.st_size,
            "total_lines": len(lines),
            "extension": full_path.suffix,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "functions": functions,
            "classes": classes,
            "imports": imports[:20]  # Limit imports shown
        }

        _audit_log("get_file_metadata", {"filepath": filepath})
        return {"success": True, "metadata": metadata, "error": None}

    except Exception as e:
        return {"success": False, "metadata": {}, "error": str(e)}


# ── Tool: propose_patch ──────────────────────────────────

_LINENO_PREFIX = re.compile(r"^\s*\d+\s\|\s?")


def _strip_read_file_numbers(text: str) -> str:
    """read_file shows '   3 | code'; models sometimes copy those prefixes. Remove them."""
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty and all(_LINENO_PREFIX.match(ln) for ln in nonempty):
        return "\n".join(_LINENO_PREFIX.sub("", ln, count=1) for ln in lines)
    return text


def _replace_snippet(source: str, old_text: str, new_text: str):
    """Replace old_text (must match exactly once) with new_text. Returns (new_source, error)."""
    source = source.replace("\r\n", "\n")
    old_text = _strip_read_file_numbers(old_text.replace("\r\n", "\n"))
    new_text = _strip_read_file_numbers(new_text.replace("\r\n", "\n"))
    if not old_text.strip():
        return None, "old_text is empty"
    n = source.count(old_text)
    if n == 1:
        return source.replace(old_text, new_text, 1), None
    if n > 1:
        return None, f"old_text matches {n} places; include more surrounding lines to make it unique"
    # whitespace-tolerant fallback: compare stripped lines
    src_lines, old_lines = source.split("\n"), old_text.split("\n")
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    key = [ln.strip() for ln in old_lines]
    hits = [i for i in range(len(src_lines) - len(key) + 1)
            if [ln.strip() for ln in src_lines[i:i + len(key)]] == key]
    if len(hits) != 1:
        return None, "old_text was not found in the file; copy the exact lines from read_file"
    i = hits[0]
    lead = lambda ln: ln[:len(ln) - len(ln.lstrip())]
    file_indent, given_indent = lead(src_lines[i]), lead(old_lines[0])
    repl = new_text.rstrip("\n").split("\n")
    if file_indent != given_indent:
        repl = [(file_indent + ln[len(given_indent):]) if ln.startswith(given_indent) and ln.strip() else ln
                for ln in repl]
    src_lines[i:i + len(key)] = repl
    return "\n".join(src_lines), None


def propose_patch(workspace_root: str, filepath: str,
                  original_content: str = "", modified_content: str = "",
                  old_text: str = None, new_text: str = None) -> dict:
    """
    Generate a unified diff patch for a file.

    Args:
        workspace_root: Workspace root path
        filepath: Relative path to the file
        original_content: Original file content
        modified_content: Proposed modified content

    Returns:
        dict with keys: success, diff, filepath, stats (additions, deletions), error
    """
    if not filepath:
        return {"success": False, "diff": "", "filepath": "", "stats": {}, "error": "No filepath"}

    # Validate path
    valid, msg, full_path = validate_path(filepath, workspace_root)
    if not valid:
        return {"success": False, "diff": "", "filepath": filepath, "stats": {}, "error": msg}

    # Preferred mode: small find-and-replace edit. The tool reads the file itself, so the model
    # never has to send (or correctly JSON-escape) whole files.
    if old_text is not None:
        try:
            original_content = full_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        except Exception as exc:
            return {"success": False, "diff": "", "filepath": filepath, "stats": {},
                    "error": f"Could not read {filepath}: {exc}"}
        modified_content, err = _replace_snippet(original_content, old_text, new_text or "")
        if err:
            return {"success": False, "diff": "", "filepath": filepath, "stats": {}, "error": err}
    elif not modified_content:
        return {"success": False, "diff": "", "filepath": filepath, "stats": {},
                "error": "Provide old_text and new_text (small snippet), or full original/modified content"}

    original_lines = original_content.splitlines(keepends=True)
    modified_lines = modified_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines, modified_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm=""
    )

    diff_text = "\n".join(line.rstrip("\r\n") for line in diff)

    # Count additions and deletions
    additions = sum(1 for line in diff_text.splitlines()
                    if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_text.splitlines()
                    if line.startswith("-") and not line.startswith("---"))

    _audit_log("propose_patch", {
        "filepath": filepath,
        "additions": additions,
        "deletions": deletions
    })

    syntax = None                       # does the patched file still compile?
    if filepath.endswith(".py"):
        check = check_python(modified_content, filepath)
        first = next((d for d in check["diagnostics"] if d["severity"] == "error"), None)
        syntax = {"ok": check["ok"], "errors": check["errors"],
                  "message": f"line {first['line']}: {first['message']}" if first else ""}

    return {
        "success": True,
        "diff": diff_text,
        "filepath": filepath,
        "stats": {"additions": additions, "deletions": deletions},
        "syntax": syntax,
        "modified_content": modified_content,   # kept for the approval step; stripped before the LLM sees it
        # only trustworthy when we read the file ourselves (snippet mode), not when the model supplied it
        "original_content": original_content if old_text is not None else None,
        "error": None
    }


# ── Tool: apply_patch ────────────────────────────────────

def apply_patch(workspace_root: str, filepath: str,
                modified_content: str, approved: bool = False) -> dict:
    """
    Apply an approved patch to a file with backup.

    Args:
        workspace_root: Workspace root path
        filepath: Relative path to the file
        modified_content: New file content to write
        approved: Must be True for patch to be applied

    Returns:
        dict with keys: success, backup_path, error
    """
    if not approved:
        return {"success": False, "backup_path": None,
                "error": "Patch not approved by user. Set approved=True after user confirmation."}

    valid, msg, full_path = validate_path(filepath, workspace_root)
    if not valid:
        return {"success": False, "backup_path": None, "error": msg}

    try:
        # Create backup
        backup_dir = Path(workspace_root) / ".codepilot_backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{full_path.stem}_{timestamp}{full_path.suffix}.bak"
        backup_path = backup_dir / backup_name

        if full_path.exists():
            shutil.copy2(str(full_path), str(backup_path))

        # Write modified content
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(modified_content, encoding="utf-8")

        _audit_log("apply_patch", {
            "filepath": filepath,
            "backup": str(backup_path),
            "approved": True
        })

        return {
            "success": True,
            "backup_path": str(backup_path),
            "error": None
        }

    except Exception as e:
        return {"success": False, "backup_path": None, "error": str(e)}


# ── Tool: run_tests ──────────────────────────────────────

def run_tests(workspace_root: str, test_path: str = None,
              use_docker: bool = True, timeout: int = None) -> dict:
    """
    Run tests in an isolated environment.

    Args:
        workspace_root: Workspace root path
        test_path: Specific test file/directory (relative to workspace)
        use_docker: Whether to use Docker isolation (falls back to subprocess)
        timeout: Execution timeout in seconds

    Returns:
        dict with keys: success, output, passed, failed, errors, skipped,
                        execution_mode, duration, error
    """
    timeout = timeout or TEST_EXECUTION_TIMEOUT
    workspace = Path(workspace_root).resolve()

    if not workspace.exists():
        return _test_error("Workspace not found")

    # Determine test target
    if test_path:
        valid, msg, _ = validate_path(test_path, workspace_root)
        if not valid:
            return _test_error(msg)
        test_target = test_path
    else:
        test_target = "."

    # Allowlisted test commands
    cmd_parts = ["python", "-m", "pytest", test_target,
                 "-v", "--tb=short", "--no-header", "-q"]

    start_time = time.time()

    if use_docker:
        result = _run_tests_docker(workspace, cmd_parts, timeout)
    else:
        result = _run_tests_subprocess(workspace, cmd_parts, timeout)

    result["duration"] = round(time.time() - start_time, 2)

    _audit_log("run_tests", {
        "test_path": test_path,
        "mode": result.get("execution_mode", "unknown"),
        "passed": result.get("passed", 0),
        "failed": result.get("failed", 0),
        "duration": result.get("duration", 0)
    })

    return result


def _test_error(msg: str) -> dict:
    """Return a standardized test error result."""
    return {
        "success": False, "output": "", "passed": 0, "failed": 0,
        "errors": 0, "skipped": 0, "execution_mode": "none",
        "duration": 0, "error": msg
    }


def _run_tests_docker(workspace: Path, cmd_parts: list, timeout: int) -> dict:
    """Run tests inside a Docker container with security restrictions."""
    try:
        # Check Docker availability
        check = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        if check.returncode != 0:
            logger.info("Docker not available, falling back to subprocess")
            return _run_tests_subprocess(workspace, cmd_parts, timeout)

        # Build Docker command with security restrictions
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",                    # No network access
            "--memory", "512m",                     # Memory limit
            "--cpus", "1.0",                        # CPU limit
            "--read-only",                          # Read-only root filesystem
            "--tmpfs", "/tmp:rw,noexec,size=100m",  # Writable /tmp
            "-v", f"{workspace}:/workspace:ro",     # Mount workspace read-only
            "-w", "/workspace",
            "--user", "nobody",                     # Non-root user
            "python:3.12-slim",
            *cmd_parts
        ]

        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace)
        )

        output = result.stdout + result.stderr
        stats = _parse_pytest_output(output)

        return {
            "success": result.returncode == 0,
            "output": output[-3000:],  # Limit output size
            "execution_mode": "docker",
            "error": None,
            **stats
        }

    except subprocess.TimeoutExpired:
        return _test_error(f"Test execution timed out after {timeout}s (Docker)")
    except FileNotFoundError:
        logger.info("Docker not found, falling back to subprocess")
        return _run_tests_subprocess(workspace, cmd_parts, timeout)
    except Exception as e:
        logger.warning(f"Docker test execution failed: {e}")
        return _run_tests_subprocess(workspace, cmd_parts, timeout)


def _run_tests_subprocess(workspace: Path, cmd_parts: list, timeout: int) -> dict:
    """
    Run tests in a subprocess (lower security, development mode).
    Creates a temporary copy of the workspace for test execution.
    """
    tmp_dir = None
    try:
        # Create a temporary copy for test isolation
        tmp_dir = tempfile.mkdtemp(prefix="codepilot_test_")
        test_workspace = Path(tmp_dir) / "workspace"
        shutil.copytree(str(workspace), str(test_workspace),
                        ignore=shutil.ignore_patterns(
                            "__pycache__", ".git", "chroma_db",
                            "*.pyc", ".venv", "venv"
                        ))

        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(test_workspace)
        )

        output = result.stdout + result.stderr
        stats = _parse_pytest_output(output)

        return {
            "success": result.returncode == 0,
            "output": output[-3000:],
            "execution_mode": "subprocess (development mode - limited isolation)",
            "error": None,
            **stats
        }

    except subprocess.TimeoutExpired:
        return _test_error(f"Test execution timed out after {timeout}s")
    except Exception as e:
        return _test_error(f"Subprocess test execution failed: {e}")
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def _parse_pytest_output(output: str) -> dict:
    """Parse pytest output to extract pass/fail/skip/error counts."""
    stats = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}

    # Try to match pytest summary line like "3 passed, 1 failed, 1 error"
    summary_match = re.search(
        r"(\d+)\s+passed", output
    )
    if summary_match:
        stats["passed"] = int(summary_match.group(1))

    failed_match = re.search(r"(\d+)\s+failed", output)
    if failed_match:
        stats["failed"] = int(failed_match.group(1))

    error_match = re.search(r"(\d+)\s+error", output)
    if error_match:
        stats["errors"] = int(error_match.group(1))

    skipped_match = re.search(r"(\d+)\s+skipped", output)
    if skipped_match:
        stats["skipped"] = int(skipped_match.group(1))

    return stats


# ── Tool: compile_check ──────────────────────────────────

def compile_check(workspace_root: str, filepath: str) -> dict:
    """Compile a workspace Python file (without running it) and report errors with line numbers."""
    valid, msg, full_path = validate_path(filepath, workspace_root)
    if not valid:
        return {"success": False, "error": msg}
    if full_path.suffix != ".py":
        return {"success": False, "error": "compile_check only supports .py files"}
    try:
        source = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"success": False, "error": f"Could not read {filepath}: {exc}"}
    result = check_python(source, filepath)
    _audit_log("compile_check", {"filepath": filepath, "errors": result["errors"],
                                 "warnings": result["warnings"]})
    return {"success": True, "compiles": result["ok"], "errors": result["errors"],
            "warnings": result["warnings"], "report": render_report(result["diagnostics"], filepath),
            "error": None}


# ── Tool: save_debugging_note ────────────────────────────

def save_debugging_note(note: str, context: dict = None) -> dict:
    """
    Save a debugging note to the reference knowledge base.

    Args:
        note: The debugging note text
        context: Optional context dict (file, error, solution)

    Returns:
        dict with keys: success, error
    """
    if not note:
        return {"success": False, "error": "Empty note"}

    try:
        from vector_store import VectorStore
        vs = VectorStore()
        success = vs.save_debugging_note(note, context)
        _audit_log("save_debugging_note", {"note_length": len(note), "success": success})
        return {"success": success, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Tool Descriptors (for agent) ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "list_files",
        "description": "List files in the workspace directory. Use to discover project structure and find relevant source files.",
        "parameters": {
            "directory": {"type": "string", "description": "Relative directory path within workspace (default: '.')", "default": "."},
            "extensions": {"type": "array", "description": "Optional filter by file extensions (e.g., ['.py', '.md'])", "default": None}
        },
        "required": []
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file with line numbers. Use to inspect source code, configuration, or documentation.",
        "parameters": {
            "filepath": {"type": "string", "description": "Relative file path within the workspace"},
            "start_line": {"type": "integer", "description": "Starting line number (1-based)", "default": None},
            "end_line": {"type": "integer", "description": "Ending line number (1-based)", "default": None}
        },
        "required": ["filepath"]
    },
    {
        "name": "search_code",
        "description": "Search for a text pattern or regex across workspace code files. Use to find function definitions, variable usage, error messages, or specific code patterns.",
        "parameters": {
            "pattern": {"type": "string", "description": "Search pattern (text or regex)"},
            "file_extensions": {"type": "array", "description": "Optional filter by file extensions", "default": None}
        },
        "required": ["pattern"]
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the RAG knowledge base for relevant code snippets and documentation. Use to find context about functions, classes, errors, or concepts from the indexed workspace.",
        "parameters": {
            "query": {"type": "string", "description": "Natural language search query"},
            "collection": {"type": "string", "description": "Collection to search: 'workspace_code' or 'reference_docs'", "default": "workspace_code"}
        },
        "required": ["query"]
    },
    {
        "name": "get_file_metadata",
        "description": "Get metadata about a file including size, line count, functions, classes, and imports. Use to understand file structure before reading it.",
        "parameters": {
            "filepath": {"type": "string", "description": "Relative file path within the workspace"}
        },
        "required": ["filepath"]
    },
    {
        "name": "run_tests",
        "description": "Execute pytest tests in an isolated environment. Use to validate code changes, find test failures, or verify fixes.",
        "parameters": {
            "test_path": {"type": "string", "description": "Relative path to test file or directory", "default": None}
        },
        "required": []
    },
    {
        "name": "compile_check",
        "description": "Compile a Python file WITHOUT running it and list syntax errors and undefined names with line numbers. Use to find syntax problems quickly and to confirm a file is valid.",
        "parameters": {
            "filepath": {"type": "string", "description": "Relative path of the .py file to check"}
        },
        "required": ["filepath"]
    },
    {
        "name": "propose_patch",
        "description": "Propose a code change as a diff for the user to approve. Pass the file path plus old_text (the exact lines to replace, copied from read_file without the line-number prefixes) and new_text (the replacement). Keep both SHORT: only the changed lines plus a line or two of context. Do not send whole files.",
        "parameters": {
            "filepath": {"type": "string", "description": "Relative file path to modify"},
            "old_text": {"type": "string", "description": "Exact existing lines to replace (short snippet, must match once)"},
            "new_text": {"type": "string", "description": "Replacement lines"}
        },
        "required": ["filepath", "old_text", "new_text"]
    },
    {
        "name": "save_debugging_note",
        "description": "Save a debugging note or finding to the knowledge base for future reference. Use after diagnosing a problem to preserve the knowledge.",
        "parameters": {
            "note": {"type": "string", "description": "The debugging note or finding to save"},
            "context": {"type": "object", "description": "Optional context dict with keys like 'file', 'error', 'solution'", "default": None}
        },
        "required": ["note"]
    }
]


def get_tool_descriptions() -> str:
    """Return formatted tool descriptions for the agent prompt."""
    descriptions = []
    for tool in TOOL_DEFINITIONS:
        params_str = ", ".join(
            f"{name}: {info['type']}"
            + (f" (required)" if name in tool.get("required", []) else " (optional)")
            for name, info in tool["parameters"].items()
        )
        descriptions.append(
            f"- **{tool['name']}**({params_str}): {tool['description']}"
        )
    return "\n".join(descriptions)


def execute_tool(tool_name: str, arguments: dict,
                 workspace_root: str) -> dict:
    """
    Execute a registered tool with validated arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments dictionary
        workspace_root: Current workspace root path

    Returns:
        Tool result dictionary
    """
    tool_map = {
        "list_files": lambda args: list_files(
            workspace_root,
            args.get("directory") or ".",
            args.get("extensions")
        ),
        "read_file": lambda args: read_file(
            workspace_root,
            args.get("filepath", ""),
            args.get("start_line"),
            args.get("end_line")
        ),
        "search_code": lambda args: search_code(
            workspace_root,
            args.get("pattern", ""),
            args.get("file_extensions")
        ),
        "search_knowledge_base": lambda args: search_knowledge_base(
            args.get("query", ""),
            workspace_root,
            args.get("collection", "workspace_code"),
            int(args.get("top_k", 5) or 5),
        ),
        "get_file_metadata": lambda args: get_file_metadata(
            workspace_root,
            args.get("filepath", "")
        ),
        "run_tests": lambda args: run_tests(
            workspace_root,
            args.get("test_path"),
            use_docker=args.get("use_docker", True)
        ),
        "compile_check": lambda args: compile_check(
            workspace_root,
            args.get("filepath", "")
        ),
        "propose_patch": lambda args: propose_patch(
            workspace_root,
            args.get("filepath", ""),
            args.get("original_content", ""),
            args.get("modified_content", ""),
            old_text=args.get("old_text"),
            new_text=args.get("new_text"),
        ),
        "save_debugging_note": lambda args: save_debugging_note(
            args.get("note", ""),
            args.get("context")
        ),
    }

    if tool_name not in tool_map:
        _audit_log("execute_tool", {"tool": tool_name, "error": "Unknown tool"})
        return {"success": False, "error": f"Unknown tool: {tool_name}. Available tools: {list(tool_map.keys())}"}

    try:
        _audit_log("execute_tool", {"tool": tool_name, "args": str(arguments)[:200]})
        return tool_map[tool_name](arguments)
    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        _audit_log("execute_tool", {"tool": tool_name, "error": str(e)})
        return {"success": False, "error": f"Tool execution error: {e}"}
