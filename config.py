"""Configuration module for CodePilot Local.

Centralizes configuration constants, model definitions, RAG parameters,
and system utility functions for Ollama interaction and workspace validation.
"""

from pathlib import Path
from typing import List, Tuple
import urllib.error
import urllib.request
import json
import os

# --- Ollama Connection Endpoints ---
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_GENERATE_URL: str = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL: str = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_EMBED_URL: str = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_TAGS_URL: str = f"{OLLAMA_BASE_URL}/api/tags"

# --- Model Configurations ---
AVAILABLE_MODELS: dict[str, str] = {
    "qwen2.5-coder:7b (Recommended)": "qwen2.5-coder:7b",
    "llama3.2:3b (Small)": "llama3.2:3b",
    "llama3.2:1b (Tiny)": "llama3.2:1b",
    "llama3:latest": "llama3:latest",
    "llama3.1:8b": "llama3.1:8b",
    "dolphin-llama3:latest": "dolphin-llama3:latest",
    "qwen3.5:latest": "qwen3.5:latest",
}

# Current Groq-hosted models. Keep model IDs exactly as Groq expects them.
# The old Llama 3.1/3.3 IDs and several legacy IDs have been retired.
CLOUD_GROQ_MODELS: dict[str, str] = {
    "GPT-OSS 20B (Recommended — faster)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (Advanced reasoning)": "openai/gpt-oss-120b",
}

CLOUD_OPENAI_MODELS: dict[str, str] = {
    "gpt-4o-mini (Recommended)": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

DEFAULT_MODEL: str = "qwen2.5-coder:7b"
EMBEDDING_MODEL: str = "nomic-embed-text"

# --- Agent & Execution Settings ---
AGENT_MAX_ITERATIONS: int = 10
AGENT_MAX_TOOL_CALLS: int = 15
LLM_TIMEOUT_SECONDS: int = 120
TEST_EXECUTION_TIMEOUT: int = 60
MAX_FIX_ATTEMPTS: int = 3

# --- Cloud rate-limit & token-budget settings (Groq free tier is ~8,000 tokens/minute) ---
GROQ_MAX_COMPLETION_TOKENS: int = 1200   # counts toward the per-minute budget on every request
LLM_RATE_LIMIT_RETRIES: int = 3          # automatic retries after a 429
LLM_RATE_LIMIT_MAX_WAIT: int = 60        # never sleep longer than this (seconds) for a retry
TOOL_RESULT_MAX_CHARS: int = 1800        # size of one tool result sent back to the model
KEEP_FULL_TOOL_RESULTS: int = 2          # older tool results are shrunk to one-line summaries

# --- RAG & Storage Settings ---
RAG_CHUNK_SIZE: int = 800  # characters
RAG_CHUNK_OVERLAP: int = 200
RAG_TOP_K: int = 5
CHROMA_PERSIST_DIR: str = "./chroma_db"
SUPPORTED_EXTENSIONS: list[str] = [
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".cfg",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
]
EXCLUDED_DIRS: set[str] = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "*.egg-info",
    "chroma_db",
    ".idea",
    ".vscode",
}

# --- Logging ---
AUDIT_LOG_FILE: str = "audit_log.jsonl"

# --- Sensitive Paths for Workspace Validation ---
SENSITIVE_DIRS: set[str] = {
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\windows\\system32",
    "/etc",
    "/var",
    "/bin",
    "/sbin",
    "/usr",
    "/sys",
    "/proc",
    "/root",
}


def check_ollama_available() -> bool:
    """Check if the local Ollama service is reachable.

    Returns:
        bool: True if Ollama server responds at OLLAMA_BASE_URL, False otherwise.
    """
    try:
        req = urllib.request.Request(OLLAMA_BASE_URL, headers={"User-Agent": "CodePilot-Local"})
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status in (200, 404, 405)  # Any valid HTTP response indicates service is up
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def get_available_ollama_models() -> List[str]:
    """Retrieve list of currently downloaded model names from Ollama.

    Calls the /api/tags endpoint on Ollama.

    Returns:
        List[str]: List of model name strings, or an empty list if Ollama is unreachable.
    """
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, headers={"User-Agent": "CodePilot-Local"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", [])
                return [m.get("name") for m in models if "name" in m]
        return []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []


def validate_workspace_path(path: str) -> Tuple[bool, str]:
    """Validate whether a path is suitable and safe for use as a workspace directory.

    Checks:
    - Path string is non-empty.
    - Path exists.
    - Path is a directory.
    - Path is not a sensitive OS system path.

    Args:
        path: Workspace path string to validate.

    Returns:
        Tuple[bool, str]: (is_valid, descriptive_message)
    """
    if not path or not path.strip():
        return False, "Workspace path cannot be empty."

    try:
        target = Path(path).expanduser().resolve()
    except Exception as exc:
        return False, f"Invalid path syntax: {exc}"

    if not target.exists():
        return False, f"Path does not exist: {target}"

    if not target.is_dir():
        return False, f"Path is not a directory: {target}"

    # Check against known system/sensitive paths
    resolved_str = str(target).lower().rstrip("/\\")
    for sensitive in SENSITIVE_DIRS:
        sensitive_lower = sensitive.lower().rstrip("/\\")
        if resolved_str == sensitive_lower or resolved_str.startswith(sensitive_lower + os.sep.lower()):
            return False, f"Path is inside a protected system directory: {target}"

    # Also check if it's the root directory (e.g. C:\ or /)
    if target.parent == target:
        return False, f"Root drive/mount cannot be used as workspace directory: {target}"

    return True, f"Workspace path is valid: {target}"
