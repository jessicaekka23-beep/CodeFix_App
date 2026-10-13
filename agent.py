"""
CodePilot Local — Agent Module

Implements a bounded tool-using agent loop with two distinct agentic workflows:
1. Code Investigation and Debugging (Task 1)
2. Automated Fix Generation and Validation (Task 2)

The agent dynamically selects tools based on the task, processes tool outputs,
and enforces iteration/tool-call limits. Supports structured-action parsing
as a fallback for models without native tool calling.
"""

import re
import json
import logging
import time
import requests
from typing import Optional

from config import (
    OLLAMA_GENERATE_URL, OLLAMA_CHAT_URL, DEFAULT_MODEL,
    AGENT_MAX_ITERATIONS, AGENT_MAX_TOOL_CALLS,
    LLM_TIMEOUT_SECONDS, MAX_FIX_ATTEMPTS,
    GROQ_MAX_COMPLETION_TOKENS, LLM_RATE_LIMIT_RETRIES, LLM_RATE_LIMIT_MAX_WAIT,
    TOOL_RESULT_MAX_CHARS, KEEP_FULL_TOOL_RESULTS,
)
from tools import (
    execute_tool, get_tool_descriptions, TOOL_DEFINITIONS,
    validate_path, propose_patch, read_file
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────

def _groq_tool_schemas() -> list:
    """Convert the app's tool registry into OpenAI-compatible function schemas."""
    schemas = []
    for tool in TOOL_DEFINITIONS:
        properties = {}
        for name, info in tool.get("parameters", {}).items():
            prop = {"type": info.get("type", "string")}
            if info.get("description"):
                prop["description"] = info["description"]
            # Complete the JSON Schema for array/object parameters.
            if prop["type"] == "array":
                prop["items"] = {"type": "string"}
            if prop["type"] == "object":
                prop["additionalProperties"] = True
            properties[name] = prop
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": tool.get("required", []),
                    "additionalProperties": False,
                },
            },
        })
    return schemas

# ─────────────────────────────────────────────
# Cloud rate-limit handling (Groq free tier: ~8k tokens/minute)
# ─────────────────────────────────────────────

_RETRY_RE = re.compile(r"(?:(\d+)m(?!s))?\s*(\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)
_GROQ_LIMITS = {"remaining": None, "reset_at": 0.0}   # learned from response headers


def _parse_duration(text: str):
    """'32.86s' -> 32.86, '2m5.2s' -> 125.2, '850ms' -> 0.85. None if unparseable."""
    m = _RETRY_RE.search(text or "")
    if not m:
        return None
    mins, val, unit = m.groups()
    secs = float(val) / 1000 if unit.lower() == "ms" else float(val)
    return secs + (int(mins) * 60 if mins else 0)


def _error_message(data) -> str:
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", err))
    return str(err or "")


def _is_rate_limit(resp, data) -> bool:
    err = data.get("error") if isinstance(data, dict) else None
    code = err.get("code") if isinstance(err, dict) else ""
    return resp.status_code == 429 or code == "rate_limit_exceeded"


def _estimate_tokens(*texts) -> int:
    return sum(len(t or "") for t in texts) // 3   # rough, errs on the safe side


def _remember_limits(headers) -> None:
    try:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = _parse_duration(headers.get("x-ratelimit-reset-tokens", ""))
        if remaining is not None:
            _GROQ_LIMITS["remaining"] = int(float(remaining))
            _GROQ_LIMITS["reset_at"] = time.time() + (reset or 0)
    except (ValueError, TypeError):
        pass


def _pace(estimated_tokens: int, on_status=None) -> None:
    """Wait *before* sending if the last response said we do not have enough budget left."""
    remaining = _GROQ_LIMITS["remaining"]
    if remaining is None or estimated_tokens <= remaining:
        return
    wait = _GROQ_LIMITS["reset_at"] - time.time()
    if 0 < wait <= LLM_RATE_LIMIT_MAX_WAIT:
        if on_status:
            on_status(f"Token budget low ({remaining} left); pausing {wait:.0f}s to avoid a rate limit...")
        time.sleep(wait + 0.5)
    _GROQ_LIMITS["remaining"] = None


def _post_cloud(endpoint, payload, headers, timeout, estimated_tokens=0, on_status=None):
    """POST with automatic wait-and-retry on rate limits. Returns (response, json_data)."""
    for attempt in range(LLM_RATE_LIMIT_RETRIES + 1):
        _pace(estimated_tokens, on_status)
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        try:
            data = resp.json()
        except ValueError:
            data = {"error": {"message": resp.text[:300] or f"HTTP {resp.status_code}"}}
        _remember_limits(resp.headers)
        if not _is_rate_limit(resp, data):
            return resp, data
        wait = None
        try:
            wait = float(resp.headers.get("retry-after", ""))
        except ValueError:
            wait = _parse_duration(_error_message(data).split("try again in")[-1])
        if attempt >= LLM_RATE_LIMIT_RETRIES or wait is None or wait > LLM_RATE_LIMIT_MAX_WAIT:
            return resp, data
        wait += 1.0
        if on_status:
            on_status(f"Rate limit reached - waiting {wait:.0f}s, then retrying "
                      f"({attempt + 1}/{LLM_RATE_LIMIT_RETRIES})...")
        time.sleep(wait)
    return resp, data


# ─────────────────────────────────────────────
# Recovering from malformed native tool calls (Groq `tool_use_failed`)
# ─────────────────────────────────────────────

_TOOL_RETRY_NUDGE = (
    "\n\nSYSTEM: Your previous tool call was rejected because its JSON was invalid. "
    "Try again with a SHORT tool call. For propose_patch send only filepath, old_text and "
    "new_text containing just the changed lines (never whole files). Escape newlines as \\n "
    "and double quotes as \\\"."
)


def _is_tool_parse_error(data) -> bool:
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return False
    return (err.get("code") == "tool_use_failed"
            or "failed to parse tool call" in str(err.get("message", "")).lower())


def _loads_lenient(text: str):
    """json.loads that also accepts raw newlines/tabs inside strings (the usual failure)."""
    try:
        return json.loads(text, strict=False)
    except (ValueError, TypeError):
        return None


def _recover_tool_call(data) -> str | None:
    """Rebuild a TOOL_CALL line from Groq's `failed_generation` text when it is salvageable."""
    err = data.get("error", {}) if isinstance(data, dict) else {}
    raw = err.get("failed_generation") if isinstance(err, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    name, args = None, None

    # <function=propose_patch>{...}</function>   or   <function=propose_patch({...})>
    m = re.search(r"<function=(\w+)\s*[>(]?\s*(\{.*\})", raw, re.DOTALL)
    if m:
        name, args = m.group(1), _loads_lenient(m.group(2))

    # {"name": "...", "arguments": {...}}
    if args is None:
        start, end = raw.find("{"), raw.rfind("}")
        obj = _loads_lenient(raw[start:end + 1]) if 0 <= start < end else None
        if isinstance(obj, dict) and obj.get("name"):
            name, args = obj["name"], obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = _loads_lenient(args)

    if name and isinstance(args, dict):
        return "TOOL_CALL: " + json.dumps({"name": name, "arguments": args})
    return None


def call_llm(prompt: str, model: str = None,
             system: str = None, temperature: float = 0.1,
             timeout: int = None,
             provider: str = "ollama",
             api_key: str = None,
             base_url: str = None,
             enable_tools: bool = False,
             on_status=None) -> dict:
    """
    Call an LLM (Ollama locally, or Groq/OpenAI cloud).

    Args:
        prompt: User prompt
        model: Model name (default from config)
        system: Optional system prompt
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        provider: "ollama", "groq", or "openai"
        api_key: API key for cloud provider
        base_url: Optional custom base URL for cloud API

    Returns:
        dict with keys: success, response, model, error, duration
    """
    model = model or DEFAULT_MODEL
    timeout = timeout or LLM_TIMEOUT_SECONDS
    start = time.time()

    # Determine provider from model prefix or parameter
    prov = (provider or "ollama").lower()
    if model.startswith("groq/") or "groq" in prov:
        prov = "groq"
        model = model.replace("groq/", "")
    elif model.startswith("openai/") or "openai" in prov:
        prov = "openai"
        model = model.replace("openai/", "")

    # ── Cloud Provider: Groq or OpenAI / OpenRouter ─────────
    if prov in ("groq", "openai", "cloud"):
        import os
        key = api_key
        if not key:
            if prov == "groq":
                key = os.environ.get("GROQ_API_KEY", "")
            else:
                key = os.environ.get("OPENAI_API_KEY", "")

        if not key:
            return {
                "success": False,
                "response": "",
                "model": model,
                "error": f"Missing API key for {prov.title()} cloud provider. Please enter your API key in Settings.",
                "duration": round(time.time() - start, 2)
            }

        endpoint = base_url.rstrip("/") if base_url else (
            "https://api.groq.com/openai/v1/chat/completions" if prov == "groq"
            else "https://api.openai.com/v1/chat/completions"
        )
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        # Groq-hosted GPT-OSS models require a supported reasoning effort.
        # Keep this parameter off for unrelated providers/models.
        is_gpt_oss = prov == "groq" and model in (
            "openai/gpt-oss-20b", "openai/gpt-oss-120b"
        )
        if is_gpt_oss:
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False
            payload["max_completion_tokens"] = GROQ_MAX_COMPLETION_TOKENS

            # GPT-OSS can emit native tool calls even when an app expects
            # custom TOOL_CALL text. During agent tasks, declare the tools
            # explicitly and allow the model to call them. Convert its native
            # call back into the existing agent-loop format below.
            if enable_tools:
                payload["tools"] = _groq_tool_schemas()
                payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        try:
            est = _estimate_tokens(system, prompt, json.dumps(payload.get("tools", ""))) \
                + payload.get("max_completion_tokens", 0)
            resp, data = _post_cloud(endpoint, payload, headers, timeout, est, on_status)

            # Groq rejects malformed native tool calls server-side. Try to salvage the call,
            # otherwise retry (up to twice) with a nudge to keep the arguments short.
            for _retry in range(2):
                if not _is_tool_parse_error(data):
                    break
                recovered = _recover_tool_call(data)
                if recovered:
                    if on_status:
                        on_status("Recovered a malformed tool call from the model output")
                    data = {"choices": [{"message": {"content": recovered}}]}
                    break
                if on_status:
                    on_status(f"Model produced invalid tool-call JSON; retrying ({_retry + 1}/2)...")
                if _TOOL_RETRY_NUDGE not in payload["messages"][-1]["content"]:
                    payload["messages"][-1]["content"] += _TOOL_RETRY_NUDGE
                payload["temperature"] = max(payload.get("temperature", 0.1), 0.3)
                resp, data = _post_cloud(endpoint, payload, headers, timeout, est, on_status)

            if "error" in data:
                err_msg = _error_message(data)
                if _is_tool_parse_error(data):
                    err_msg = ("The cloud model kept producing a malformed tool call. Click again, "
                               "or switch to Local (Ollama), which does not use native tool calls.")
                if _is_rate_limit(resp, data):
                    err_msg = ("Groq free-tier token limit reached even after waiting and retrying. "
                               "Wait about a minute and click again, shorten the request, or switch to "
                               "Local (Ollama). Details: " + err_msg[:200])
                return {
                    "success": False,
                    "response": "",
                    "model": model,
                    "error": f"{prov.title()} error: {err_msg}",
                    "duration": round(time.time() - start, 2)
                }

            choices = data.get("choices", [])
            if not choices:
                return {
                    "success": False,
                    "response": "",
                    "model": model,
                    "error": f"Empty response choices from {prov.title()}",
                    "duration": round(time.time() - start, 2)
                }

            message = choices[0].get("message", {})
            content = message.get("content") or ""
            native_calls = message.get("tool_calls") or []
            if native_calls:
                first_call = native_calls[0]
                function = first_call.get("function", {})
                name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {}
                content = "TOOL_CALL: " + json.dumps({
                    "name": name,
                    "arguments": arguments if isinstance(arguments, dict) else {}
                })
            return {
                "success": True,
                "response": content,
                "model": model,
                "error": None,
                "duration": round(time.time() - start, 2)
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "response": "",
                "model": model,
                "error": f"Request to {prov.title()} timed out after {timeout}s.",
                "duration": round(time.time() - start, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "model": model,
                "error": f"Failed to call {prov.title()} API: {e}",
                "duration": round(time.time() - start, 2)
            }

    # ── Local Provider: Ollama ──────────────────────────────
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    if system:
        payload["system"] = system

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=timeout
        )
        data = response.json()

        if "error" in data:
            return {
                "success": False,
                "response": "",
                "model": model,
                "error": data["error"],
                "duration": round(time.time() - start, 2)
            }

        return {
            "success": True,
            "response": data.get("response", ""),
            "model": model,
            "error": None,
            "duration": round(time.time() - start, 2)
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "response": "",
            "model": model,
            "error": f"LLM request timed out after {timeout}s. The model may be too large for available memory.",
            "duration": round(time.time() - start, 2)
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "response": "",
            "model": model,
            "error": "Cannot connect to Ollama. Ensure 'ollama serve' is running on http://localhost:11434",
            "duration": round(time.time() - start, 2)
        }
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "model": model,
            "error": str(e),
            "duration": round(time.time() - start, 2)
        }


# ─────────────────────────────────────────────
# Structured Action Parsing
# ─────────────────────────────────────────────

def parse_agent_action(response_text: str) -> dict:
    """
    Parse the agent's response to extract tool calls or final answers.

    The agent is prompted to respond in a specific format:
    - TOOL_CALL: {"name": "tool_name", "arguments": {...}}
    - FINAL_ANSWER: <answer text>

    Returns:
        dict with keys: type ("tool_call", "final_answer", "error"),
                        name, arguments, text
    """
    text = response_text.strip()

    # Try to find FINAL_ANSWER
    final_match = re.search(
        r"FINAL_ANSWER:\s*(.+)",
        text, re.DOTALL | re.IGNORECASE
    )
    if final_match:
        return {
            "type": "final_answer",
            "text": final_match.group(1).strip()
        }

    # Try to find TOOL_CALL with JSON
    tool_match = re.search(
        r"TOOL_CALL:\s*(\{.+?\})\s*$",
        text, re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    if tool_match:
        try:
            tool_data = json.loads(tool_match.group(1))
            name = tool_data.get("name", "")
            arguments = tool_data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"query": arguments}
            return {
                "type": "tool_call",
                "name": name,
                "arguments": arguments
            }
        except json.JSONDecodeError:
            pass

    # Try to find JSON block with tool call structure anywhere in the text
    json_pattern = re.search(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
        text, re.DOTALL
    )
    if json_pattern:
        try:
            name = json_pattern.group(1)
            args = json.loads(json_pattern.group(2))
            return {
                "type": "tool_call",
                "name": name,
                "arguments": args
            }
        except json.JSONDecodeError:
            pass

    # If response is long enough and doesn't contain tool patterns,
    # treat it as a final answer
    if len(text) > 100 and "TOOL_CALL" not in text.upper():
        return {
            "type": "final_answer",
            "text": text
        }

    return {
        "type": "error",
        "text": f"Could not parse agent response. Raw output: {text[:500]}"
    }


# ─────────────────────────────────────────────
# Agent System Prompts
# ─────────────────────────────────────────────

INVESTIGATION_SYSTEM_PROMPT = """You are CodePilot, an expert code investigation and debugging agent. You analyze Python projects to diagnose errors, find bugs, and explain code behavior.

You have access to these tools:
{tools}

## Response Format
You MUST respond with exactly ONE of these formats:

1. To call a tool:
TOOL_CALL: {{"name": "tool_name", "arguments": {{"param": "value"}}}}

2. To provide your final answer:
FINAL_ANSWER: Your detailed analysis here...

## Investigation Strategy
1. First, list files to understand the project structure.
2. Read relevant source files that might be related to the reported issue. Use compile_check to find syntax errors fast.
3. Search for specific patterns, function names, or error messages.
4. Search the knowledge base for relevant context.
5. When you have enough evidence, provide your final answer.

## Rules
- Always start by understanding the project structure.
- Read files before making assumptions about their contents.
- Reference specific file names, line numbers, and code when explaining issues.
- State uncertainty explicitly if evidence is insufficient.
- Never fabricate file contents or code that you haven't actually read.
- Distinguish clearly between retrieved evidence and your own inference.
- Do NOT call a tool and also provide a FINAL_ANSWER in the same response.
"""

FIX_GENERATION_SYSTEM_PROMPT = """You are CodePilot, an expert code fix generation agent. You diagnose bugs in Python projects, generate fixes, and validate them through testing.

You have access to these tools:
{tools}

## Response Format
You MUST respond with exactly ONE of these formats:

1. To call a tool:
TOOL_CALL: {{"name": "tool_name", "arguments": {{"param": "value"}}}}

2. To provide your final answer:
FINAL_ANSWER: Your detailed analysis and proposed fix here...

## Fix Generation Strategy
1. First, investigate the bug by listing files and reading relevant code.
2. Search for related patterns and knowledge base entries.
3. Understand the root cause before proposing a fix.
4. Read the original file content that needs to be modified.
5. Use propose_patch to generate a diff showing the fix. Its result includes a `syntax` check: if the patched file does not compile, revise the patch.
6. Run tests to validate the fix if test files are available.
7. If tests fail, analyze the failure and attempt a revision (up to {max_attempts} attempts).
8. Provide a final answer summarizing the fix, evidence, and test results.

## Rules
- ALWAYS read the file before proposing changes.
- ALWAYS use propose_patch to create a diff preview.
- NEVER apply changes without showing the diff first.
- NEVER claim a fix works just because you generated it — test results must confirm.
- Clearly report: passed tests, failed tests, skipped tests, tests that could not run.
- State what you changed, why, and what evidence supports the fix.
- Do NOT call a tool and also provide a FINAL_ANSWER in the same response.
"""

CHAT_SYSTEM_PROMPT = """You are CodePilot Local, a helpful AI coding assistant that runs entirely on local hardware. You help with Python programming, code review, debugging, and general coding questions.

Be concise, accurate, and helpful. If you're unsure, say so."""


# ─────────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────────

def _build_conversation(header: str, turns: list) -> str:
    """Rebuild the prompt, keeping only the latest tool results in full.

    Older tool results collapse to a one-line summary so the prompt stops growing every
    iteration (the main cause of hitting cloud tokens-per-minute limits).
    """
    tool_idx = [i for i, t in enumerate(turns) if t["short"]]
    keep_full = set(tool_idx[-KEEP_FULL_TOOL_RESULTS:])
    parts = [header]
    for i, t in enumerate(turns):
        if t["short"] and i not in keep_full:
            parts.append("\n" + t["short"])
        else:
            parts.append(t["text"])
    return "".join(parts)


def _seed_retrieval(user_query: str, workspace_root: str, tool_history: list,
                    sources: list, status, emit=None) -> str:
    """Retrieval-first RAG: fetch context BEFORE the first LLM call so every run is grounded.

    Searches the project index (workspace_code) and the saved debugging notes (reference_docs),
    records the calls in the tool log, and returns a compact context block for the prompt.
    The agent can still call search_knowledge_base itself for follow-up questions.
    """
    query = " ".join(user_query.split())[:600]
    blocks, budget = [], 1800
    seen_hits = []                                     # for the optional on_event demo hook
    for collection, label, k in (("workspace_code", "project code", 3),
                                 ("reference_docs", "past debugging notes", 2)):
        try:
            res = execute_tool("search_knowledge_base",
                               {"query": query, "collection": collection, "top_k": k},
                               workspace_root)
        except Exception as exc:                       # RAG must never crash the agent
            status(f"RAG unavailable ({exc}); continuing without seeded context")
            return ""
        hits = res.get("results", []) if res.get("success") else []
        tool_history.append({
            "iteration": 0, "tool": "search_knowledge_base",
            "arguments": {"query": query[:80], "collection": collection, "auto": True},
            "result_summary": f"Retrieved {len(hits)} chunks from {label} (auto)",
            "success": res.get("success", False),
        })
        for h in hits:
            src, start, end = h.get("source", "?"), h.get("start_line", "?"), h.get("end_line", "?")
            sources.append({"source": src, "chunk_id": h.get("chunk_id", ""), "lines": f"{start}-{end}"})
            seen_hits.append({"collection": collection, "label": label, "source": src, "chunk_id": h.get("chunk_id", ""),
                              "start": start, "end": end, "distance": h.get("distance"),
                              "text": (h.get("text") or "").strip()[:700]})
            snippet = (h.get("text") or "").strip()[:500]
            entry = f"[{label}: {src} lines {start}-{end}]\n{snippet}\n"
            if budget - len(entry) < 0:
                break
            budget -= len(entry)
            blocks.append(entry)
    if emit:
        emit("seed", query=query, hits=seen_hits, used=len(blocks))
    if not blocks:
        status("No indexed context found - index the workspace first for grounded answers")
        return ""
    status(f"RAG: retrieved {len(blocks)} relevant chunks")
    return ("Retrieved context (from the RAG index; verify against the real files before "
            "relying on it):\n" + "\n".join(blocks) + "\n")


def run_agent_loop(
    user_query: str,
    workspace_root: str,
    model: str = None,
    task_type: str = "investigate",
    max_iterations: int = None,
    max_tool_calls: int = None,
    on_status: callable = None,
    provider: str = "ollama",
    api_key: str = None,
    base_url: str = None,
    on_event: callable = None
) -> dict:
    """
    Run the bounded agent loop for code investigation or fix generation.

    Args:
        user_query: The user's question or request
        workspace_root: Absolute path to the workspace
        model: LLM model to use
        task_type: "investigate" (Task 1) or "fix" (Task 2)
        max_iterations: Override max iterations
        max_tool_calls: Override max tool calls
        on_status: Callback for status updates (receives str)
        provider: "ollama", "groq", or "openai"
        api_key: Cloud API key (if using cloud provider)
        base_url: Optional custom base URL for cloud provider
        on_event: Optional callback receiving structured events (dicts with a "type" key:
                  start, seed, iteration, llm, tool, patch, note, final). Used by the demo UI.

    Returns:
        dict with keys: success, answer, tool_history, sources, error,
                        iterations, total_duration, model, task_type,
                        proposed_patches (for fix task)
    """
    model = model or DEFAULT_MODEL
    max_iter = max_iterations or AGENT_MAX_ITERATIONS
    max_tools = max_tool_calls or AGENT_MAX_TOOL_CALLS

    # Select system prompt
    tools_desc = get_tool_descriptions()
    if task_type == "investigate":
        system = INVESTIGATION_SYSTEM_PROMPT.format(tools=tools_desc)
    elif task_type == "fix":
        system = FIX_GENERATION_SYSTEM_PROMPT.format(
            tools=tools_desc, max_attempts=MAX_FIX_ATTEMPTS
        )
    else:
        # Simple chat — no agent loop needed
        result = call_llm(
            user_query, model=model, system=CHAT_SYSTEM_PROMPT,
            provider=provider, api_key=api_key, base_url=base_url,
            on_status=on_status
        )
        return {
            "success": result["success"],
            "answer": result["response"],
            "tool_history": [],
            "sources": [],
            "error": result["error"],
            "iterations": 1,
            "total_duration": result["duration"],
            "model": model,
            "task_type": "chat",
            "proposed_patches": []
        }

    # Build conversation history
    tool_history = []
    sources = []
    proposed_patches = []
    header = f"User request: {user_query}\n\n"
    turns = []   # {"text": full text, "short": one-line summary or None for notes}
    iteration = 0
    tool_call_count = 0
    start_time = time.time()

    def status(msg):
        if on_status:
            on_status(msg)
        logger.info(msg)

    def emit(kind, **data):
        if on_event:
            try:
                on_event({"type": kind, **data})
            except Exception:                              # a display hook must never break the agent
                logger.debug("on_event failed", exc_info=True)

    status(f"Starting {task_type} task with model {model} ({provider})")
    emit("start", task=task_type, model=model, provider=provider, max_iterations=max_iter, max_tool_calls=max_tools)
    header += _seed_retrieval(user_query, workspace_root, tool_history, sources, status, emit)

    while iteration < max_iter:
        iteration += 1
        status(f"Iteration {iteration}/{max_iter}")
        emit("iteration", n=iteration, max=max_iter)

        # Call LLM
        prompt = _build_conversation(header, turns) + "\nRespond with either a TOOL_CALL or FINAL_ANSWER."
        llm_started = time.time()
        llm_result = call_llm(
            prompt, model=model, system=system,
            provider=provider, api_key=api_key, base_url=base_url,
            enable_tools=True, on_status=status
        )

        if not llm_result["success"]:
            return {
                "success": False,
                "answer": "",
                "tool_history": tool_history,
                "sources": sources,
                "error": f"LLM error: {llm_result['error']}",
                "iterations": iteration,
                "total_duration": round(time.time() - start_time, 2),
                "model": model,
                "task_type": task_type,
                "proposed_patches": proposed_patches
            }

        # Parse the response
        action = parse_agent_action(llm_result["response"])
        response_text = llm_result["response"] or ""
        thought = response_text.split("TOOL_CALL", 1)[0].strip() if action["type"] == "tool_call" else ""
        emit("llm", seconds=round(time.time() - llm_started, 1), prompt_chars=len(prompt), reply_chars=len(response_text),
             action=action["type"], thought=thought[:400])

        if action["type"] == "final_answer":
            status("Agent reached final answer")
            emit("final", chars=len(action["text"] or ""))
            return {
                "success": True,
                "answer": action["text"],
                "tool_history": tool_history,
                "sources": sources,
                "error": None,
                "iterations": iteration,
                "total_duration": round(time.time() - start_time, 2),
                "model": model,
                "task_type": task_type,
                "proposed_patches": proposed_patches
            }

        elif action["type"] == "tool_call":
            tool_call_count += 1
            if tool_call_count > max_tools:
                status(f"Tool call limit reached ({max_tools})")
                emit("note", text=f"Tool call limit reached ({max_tools}); asking for a final answer")
                # Force a final answer
                turns.append({"short": None, "text": (
                    f"\n\nSYSTEM: Tool call limit reached ({max_tools}). "
                    "You must now provide a FINAL_ANSWER based on all information gathered so far."
                )})
                continue

            tool_name = action["name"]
            tool_args = action.get("arguments", {})
            status(f"Calling tool: {tool_name}({json.dumps(tool_args)[:100]})")

            # Execute the tool
            tool_started = time.time()
            tool_result = execute_tool(tool_name, tool_args, workspace_root)
            patch_content = tool_result.pop("modified_content", None)   # not sent back to the LLM
            patch_original = tool_result.pop("original_content", None)

            # Record in history
            history_entry = {
                "iteration": iteration,
                "tool": tool_name,
                "arguments": tool_args,
                "result_summary": _summarize_tool_result(tool_result),
                "success": tool_result.get("success", False)
            }
            tool_history.append(history_entry)
            emit("tool", name=tool_name, args=tool_args, summary=history_entry["result_summary"],
                 success=history_entry["success"], seconds=round(time.time() - tool_started, 2))

            # Collect sources from knowledge base results
            if tool_name == "search_knowledge_base" and tool_result.get("success"):
                for r in tool_result.get("results", []):
                    sources.append({
                        "source": r.get("source", ""),
                        "chunk_id": r.get("chunk_id", ""),
                        "lines": f"{r.get('start_line', '?')}-{r.get('end_line', '?')}"
                    })

            # Collect proposed patches
            if tool_name == "propose_patch" and tool_result.get("success"):
                emit("patch", filepath=tool_result.get("filepath", ""), stats=tool_result.get("stats", {}),
                     syntax_ok=(tool_result.get("syntax") or {}).get("ok", True))
                proposed_patches.append({
                    "filepath": tool_result.get("filepath", ""),
                    "diff": tool_result.get("diff", ""),
                    "stats": tool_result.get("stats", {}),
                    "syntax": tool_result.get("syntax"),
                    "modified_content": patch_content,
                    "original_content": patch_original,
                })

            # Add tool result to conversation
            # Compact JSON (no indent) and a hard size cap keep each request small
            result_str = json.dumps(tool_result, separators=(",", ":"), default=str)
            if len(result_str) > TOOL_RESULT_MAX_CHARS:
                result_str = result_str[:TOOL_RESULT_MAX_CHARS] + "...(truncated)"

            turns.append({
                "short": (f"Tool call: {tool_name}({json.dumps(tool_args)[:120]}) -> "
                          f"{history_entry['result_summary']}"),
                "text": (f"\n\nTool call: {tool_name}({json.dumps(tool_args)})\n"
                         f"Tool result:\n{result_str}\n"),
            })

        elif action["type"] == "error":
            status(f"Parse error: {action['text'][:100]}")
            emit("note", text="The model's reply could not be parsed; asking it to retry")
            turns.append({"short": None, "text": (
                f"\n\nSYSTEM: Your last response could not be parsed. "
                "Please respond with either:\n"
                'TOOL_CALL: {"name": "tool_name", "arguments": {"param": "value"}}\n'
                "or:\n"
                "FINAL_ANSWER: your answer here\n"
            )})

    # Max iterations reached
    status(f"Max iterations reached ({max_iter})")
    emit("note", text=f"Max iterations reached ({max_iter}); asking for a summary")
    # Make one final attempt to get a summary
    turns.append({"short": None, "text": (
        f"\n\nSYSTEM: Maximum iterations ({max_iter}) reached. "
        "Provide a FINAL_ANSWER summarizing everything you have found so far."
    )})
    llm_result = call_llm(
        _build_conversation(header, turns), model=model, system=system,
        provider=provider, api_key=api_key, base_url=base_url,
        enable_tools=True, on_status=status
    )
    final_text = llm_result.get("response", "Investigation stopped: iteration limit reached.")

    # Try to extract clean final answer
    action = parse_agent_action(final_text)
    if action["type"] == "final_answer":
        final_text = action["text"]

    return {
        "success": True,
        "answer": final_text,
        "tool_history": tool_history,
        "sources": sources,
        "error": f"Reached maximum iterations ({max_iter})",
        "iterations": iteration,
        "total_duration": round(time.time() - start_time, 2),
        "model": model,
        "task_type": task_type,
        "proposed_patches": proposed_patches
    }


def _summarize_tool_result(result: dict) -> str:
    """Create a brief summary of a tool result for the history."""
    if not result.get("success"):
        return f"Error: {result.get('error', 'Unknown error')}"

    if "files" in result:
        return f"Found {len(result['files'])} files"
    if "content" in result:
        lines = result.get("total_lines", "?")
        return f"Read file ({lines} lines)"
    if "matches" in result:
        return f"Found {len(result['matches'])} matches"
    if "results" in result:
        return f"Retrieved {len(result['results'])} results"
    if "metadata" in result:
        return f"Got metadata for {result['metadata'].get('name', '?')}"
    if "compiles" in result:
        return "Compiles cleanly" if result["compiles"] else f"{result.get('errors', 0)} compile error(s)"
    if "diff" in result:
        stats = result.get("stats", {})
        return f"Patch: +{stats.get('additions', 0)}/-{stats.get('deletions', 0)}"
    if "output" in result:
        passed = result.get("passed", 0)
        failed = result.get("failed", 0)
        return f"Tests: {passed} passed, {failed} failed"

    return "OK"
