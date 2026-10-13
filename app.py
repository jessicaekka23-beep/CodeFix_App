"""
CodePilot Local - Streamlit UI

Two agentic tasks run over a RAG index of your own code:
  1. Investigate    - find the root cause of a bug (retrieve -> inspect -> explain)
  2. Fix & Validate - propose a minimal diff, let you approve it, apply it with a backup
Plus a Compiler panel to check (and optionally run) code and send the errors to the agent.

Local inference through Ollama is the default; cloud engines are optional.
"""

import hashlib
import html
import logging
import os
from pathlib import Path

import streamlit as st

try:                                    # optional: load GROQ_API_KEY etc. from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="CodePilot Local", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from agent import run_agent_loop
from compiler import check_python, render_report, run_python
from config import (
    AVAILABLE_MODELS, CLOUD_GROQ_MODELS, CLOUD_OPENAI_MODELS, DEFAULT_MODEL, EXCLUDED_DIRS,
    check_ollama_available, get_available_ollama_models, validate_workspace_path,
)
from tools import apply_patch

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_WORKSPACE = BASE_DIR / "test_workspace"
TAB_FIX, TAB_COMPILE = 1, 2            # positions in st.tabs(), used by "send to agent"

# ── Session state ─────────────────────────────────────────
_default_ws = str(SAMPLE_WORKSPACE) if SAMPLE_WORKSPACE.is_dir() else ""


def _initial_theme() -> bool:
    """Start in whatever theme Streamlit/the OS is already using."""
    try:
        return st.context.theme.type != "light"
    except Exception:
        return True


for _k, _v in {
    "provider": "ollama", "model": DEFAULT_MODEL, "api_key": "", "base_url": "",
    "workspace_path": _default_ws, "workspace_input": _default_ws,
    "results": {}, "pending_patches": [], "last_fix_query": "", "compile_result": None,
    "compile_notice": None, "goto_tab": None,
}.items():
    st.session_state.setdefault(_k, _v)
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = _initial_theme()

# ── Theme (light / dark) ──────────────────────────────────
PALETTES = {
    "light": dict(bg="#f5f6fb", surface="#ffffff", surface2="#eef0f7", border="#dde1ee",
                  text="#1a1e2c", muted="#5b6479", accent="#4f46e5", accent_strong="#4f46e5",
                  ok="#15803d", warn="#b45309", bad="#c62828", code_bg="#f0f2f9",
                  add="rgba(22,163,74,.16)", dele="rgba(220,38,38,.14)", shadow="0 1px 3px rgba(20,24,40,.08)"),
    "dark":  dict(bg="#0e111c", surface="#161a2a", surface2="#1e2336", border="#2b3150",
                  text="#e7e9f4", muted="#9aa3bd", accent="#9a9cff", accent_strong="#6366f1",
                  ok="#4ade80", warn="#fbbf24", bad="#f87171", code_bg="#1b2033",
                  add="rgba(74,222,128,.15)", dele="rgba(248,113,113,.15)", shadow="0 1px 3px rgba(0,0,0,.4)"),
}

STATIC_CSS = """
.stApp, [data-testid="stAppViewContainer"] {background: var(--bg) !important; color: var(--text);}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {display: none !important;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background: var(--surface) !important;}
section[data-testid="stSidebar"] {border-right: 1px solid var(--border);}
.stApp :where(p, li, span, label, h1, h2, h3, h4, h5, h6, summary, td, th) {color: var(--text) !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: var(--muted) !important;}
.stApp :where(code) {background: var(--code-bg) !important; color: var(--accent) !important; border-radius: 6px;}
.stApp hr {border-color: var(--border) !important;}

/* inputs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] > div {background: var(--surface) !important; border-color: var(--border) !important; border-radius: 10px !important;}
.stApp input, .stApp textarea {color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background: transparent !important;}
.stApp textarea.code-area, .stApp [data-testid="stTextArea"] textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .86rem;}
.stApp ::placeholder {color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: .75;}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {background: var(--surface) !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {color: var(--text) !important;}
[data-baseweb="popover"] li:hover {background: var(--surface2) !important;}

/* buttons */
.stApp [data-testid^="stBaseButton-secondary"] {background: var(--surface) !important; border: 1px solid var(--border) !important;}
.stApp [data-testid^="stBaseButton-secondary"]:hover {border-color: var(--accent) !important;}
.stApp [data-testid="stBaseButton-primary"] {background: var(--accent-strong) !important; border: 1px solid var(--accent-strong) !important;}
.stApp [data-testid="stBaseButton-primary"] :where(p, span) {color: #fff !important;}
.stApp [data-testid^="stBaseButton"] {border-radius: 10px !important; font-weight: 600;}
.stApp [data-testid^="stBaseButton"]:disabled {opacity: .45;}

/* tabs, expanders, containers */
.stApp button[role="tab"] {font-weight: 600; padding: 10px 18px;}
.stApp button[role="tab"] :where(p, span) {color: var(--muted) !important;}
.stApp button[role="tab"][aria-selected="true"] :where(p, span) {color: var(--accent) !important;}
.stApp [data-baseweb="tab-highlight"] {background: var(--accent-strong) !important;}
.stApp [data-baseweb="tab-border"] {background: var(--border) !important;}
.stApp [data-testid="stExpander"] details {background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px;}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--border) !important; border-radius: 12px;}

/* header */
.brand {display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;}
.brand-title {font-size: 1.7rem; font-weight: 800; letter-spacing: -.5px; color: var(--text) !important;}
.brand-sub {color: var(--muted) !important; font-size: .95rem;}
.chips {margin: 6px 0 14px;}
.chip {display: inline-block; padding: 3px 12px; margin: 0 8px 6px 0; border-radius: 999px; font-size: .78rem;
       background: var(--surface) !important; border: 1px solid var(--border); color: var(--text) !important;}
.chip::before {content: "● "; color: var(--muted);}
.chip.ok::before {color: var(--ok);} .chip.warn::before {color: var(--warn);} .chip.bad::before {color: var(--bad);}

/* stat cards */
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 6px 0 14px;}
.stat {background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; box-shadow: var(--shadow);}
.stat-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; color: var(--muted) !important;}
.stat-value {font-size: 1.45rem; font-weight: 700; line-height: 1.25;}
.stat-value.ok {color: var(--ok) !important;} .stat-value.bad {color: var(--bad) !important;} .stat-value.warn {color: var(--warn) !important;}

/* callouts */
.callout {border-left: 4px solid var(--accent); background: var(--surface); border-radius: 10px; padding: 10px 14px; margin: 8px 0; font-size: .92rem;}
.callout.ok {border-color: var(--ok); background: color-mix(in srgb, var(--ok) 11%, var(--surface));}
.callout.warn {border-color: var(--warn); background: color-mix(in srgb, var(--warn) 11%, var(--surface));}
.callout.bad {border-color: var(--bad); background: color-mix(in srgb, var(--bad) 11%, var(--surface));}
.callout b, .callout span {color: var(--text) !important;}

/* code boxes + diffs */
.codebox {background: var(--code-bg); border: 1px solid var(--border); border-radius: 10px; padding: 8px 0; overflow-x: auto;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .83rem; line-height: 1.5; margin: 6px 0;}
.codebox .ln {white-space: pre; padding: 0 14px; color: var(--text) !important;}
.codebox .ln.add {background: var(--add);} .codebox .ln.del {background: var(--dele);}
.codebox .ln.hunk {color: var(--accent) !important;} .codebox .ln.meta {color: var(--muted) !important;}
.codebox .ln.caret {color: var(--bad) !important; font-weight: 700;}
.codebox .gutter {color: var(--muted) !important; display: inline-block; min-width: 3ch; text-align: right; margin-right: 12px; user-select: none;}
.codebox-title {font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; color: var(--muted) !important; margin-top: 10px;}

/* diagnostics */
.diag {background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--bad); border-radius: 10px;
       padding: 10px 14px 4px; margin-bottom: 10px;}
.diag.warning {border-left-color: var(--warn);}
.diag-head {display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;}
.diag-msg {font-weight: 600;}
.pill {font-size: .68rem; font-weight: 700; padding: 1px 9px; border-radius: 999px; white-space: nowrap;
       background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent) !important;}
.pill.error {background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad) !important;}
.pill.warning {background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn) !important;}
.loc {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; color: var(--muted) !important;}

/* agent tool log */
.tool-row {display: flex; align-items: baseline; gap: 10px; padding: 8px 12px; margin-bottom: 6px; border-radius: 10px;
           background: var(--surface2); font-size: .88rem; flex-wrap: wrap;}
.tool-row.rag {border-left: 3px solid var(--accent);}
.tool-name {font-weight: 700; white-space: nowrap;}
.tool-args {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .78rem; color: var(--muted) !important;}

/* how-it-works pipeline */
.flow {display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 8px 0 16px;}
.flow .node {background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 8px 14px; font-size: .86rem; font-weight: 600;}
.flow .node small {display: block; font-weight: 400; color: var(--muted) !important;}
.flow .arrow {color: var(--accent) !important; font-weight: 700;}
"""


def _inject_css(dark: bool):
    p = PALETTES["dark" if dark else "light"]
    variables = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in p.items())
    st.markdown(f"<style>:root{{{variables}}}{STATIC_CSS}</style>", unsafe_allow_html=True)


_inject_css(st.session_state["dark_mode"])


# ── Small HTML helpers (theme-safe: everything uses CSS variables) ──
def _e(x) -> str:
    return html.escape(str(x))


def _chip(text, state=""):
    return f'<span class="chip {state}">{_e(text)}</span>'


def _stats(items):
    cells = "".join(f'<div class="stat"><div class="stat-label">{_e(label)}</div>'
                    f'<div class="stat-value {cls}">{_e(value)}</div></div>' for label, value, cls in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def _callout(kind, text):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def _codebox(text, title=""):
    rows = "".join(f'<div class="ln">{_e(l) or "&nbsp;"}</div>' for l in (text or "").splitlines() or [""])
    head = f'<div class="codebox-title">{_e(title)}</div>' if title else ""
    st.markdown(f'{head}<div class="codebox">{rows}</div>', unsafe_allow_html=True)


def _diff_html(diff: str) -> str:
    rows = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            cls = "meta"
        elif line.startswith("@@"):
            cls = "hunk"
        elif line.startswith("+"):
            cls = "add"
        elif line.startswith("-"):
            cls = "del"
        else:
            cls = ""
        rows.append(f'<div class="ln {cls}">{_e(line) or "&nbsp;"}</div>')
    return f'<div class="codebox">{"".join(rows)}</div>'


# ── Helpers ───────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def _ollama_status():
    ok = check_ollama_available()
    return ok, (get_available_ollama_models() if ok else [])


def _kb_counts():
    """(code_chunks, saved_notes) in the vector store, or None if unavailable."""
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        return (vs.get_collection_stats("workspace_code").get("count", 0),
                vs.get_collection_stats("reference_docs").get("count", 0))
    except Exception:
        return None


def _do_index():
    """Re-index the workspace: clear the old project chunks, then embed the current files."""
    with st.spinner("Indexing workspace..."):
        try:
            from vector_store import VectorStore
            vs = VectorStore()
            vs.clear_collection("workspace_code")      # avoid stale chunks from another project
            result = vs.index_workspace(st.session_state.workspace_path)
            st.toast(f"Indexed {result.get('indexed', 0)} chunks from "
                     f"{result.get('total_documents', 0)} files", icon="✅")
            if result.get("errors", 0):
                st.sidebar.warning(f"{result['errors']} indexing errors (is `nomic-embed-text` pulled?)")
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            st.sidebar.error(f"Indexing failed: {e}")


def _provider_ready(ollama_ok: bool):
    provider = st.session_state.provider
    if provider == "ollama":
        return (True, "") if ollama_ok else (False, "Ollama is not running. Start it with `ollama serve`.")
    if not st.session_state.api_key:
        return False, f"Missing API key for {provider.title()}. Enter it in the sidebar."
    return True, ""


def _set_value(key, value):
    st.session_state[key] = value


def _use_sample_workspace():
    st.session_state["workspace_input"] = st.session_state["workspace_path"] = str(SAMPLE_WORKSPACE)


# ── Agent result rendering ────────────────────────────────
def _render_tool_log(history: list):
    rows = []
    for e in history:
        args = e.get("arguments", {})
        is_rag = e.get("tool") == "search_knowledge_base"
        badge = ("RAG · auto" if args.get("auto") else "RAG") if is_rag else "tool"
        arg_text = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items() if k != "auto")[:140]
        rows.append(f'<div class="tool-row {"rag" if is_rag else ""}">{"✅" if e.get("success") else "❌"}'
                    f'<span class="tool-name">{_e(e.get("tool", "?"))}</span><span class="pill">{badge}</span>'
                    f'<span class="tool-args">{_e(arg_text)}</span><span>{_e(e.get("result_summary", ""))}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def _display_agent_result(result: dict):
    if not result:
        return
    history = result.get("tool_history", [])
    sources = {(s.get("source"), s.get("lines")) for s in result.get("sources", [])}
    _stats([("Iterations", result.get("iterations", 0), ""), ("Time", f"{result.get('total_duration', 0)}s", ""),
            ("Tool calls", len(history), ""), ("RAG sources", len(sources), "")])
    if not result.get("success"):
        _callout("bad", f"<b>Failed.</b> {_e(result.get('error', 'Unknown error'))}")
    elif result.get("error"):
        _callout("warn", _e(result["error"]))
    if result.get("answer"):
        with st.container(border=True):
            st.markdown("#### 📊 Analysis")
            st.markdown(result["answer"])
    if history:
        with st.expander(f"🔧 What the agent did ({len(history)} steps)", expanded=True):
            _render_tool_log(history)
    if sources:
        with st.expander(f"📚 Retrieved sources ({len(sources)})"):
            for src, lines in sorted(sources, key=lambda x: (str(x[0]), str(x[1]))):
                st.markdown(f"- `{src}` (lines {lines})")


def _run_agent(task_type: str, query: str, label: str) -> dict:
    with st.status(label, expanded=True) as box:
        result = run_agent_loop(
            user_query=query, workspace_root=st.session_state.workspace_path,
            model=st.session_state.model, task_type=task_type,
            on_status=lambda msg: box.write(f"▸ {msg}"), provider=st.session_state.provider,
            api_key=st.session_state.api_key, base_url=st.session_state.base_url)
        ok = result.get("success")
        box.update(label="Done" if ok else "Failed", state="complete" if ok else "error", expanded=False)
    return result


# ── Patch approval ────────────────────────────────────────
def _save_note(patch: dict) -> bool:
    """Teach the knowledge base: store the applied fix so future runs can retrieve it."""
    try:
        from vector_store import VectorStore
        note = (f"Bug report: {st.session_state.last_fix_query[:600]}\n"
                f"File fixed: {patch['filepath']}\nApplied fix (diff):\n{patch['diff'][:1500]}")
        return bool(VectorStore().save_debugging_note(note, {"file": patch["filepath"], "type": "applied_fix"}))
    except Exception:
        return False


def _apply_patch(patch: dict):
    workspace = st.session_state.workspace_path
    new_content = patch.get("modified_content")
    if not new_content:
        patch["error"] = "This patch has no stored content. Ask the agent to propose it again."
        return
    base = patch.get("original_content")
    if base is not None:               # refuse if the file changed since the patch was proposed
        try:
            current = (Path(workspace) / patch["filepath"]).read_text(
                encoding="utf-8", errors="replace").replace("\r\n", "\n")
        except Exception:
            current = None
        if current is not None and current != base:
            patch["error"] = "The file changed after this patch was proposed. Run the fix again."
            return
    result = apply_patch(workspace, patch["filepath"], new_content, approved=True)
    if result.get("success"):
        patch["applied"], patch["backup"], patch["error"] = True, result.get("backup_path"), None
        patch["note_saved"] = _save_note(patch)
    else:
        patch["error"] = result.get("error", "Unknown error")


def _render_patches():
    for i, patch in enumerate(st.session_state.pending_patches):
        stats = patch.get("stats", {})
        syntax = patch.get("syntax")
        broken = bool(syntax) and not syntax.get("ok", True)
        with st.container(border=True):
            st.markdown(f"**Patch {i + 1} · `{patch['filepath']}`** &nbsp; "
                        f"`+{stats.get('additions', 0)}` `-{stats.get('deletions', 0)}`")
            st.markdown(_diff_html(patch["diff"]), unsafe_allow_html=True)
            if syntax:
                if broken:
                    _callout("bad", f"<b>Patched file does not compile</b> · {_e(syntax.get('message', ''))}")
                else:
                    _callout("ok", "<b>Compile check passed</b> · the patched file is syntactically valid")
            if patch.get("error"):
                _callout("bad", _e(patch["error"]))
            if patch.get("applied"):
                _callout("ok", f"<b>Applied.</b> Backup saved at <code>{_e(patch.get('backup'))}</code>")
                if patch.get("note_saved"):
                    st.caption("📚 Saved to the knowledge base, so the agent can retrieve this fix next time.")
            elif patch.get("rejected"):
                _callout("", "Rejected. No changes were made.")
            else:
                b1, b2, _ = st.columns([1.3, 1, 3])
                if b1.button("✅ Approve & apply", key=f"approve_{i}", type="primary", disabled=broken,
                             help="Disabled: the patched file has a syntax error" if broken else None):
                    _apply_patch(patch)
                    st.rerun()
                if b2.button("❌ Reject", key=f"reject_{i}"):
                    patch["rejected"] = True
                    st.rerun()


# ── Sidebar ───────────────────────────────────────────────
ollama_ok, ollama_models = _ollama_status()
counts = _kb_counts()

with st.sidebar:
    st.markdown("### 🛡️ CodePilot Local")
    st.caption("Private AI debugging agent")

    st.markdown("##### ⚙️ Engine")
    engines = {"🏠 Local (Ollama)": "ollama", "⚡ Groq Cloud": "groq", "🌐 OpenAI / OpenRouter": "openai"}
    engine = engines[st.selectbox("Inference engine", list(engines),
                                  index=list(engines.values()).index(st.session_state.provider),
                                  label_visibility="collapsed")]
    st.session_state.provider = engine

    if engine == "ollama":
        options = {d: m for d, m in AVAILABLE_MODELS.items() if m in ollama_models or not ollama_ok}
        if options:
            st.session_state.model = options[st.selectbox("Model", list(options))]
        else:
            st.warning("No compatible local models found")
            st.session_state.model = DEFAULT_MODEL
        st.caption("🟢 Ollama connected · code stays on this machine" if ollama_ok
                   else "🔴 Ollama not running. Start it with `ollama serve`")
    else:
        st.caption("⚠️ Cloud mode sends prompts and retrieved code to a third party.")
        env_key = "GROQ_API_KEY" if engine == "groq" else "OPENAI_API_KEY"
        st.session_state.api_key = st.text_input("API key", type="password",
                                                 value=st.session_state.api_key or os.environ.get(env_key, ""))
        if engine == "groq":
            st.session_state.base_url = "https://api.groq.com/openai/v1"
            models = CLOUD_GROQ_MODELS
        else:
            st.session_state.base_url = st.text_input(
                "Base URL", value=st.session_state.base_url or "https://api.openai.com/v1")
            models = CLOUD_OPENAI_MODELS
        st.session_state.model = models[st.selectbox("Model", list(models))]

    st.divider()
    st.markdown("##### 📂 Workspace")
    ws_input = st.text_input("Project folder", key="workspace_input", placeholder=r"C:\path\to\your\project")
    valid, msg = validate_workspace_path(ws_input) if ws_input else (False, "")
    if valid:
        st.session_state.workspace_path = str(Path(ws_input).expanduser().resolve())
        st.caption(f"🟢 {Path(st.session_state.workspace_path).name}")
    else:
        st.session_state.workspace_path = ""
        if msg:
            st.caption(f"🔴 {msg}")
    if SAMPLE_WORKSPACE.is_dir():
        st.button("Use the sample buggy project", on_click=_use_sample_workspace, use_container_width=True)

    st.divider()
    st.markdown("##### 🔍 Knowledge base (RAG)")
    _stats([("Code chunks", counts[0] if counts else 0, ""), ("Notes", counts[1] if counts else 0, "")])
    if st.button("📥 Index workspace", use_container_width=True, type="primary",
                 disabled=not st.session_state.workspace_path):
        _do_index()
        st.rerun()
    if counts is None:
        st.caption("Index unavailable. Check ChromaDB is installed and Ollama has `nomic-embed-text`.")

# ── Header ────────────────────────────────────────────────
local = st.session_state.provider == "ollama"
workspace = st.session_state.workspace_path
head_l, head_r = st.columns([5, 1.2], vertical_alignment="center")
with head_l:
    st.markdown(
        '<div class="brand"><span class="brand-title">🛡️ CodePilot</span>'
        '<span class="brand-sub">Investigate bugs, generate fixes, approve every change.</span></div>'
        '<div class="chips">' + "".join([
            _chip("Local · private" if local else "Cloud engine · code leaves this machine",
                  ("ok" if ollama_ok else "bad") if local else "warn"),
            _chip(f"Model: {st.session_state.model}"),
            _chip(f"Workspace: {Path(workspace).name}" if workspace else "No workspace", "ok" if workspace else "bad"),
            _chip(f"RAG index: {counts[0]} chunks" if counts else "RAG index unavailable",
                  "ok" if counts and counts[0] else "bad")]) + "</div>", unsafe_allow_html=True)
with head_r:
    st.toggle("🌙 Dark mode", key="dark_mode")

tabs = st.tabs(["🔍 Investigate", "🔧 Fix & Validate", "⚙️ Compiler", "🧭 How it works"])
tab_investigate, tab_fix, tab_compile, tab_how = tabs

EXAMPLE_INVESTIGATE = "Why does calculate_average in buggy_math.py return wrong results for a list like [2, 4, 6]?"
EXAMPLE_FIX = "Fix the bug in calculate_average in buggy_math.py so the unit tests pass."


def _task_inputs(prefix, query_label, example, error_label):
    st.text_area(query_label, key=f"{prefix}_query", height=96, placeholder="Describe the problem in plain English")
    st.text_area(error_label, key=f"{prefix}_error", height=70, placeholder="Traceback (most recent call last): ...")
    st.button("💡 Fill an example", key=f"{prefix}_example", on_click=_set_value, args=(f"{prefix}_query", example))


def _full_query(prefix, error_heading):
    query = st.session_state.get(f"{prefix}_query", "").strip()
    error = st.session_state.get(f"{prefix}_error", "").strip()
    return query + (f"\n\n{error_heading}:\n```\n{error}\n```" if error else "")


def _guard(prefix: str, empty_msg: str):
    """Return True when the task can start; otherwise show why not."""
    ready, err = _provider_ready(ollama_ok)
    if not st.session_state.get(f"{prefix}_query", "").strip():
        st.warning(empty_msg)
    elif not ready:
        st.error(err)
    else:
        return True
    return False


# ── Task 1: Investigate ───────────────────────────────────
with tab_investigate:
    st.markdown("#### Task 1 · Code investigation")
    st.caption("The agent retrieves relevant code with RAG, reads files, searches, and explains the "
               "root cause with file and line citations. It never changes your files.")
    if not workspace:
        st.warning("Select a workspace in the sidebar first.")
    else:
        _task_inputs("inv", "What should I investigate?", EXAMPLE_INVESTIGATE, "Error or traceback (optional)")
        if st.button("🔍 Start investigation", type="primary", key="btn_investigate") \
                and _guard("inv", "Describe the problem first."):
            st.session_state.results["investigate"] = _run_agent(
                "investigate", _full_query("inv", "Error/Traceback"), "Investigating...")
        _display_agent_result(st.session_state.results.get("investigate"))

# ── Task 2: Fix & Validate ────────────────────────────────
with tab_fix:
    st.markdown("#### Task 2 · Fix generation")
    st.caption("The agent finds the cause, proposes a minimal diff, and waits. Nothing is written "
               "until you approve; applying a patch saves a backup first.")
    if not workspace:
        st.warning("Select a workspace in the sidebar first.")
    else:
        _task_inputs("fix", "What should be fixed?", EXAMPLE_FIX, "Failing test or compile error (optional)")
        if st.button("🔧 Generate fix", type="primary", key="btn_fix") and _guard("fix", "Describe the bug first."):
            query = _full_query("fix", "Error details")
            st.session_state.last_fix_query = query
            result = _run_agent("fix", query, "Generating a fix...")
            st.session_state.results["fix"] = result
            st.session_state.pending_patches = result.get("proposed_patches", [])

        fix_result = st.session_state.results.get("fix")
        if fix_result:
            _display_agent_result(fix_result)
            if st.session_state.pending_patches:
                st.markdown("#### 📋 Proposed patches")
                _render_patches()
            elif fix_result.get("success"):
                st.info("The agent did not propose a patch. Read its analysis above, or rerun with "
                        "more detail about the failing function.")

# ── Compiler panel ────────────────────────────────────────
EXAMPLES = {
    "Syntax error": 'def average(numbers):\n    total = sum(numbers\n    return total / len(numbers)\n\nprint(average([2, 4, 6]))\n',
    "Undefined name": 'import os\n\ndef greet(name):\n    message = "Hello, " + nme\n    return message\n\nprint(greet("Ada"))\n',
    "Runtime error": 'def average(numbers):\n    return sum(numbers) / len(numbers)\n\nprint(average([]))\n',
}


def _workspace_py_files(root: str, limit: int = 300) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith("."))
        for f in sorted(filenames):
            if f.endswith(".py"):
                found.append(Path(dirpath, f).relative_to(root).as_posix())
        if len(found) >= limit:
            break
    return found[:limit]


def _compile_source(source_mode: str, run: bool):
    """Compile (and optionally run) the chosen source; store the outcome for rendering."""
    if source_mode == "Workspace file":
        rel = st.session_state.get("compile_file")
        if not rel:
            st.session_state.compile_notice = ("warn", "Pick a file first.")
            return
        name = rel
        source = (Path(workspace) / rel).read_text(encoding="utf-8", errors="replace")
    else:
        name, source = "snippet.py", st.session_state.get("compile_code", "")
        if not source.strip():
            st.session_state.compile_notice = ("warn", "Paste some code first.")
            return
    st.session_state.compile_notice = None
    check = check_python(source, name)
    ran = run_python(source, timeout=10, script_name=Path(name).name) if run and check["ok"] else None
    st.session_state.compile_result = {"name": name, "source": source, "check": check, "run": ran,
                                       "run_requested": run, "snippet": source_mode != "Workspace file"}


def _error_report(res: dict) -> str:
    parts = []
    if res["check"]["diagnostics"]:
        parts.append(render_report(res["check"]["diagnostics"], res["name"]))
    if res.get("run") and res["run"]["diagnostics"]:
        parts.append("Runtime error:\n" + render_report(res["run"]["diagnostics"], res["name"])
                     + ("\n\n" + res["run"]["stderr"].strip() if res["run"]["stderr"].strip() else ""))
    return "\n\n".join(parts)


def _send_to_fix():
    """Hand the compiler output to Task 2. A pasted snippet is first saved as a scratch file."""
    res = st.session_state.compile_result
    target = res["name"]
    if res["snippet"]:
        target = f"scratch_{hashlib.md5(res['source'].encode()).hexdigest()[:8]}.py"
        (Path(workspace) / target).write_text(res["source"], encoding="utf-8")
    st.session_state["fix_query"] = f"Fix the errors in {target} so it compiles and runs correctly."
    st.session_state["fix_error"] = _error_report(res)[:3000]
    saved = f"Saved the snippet as <code>{_e(target)}</code> in your workspace. " if res["snippet"] else ""
    st.session_state.compile_notice = ("ok", saved + "Errors sent. Open the <b>Fix &amp; Validate</b> tab and press <i>Generate fix</i>.")
    st.session_state.goto_tab = TAB_FIX


def _render_diagnostic(d: dict, source: str):
    loc = f"line {d['line']}" + (f", col {d['col']}" if d.get("col") else "")
    block = ""
    if d.get("text"):
        block = f'<div class="codebox"><div class="ln"><span class="gutter">{d["line"]}</span>{_e(d["text"])}</div>'
        if d.get("col"):
            width = max(1, (d["end_col"] - d["col"]) if d.get("end_col") and d.get("end_line") == d["line"] else 1)
            block += (f'<div class="ln caret"><span class="gutter"></span>{" " * (d["col"] - 1)}{"^" * min(width, 40)}</div>')
        block += "</div>"
    st.markdown(f'<div class="diag {d["severity"]}"><div class="diag-head">'
                f'<span class="pill {d["severity"]}">{d["severity"].upper()}</span>'
                f'<span class="pill">{_e(d["kind"])}</span><span class="loc">{loc}</span></div>'
                f'<div class="diag-msg">{_e(d["message"])}</div>{block}</div>', unsafe_allow_html=True)


def _render_compile_result(res: dict):
    check, ran = res["check"], res.get("run")
    run_errors = len(ran["diagnostics"]) if ran else 0
    errors, warns = check["errors"] + run_errors, check["warnings"]
    st.markdown("#### Result")
    if errors:
        _callout("bad", f"<b>Build failed</b> · {errors} error{'s' if errors != 1 else ''}, "
                        f"{warns} warning{'s' if warns != 1 else ''}")
    elif ran:
        _callout("ok", f"<b>Compiled and ran successfully</b> · {warns} warning{'s' if warns != 1 else ''}")
    else:
        _callout("ok", f"<b>Compiled successfully</b> · {warns} warning{'s' if warns != 1 else ''}. "
                       "Syntax and names are valid; use <i>Compile &amp; run</i> to catch runtime errors.")
    if not check["linter"]:
        st.caption("Install `pyflakes` (pip install pyflakes) for undefined-name and unused-import checks.")
    _stats([("Errors", errors, "bad" if errors else "ok"), ("Warnings", warns, "warn" if warns else ""),
            ("Lines", len(res["source"].splitlines()), ""), ("Run time", f"{ran['duration']}s" if ran else "–", "")])

    for d in check["diagnostics"]:
        _render_diagnostic(d, res["source"])
    if res["run_requested"] and not ran and not check["ok"]:
        st.caption("Skipped the run because the code does not compile.")
    if ran:
        st.markdown(f"##### Run output &nbsp; <span class='pill'>{_e(ran['mode'])}</span>", unsafe_allow_html=True)
        for d in ran["diagnostics"]:
            _render_diagnostic(d, res["source"])
        if ran["stdout"].strip():
            _codebox(ran["stdout"], "stdout")
        if ran["stderr"].strip():
            _codebox(ran["stderr"], "stderr")
        if ran["ok"] and not ran["stdout"].strip():
            st.caption("The program finished without printing anything.")
    if errors or warns:
        st.button("🔧 Send errors to Fix & Validate", key="btn_send_fix", on_click=_send_to_fix,
                  disabled=not workspace,
                  help=None if workspace else "Select a workspace first so the agent has files to patch.")


with tab_compile:
    st.markdown("#### Compiler panel")
    st.caption("Paste buggy code or pick a file, then compile it to get errors with line and column. "
               "Compiling never executes your code; **Compile & run** does (Docker if available, "
               "otherwise a temp folder with a 10 s timeout).")
    mode = st.radio("Source", ["Paste code", "Workspace file"], horizontal=True, label_visibility="collapsed",
                    key="compile_mode")
    if mode == "Paste code":
        st.text_area("Code", key="compile_code", height=260, label_visibility="collapsed",
                     placeholder="# Paste Python code here")
        ex_cols = st.columns(len(EXAMPLES) + 2)
        ex_cols[0].caption("Try an example:")
        for col, (label, code) in zip(ex_cols[1:], EXAMPLES.items()):
            col.button(label, key=f"ex_{label}", on_click=_set_value, args=("compile_code", code),
                       use_container_width=True)
    elif not workspace:
        st.warning("Select a workspace in the sidebar first.")
    else:
        files = _workspace_py_files(workspace)
        if files:
            st.selectbox("File", files, key="compile_file", label_visibility="collapsed")
        else:
            st.info("No Python files found in this workspace.")

    b1, b2, _ = st.columns([1, 1.3, 3])
    do_compile = b1.button("⚙️ Compile", type="primary", key="btn_compile")
    do_run = b2.button("▶️ Compile & run", key="btn_run")
    if do_compile or do_run:
        _compile_source(mode, run=do_run)
    if st.session_state.compile_notice:
        _callout(*st.session_state.compile_notice)
    if st.session_state.compile_result:
        _render_compile_result(st.session_state.compile_result)

# ── How it works ──────────────────────────────────────────
with tab_how:
    st.markdown("#### How CodePilot works")
    st.markdown(
        '<div class="flow">'
        '<div class="node">Project files</div><span class="arrow">→</span>'
        '<div class="node">Chunks<small>800 chars, overlap</small></div><span class="arrow">→</span>'
        '<div class="node">Embeddings<small>nomic-embed-text</small></div><span class="arrow">→</span>'
        '<div class="node">ChromaDB<small>code + notes</small></div><span class="arrow">→</span>'
        '<div class="node">Retrieve<small>top-k chunks</small></div><span class="arrow">→</span>'
        '<div class="node">Agent loop<small>LLM ⇄ tools</small></div><span class="arrow">→</span>'
        '<div class="node">Analysis or diff</div></div>', unsafe_allow_html=True)
    st.markdown("""
**RAG pipeline.** Project files are split into overlapping line-based chunks, embedded locally and stored
in ChromaDB (`workspace_code` for your code, `reference_docs` for saved debugging notes). Before the model
starts, your question is embedded and the closest chunks from both collections are added to the prompt. The
agent can call `search_knowledge_base` again whenever it needs more, and approved fixes are saved as notes so
the next similar bug retrieves them.

**Agentic task 1, Investigate.** The model repeatedly picks a tool (`list_files`, `read_file`, `search_code`,
`search_knowledge_base`, `get_file_metadata`, `compile_check`), reads the result, and decides the next step
until it can give a cited answer. Bounded to 10 iterations.

**Agentic task 2, Fix & Validate.** Same loop plus `propose_patch` and `run_tests`. Every proposed patch is
compile-checked automatically; a patch that does not compile cannot be approved. **Approve & apply** writes the
file only after saving a backup.

**Compiler panel.** Byte-compiles code for syntax errors, runs `pyflakes` for undefined names, optionally runs
the program to capture runtime tracebacks, and can hand the errors straight to Task 2.

**Safety.** Paths are confined to the workspace, tests and runs have timeouts (Docker when available), the
run sandbox does not inherit your API keys, and every tool call is written to `audit_log.jsonl`.
""")

# ── Best-effort: jump to a tab after "send to agent" ──────
_goto = st.session_state.pop("goto_tab", None)
if _goto is not None:
    try:    # needs a recent Streamlit; on older versions the on-screen hint still tells the user where to go
        st.html(f"<script>const t=document.querySelectorAll('button[role=tab]');"
                f"if(t[{int(_goto)}])t[{int(_goto)}].click();</script>", unsafe_allow_javascript=True)
    except Exception:
        pass
