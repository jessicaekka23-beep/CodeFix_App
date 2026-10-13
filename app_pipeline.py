"""
CodePilot Local - pipeline mode.

One prompt, then watch the whole process in one place:
  Step 0  Index        - (once per project) cut your code into chunks, embed them, store them for RAG
  Step 1  Investigate  - Agent 1 retrieves context, then reads/searches until it finds the root cause (changes nothing)
  Step 2  Fix          - Agent 2 starts from step 1's findings and proposes a patch
  Step 3  Review       - you approve or reject the patch

Run with:  streamlit run app_pipeline.py
"""
import os
import time
from pathlib import Path

import streamlit as st

try:                                    # optional: load GROQ_API_KEY etc. from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="CodePilot Local", layout="wide")

from config import (
    AGENT_MAX_ITERATIONS, AGENT_MAX_TOOL_CALLS, AVAILABLE_MODELS, CLOUD_GROQ_MODELS, CLOUD_OPENAI_MODELS, DEFAULT_MODEL,
    EMBEDDING_MODEL, MAX_FIX_ATTEMPTS, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_TOP_K,
    check_ollama_available, get_available_ollama_models, validate_workspace_path,
)
from demo_data import STAGE_LABELS, STAGES, IndexRun, project_point
from pipeline_runner import STEPS, PipelineRun
from tools import apply_patch
from xp_theme import THEME_ON, hero_html, progress_html, taskbar_html, theme_css, title_bar
from xp_widgets import chunk_bars_html, flow_html, map_svg, seed_html, table_html, trace_html, vector_strip_html

SAMPLE_WORKSPACE = Path(__file__).resolve().parent / "test_workspace"

# Ready-made problems for the bundled test_workspace/. "Run sample" cycles through them.
SAMPLES = [
    ("calculate_average is off", "Why does calculate_average in buggy_math.py return wrong results for [2, 4, 6]? Fix it so the unit tests pass."),
    ("find_max returns the minimum", "find_max in buggy_math.py returns the smallest item instead of the largest. Find the cause and fix it."),
    ("count_words crashes on None", "count_words in buggy_strings.py raises an error when it is given None. Investigate and propose a safe fix."),
    ("reverse_words reverses letters", "reverse_words in buggy_strings.py reverses the characters of the whole sentence instead of the order of the words. Fix it."),
    ("fibonacci breaks on 0 and 1", "fibonacci in buggy_math.py gives wrong results for n=0 and n=1. Find out why and fix it."),
]

if THEME_ON:                                    # retro look: CSS only. Set CODEPILOT_NO_THEME=1 to switch it off.
    st.markdown(theme_css(bool(st.session_state.get("dark_mode"))), unsafe_allow_html=True)


def _win(key):
    """A themed window container. Falls back to a plain container if this Streamlit can't take key=."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def _bar(text, icon="app", flat=False):
    """Window title bar (themed) or a plain subheader when the theme is off."""
    if THEME_ON:
        st.markdown(title_bar(text, icon, flat=flat, buttons=not flat), unsafe_allow_html=True)
    else:
        st.subheader(text)

for k, v in {"provider": "ollama", "model": DEFAULT_MODEL, "api_key": "", "base_url": "",
             "workspace_input": str(SAMPLE_WORKSPACE) if SAMPLE_WORKSPACE.is_dir() else "",
             "patches": []}.items():
    st.session_state.setdefault(k, v)


@st.cache_data(ttl=10, show_spinner=False)
def _ollama_status():
    ok = check_ollama_available()
    return ok, (get_available_ollama_models() if ok else [])


@st.cache_data(ttl=5, show_spinner=False)
def _kb_count():
    """Chunks currently in the project index, or None if the vector store is unavailable."""
    try:
        from vector_store import VectorStore
        return VectorStore().get_collection_stats("workspace_code").get("count", 0)
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=32)
def _embed_text(text):
    """Embed a query the same way the index embedded the chunks (for placing it on the map)."""
    try:
        import rag
        return rag.generate_embedding(text)
    except Exception:
        return None


# ── Sidebar: engine + workspace ───────────────────────────
ollama_ok, ollama_models = _ollama_status()
with st.sidebar:
    _bar("Control Panel", "app", flat=True)
    if THEME_ON:
        st.toggle("Dark theme", key="dark_mode")
    st.subheader("Engine")
    engines = {"Local (Ollama)": "ollama", "Groq Cloud": "groq", "OpenAI / OpenRouter": "openai"}
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
        st.caption("Ollama connected" if ollama_ok else "Ollama not running. Start it with `ollama serve`")
    else:
        st.caption("Cloud mode sends prompts and retrieved code to a third party.")
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
    st.subheader("Workspace")
    ws_input = st.text_input("Project folder", key="workspace_input")
    valid, msg = validate_workspace_path(ws_input) if ws_input else (False, "")
    workspace = str(Path(ws_input).expanduser().resolve()) if valid else ""
    st.caption(f"Using {Path(workspace).name}" if workspace else (msg or "Enter a project folder"))
    st.caption(f"Embeddings always come from local Ollama (`{EMBEDDING_MODEL}`), even with a cloud engine.")


# ── Helpers ───────────────────────────────────────────────
def _provider_ready():
    if st.session_state.provider == "ollama":
        return (True, "") if ollama_ok else (False, "Ollama is not running. Start it with `ollama serve`.")
    if not st.session_state.api_key:
        return False, "Missing API key. Enter it in the sidebar."
    return True, ""


def _index_ready():
    """True if there is an index we can trust for this workspace (built here, or left over from an earlier session)."""
    idx = st.session_state.get("idx")
    if idx is not None:                                  # built in this session: only good for the folder it was built from
        return idx.workspace == workspace and (idx.ok() or idx.alive())
    return bool(_kb_count())


def _load_sample(run_it):
    """Button callback: put the next sample problem in the prompt box (and point at the sample project)."""
    i = st.session_state.get("sample_i", 0) % len(SAMPLES)
    st.session_state.sample_i = i + 1
    st.session_state.sample_label = SAMPLES[i][0]
    st.session_state.workspace_input = str(SAMPLE_WORKSPACE)
    st.session_state.prompt = SAMPLES[i][1]
    st.session_state.autorun = run_it
    st.session_state.pop("run", None)


def _begin_index():
    st.session_state.probe = None
    st.session_state.idx = IndexRun(workspace).start()


def _start_run(prompt_text, skip_index=False):
    """Validate and start the pipeline. Returns an error message, or None after starting (caller reruns).

    If nothing is indexed yet (and the theme's Index window is available) the index is built first, where you can
    watch it, and the run starts when it finishes.
    """
    ready, err = _provider_ready()
    if not prompt_text.strip():
        return "Write a prompt first."
    if not ready:
        return err
    if THEME_ON and not skip_index and _kb_count() is not None and not _index_ready():
        st.session_state.pending_run = prompt_text.strip()
        st.session_state.patches = []
        st.session_state.pop("run", None)
        _begin_index()
        return None
    st.session_state.patches = []
    st.session_state.run = PipelineRun(
        prompt_text.strip(), workspace, st.session_state.model, provider=st.session_state.provider,
        api_key=st.session_state.api_key, base_url=st.session_state.base_url).start()
    return None


def _live_text(run, step):
    bar = (progress_html() + "\n\n") if THEME_ON else ""
    events = list(run.events[step])
    if THEME_ON and events:
        body = trace_html(events, running=True)
    else:
        body = "```\n" + ("\n".join(f"> {m}" for m in list(run.logs[step])[-30:]) or "starting...") + "\n```"
    return f"{bar}**Running** - {run.elapsed(step):.0f}s\n\n{body}"


def _waiting_text(run):
    return "Skipped." if run.stopped else "Waiting for the investigation to finish..."


def _show_result(run, step):
    r = run.results[step]
    history = r.get("tool_history", [])
    events = list(run.events[step])
    if r.get("success"):
        st.success(f"Done in {run.elapsed(step):.0f}s - {r.get('iterations', 0)} iterations, {len(history)} tool calls")
    else:
        st.error(f"Failed: {r.get('error') or 'unknown error'}")
    if r.get("success") and r.get("error"):
        st.warning(r["error"])
    if THEME_ON and events:
        with st.expander(f"Agent trace ({len(events)} events): retrieval, thinking, tool calls", expanded=True):
            st.markdown(trace_html(events), unsafe_allow_html=True)
    if r.get("answer"):
        st.markdown(r["answer"])
    if history and not (THEME_ON and events):
        with st.expander(f"Steps the agent took ({len(history)})"):
            for e in history:
                st.write(f"{'OK' if e.get('success') else 'FAIL'} `{e.get('tool', '?')}` {e.get('result_summary', '')}")
    if step == "fix" and run.queries.get("fix"):
        with st.expander("What the Fix agent was told (your prompt + the investigation findings)"):
            st.code(run.queries["fix"], language=None)


def _apply(i, workspace_root):
    """Approve & apply one patch (same safety checks as the main app: refuse if the file changed)."""
    patch = st.session_state.patches[i]
    new_content = patch.get("modified_content")
    if not new_content:
        patch["error"] = "This patch has no stored content. Run again."
        return
    base = patch.get("original_content")
    if base is not None:
        try:
            current = (Path(workspace_root) / patch["filepath"]).read_text(
                encoding="utf-8", errors="replace").replace("\r\n", "\n")
        except Exception:
            current = None
        if current is not None and current != base:
            patch["error"] = "The file changed after this patch was proposed. Run again."
            return
    result = apply_patch(workspace_root, patch["filepath"], new_content, approved=True)
    if result.get("success"):
        patch["applied"], patch["backup"], patch["error"] = True, result.get("backup_path"), None
    else:
        patch["error"] = result.get("error", "Unknown error")


def _reject(i):
    st.session_state.patches[i]["rejected"] = True


def _show_patches(workspace_root):
    for i, patch in enumerate(st.session_state.patches):
        stats = patch.get("stats", {})
        syntax = patch.get("syntax")
        broken = bool(syntax) and not syntax.get("ok", True)
        with st.container(border=True):
            st.markdown(f"**Patch {i + 1}: `{patch['filepath']}`** (+{stats.get('additions', 0)} / -{stats.get('deletions', 0)})")
            st.code(patch["diff"], language="diff")
            if broken:
                st.error(f"Patched file does not compile: {syntax.get('message', '')}")
            elif syntax:
                st.success("Compile check passed")
            if patch.get("error"):
                st.error(patch["error"])
            if patch.get("applied"):
                st.success(f"Applied. Backup saved at {patch.get('backup')}")
            elif patch.get("rejected"):
                st.info("Rejected. No changes were made.")
            else:
                c1, c2, _ = st.columns([1.3, 1, 3])
                c1.button("Approve & apply", key=f"approve_{i}", type="primary", disabled=broken,
                          on_click=_apply, args=(i, workspace_root))
                c2.button("Reject", key=f"reject_{i}", on_click=_reject, args=(i,))


# ── Demo views: indexing, retrieval, map ──────────────────
def _index_nodes(idx):
    """Stage strip for an IndexRun: Load -> Chunk -> Embed -> Store -> Map."""
    subs = {
        "load": f"{len(idx.docs)} files" if idx.docs else "",
        "chunk": f"{len(idx.chunks)} chunks" if idx.chunks else "",
        "embed": f"{idx.embedding['dim']} numbers each" if idx.embedding else "",
        "store": (f"{idx.stats.get('indexed', 0)} new, {idx.stats.get('skipped_duplicates', 0)} reused" if idx.stats else ""),
        "map": "2-D view" if idx.map else "",
    }
    failed = STAGES[len(idx.done_stages)] if (idx.ended and idx.error and len(idx.done_stages) < len(STAGES)) else None
    nodes = []
    for name in STAGES:
        state = "done" if name in idx.done_stages else "active" if idx.stage == name else "bad" if name == failed else "idle"
        nodes.append((STAGE_LABELS[name], subs[name] or ("working..." if state == "active" else "-"), state))
    return nodes


def _live_index_text(idx):
    return (f"{progress_html()}\n\n**Indexing** - {idx.elapsed():.0f}s"
            + (" (embedding every chunk with Ollama is the slow part)" if idx.stage == "store" else "")
            + "\n\n" + flow_html(_index_nodes(idx)))


def _probe_overlay(idx):
    """What to draw on the map: the user's own search if they made one, else the last run's retrieval."""
    probe = st.session_state.get("probe")
    if probe:
        return probe["xy"], [h["chunk_id"] for h in probe["hits"]], "your search"
    run = st.session_state.get("run")
    if run:
        for ev in run.events["investigate"]:
            if ev.get("type") == "seed":
                vec = _embed_text(ev.get("query", ""))
                xy = project_point(idx.map, vec) if vec else None
                ids = [h["chunk_id"] for h in ev.get("hits", []) if h.get("collection") == "workspace_code"]
                return xy, ids, "the retrieval your last run did"
    return None, [], ""


def _run_probe(query):
    """Search the real index the way the agent does (same tool, same collection) and remember the result."""
    from tools import execute_tool
    res = execute_tool("search_knowledge_base", {"query": query, "collection": "workspace_code", "top_k": RAG_TOP_K}, workspace)
    hits = [{"collection": "workspace_code", "label": "project code", "source": h.get("source"), "chunk_id": h.get("chunk_id"),
             "start": h.get("start_line"), "end": h.get("end_line"), "distance": h.get("distance"), "text": (h.get("text") or "")[:700]}
            for h in res.get("results", [])]
    vec = _embed_text(query)
    st.session_state.probe = {"query": query, "hits": hits, "error": None if res.get("success") else res.get("error"),
                              "xy": project_point(st.session_state.idx.map, vec) if vec else None}


def _show_index(idx):
    st.markdown(flow_html(_index_nodes(idx)), unsafe_allow_html=True)
    if idx.error:
        (st.warning if idx.stats else st.error)(idx.error)
    if not idx.docs:
        return
    t_files, t_chunks, t_embed, t_store, t_map = st.tabs(["1 Files", "2 Chunks", "3 Embedding", "4 Stored", "5 Map + search"])

    with t_files:
        st.caption("The indexer walks the project folder and loads every supported source file (skipping folders like .git, venv and node_modules, and files over 1 MB).")
        st.markdown(table_html(["file", "type", "lines", "characters"],
                               [(d["file"], d["ext"], d["lines"], d["chars"]) for d in sorted(idx.docs, key=lambda d: d["file"])]),
                    unsafe_allow_html=True)

    with t_chunks:
        if not idx.chunks:
            st.caption("Not reached yet.")
        else:
            avg = sum(c["chars"] for c in idx.chunks) // max(len(idx.chunks), 1)
            st.caption(f"Whole files are too big to search or paste into a prompt, so each is cut into overlapping pieces of about "
                       f"{RAG_CHUNK_SIZE} characters (overlap {RAG_CHUNK_OVERLAP}), always on line boundaries so code stays readable. "
                       f"{len(idx.chunks)} chunks here, {avg} characters on average.")
            files = sorted({c["file"] for c in idx.chunks})
            chosen = st.selectbox("File", files, key="idx_chunk_file")
            cs = [c for c in idx.chunks if c["file"] == chosen]
            total = next((d["lines"] for d in idx.docs if d["file"] == chosen), max(c["end"] for c in cs))
            st.markdown(chunk_bars_html(cs, total), unsafe_allow_html=True)
            n = st.select_slider("Preview chunk", options=list(range(1, len(cs) + 1)), key=f"idx_chunk_n_{chosen}") if len(cs) > 1 else 1
            c = cs[n - 1]
            st.caption(f"{c['file']} lines {c['start']}-{c['end']} ({c['chars']} characters). This exact text is what gets embedded and later retrieved.")
            st.code(c["text"], language="python")

    with t_embed:
        e = idx.embedding
        if not e:
            st.caption("Not reached yet.")
        else:
            st.caption(f"An embedding model (`{EMBEDDING_MODEL}`, running in Ollama) reads a chunk and outputs a list of {e['dim']} numbers, "
                       "a fingerprint of what the code is about. Here is one real chunk and its fingerprint:")
            st.caption(f"{e['source']} lines {e['lines']}")
            st.code(e["text"], language="python")
            st.markdown(vector_strip_html(e["values"]), unsafe_allow_html=True)
            st.caption(f"First 96 of {e['dim']} numbers (pink = positive, blue = negative, stronger colour = bigger value). "
                       f"Took {e['seconds']:.2f}s. First values: " + ", ".join(f"{v:.3f}" for v in e["values"][:8]) + " ...")
            st.info("Chunks about similar things get similar numbers. That is how a question like \"why is the average wrong?\" "
                    "can find `calculate_average` without sharing any exact words with it.")

    with t_store:
        if not idx.stats:
            st.caption("Not reached yet.")
        else:
            from config import CHROMA_PERSIST_DIR
            st.caption("Each chunk is saved in ChromaDB (a local vector database) together with its vector and its file and line range. "
                       "Its id is a hash of the file name plus the text, so re-indexing skips anything unchanged.")
            rows = [("files loaded", len(idx.docs)), ("chunks cut", len(idx.chunks)), ("newly stored", idx.stats.get("indexed", 0)),
                    ("already there (skipped)", idx.stats.get("skipped_duplicates", 0)), ("errors", idx.stats.get("errors", 0)),
                    ("chunks in the index now", idx.count if idx.count is not None else "?"), ("saved in", str(CHROMA_PERSIST_DIR))]
            rows += [(f"time: {STAGE_LABELS[s].lower()}", f"{t:.2f}s") for s, t in idx.timings.items()]
            st.markdown(table_html(["what", "value"], rows), unsafe_allow_html=True)

    with t_map:
        if not idx.map:
            st.caption("The map needs numpy and at least two stored chunks.")
        else:
            st.caption("Every stored chunk squashed from its numbers down to a point on a 2-D map (PCA). Colour = file. Chunks that mean similar "
                       "things tend to sit close together. Search below, or run a task, to see where a question lands and which chunks it pulls in.")
            q = st.text_input("Search the index like the agent does", key="probe_q",
                              placeholder="e.g. why is the average calculated wrongly?")
            if st.button("Search", key="probe_go", disabled=not q.strip()):
                _run_probe(q.strip())
            probe = st.session_state.get("probe")
            if probe and probe.get("error"):
                st.error(probe["error"])
            xy, ids, source = _probe_overlay(idx)
            st.markdown(map_svg(idx.map["points"], idx.map_meta, xy, ids), unsafe_allow_html=True)
            if ids:
                st.caption(f"Showing {source}: rings are the chunks it retrieved, numbered by closeness.")
            if probe and probe.get("hits"):
                st.markdown(seed_html({"query": probe["query"], "hits": probe["hits"]}), unsafe_allow_html=True)


def _flow_nodes(run, idx):
    """The pipeline at a glance: Index -> Retrieve -> Investigate -> Fix -> Review."""
    kb = _kb_count()
    if idx is not None and idx.alive():
        index = ("Index", "building...", "active")
    elif idx is not None and idx.ok():
        index = ("Index", f"{idx.count} chunks", "done")
    elif idx is not None and idx.error and not idx.stats:
        index = ("Index", "failed", "bad")
    elif kb:
        index = ("Index", f"{kb} chunks (earlier)", "done")
    else:
        index = ("Index", "not built yet", "idle")
    retrieve, inv, fix = ("Retrieve", "-", "idle"), ("Investigate", "-", "idle"), ("Fix", "-", "idle")
    if run:
        seeds = [e for e in run.events["investigate"] if e.get("type") == "seed"]
        if seeds:
            retrieve = ("Retrieve", f"{len(seeds[0].get('hits') or [])} chunks", "done")
        elif run.running("investigate"):
            retrieve = ("Retrieve", "searching index...", "active")
        nodes = {}
        for name, step in (("Investigate", "investigate"), ("Fix", "fix")):
            if run.done(step):
                r = run.results[step]
                nodes[step] = (name, f"{r.get('iterations', 0)} loops, {len(r.get('tool_history', []))} tools", "done" if r.get("success") else "bad")
            elif run.running(step):
                nodes[step] = (name, "thinking...", "active")
            elif run.stopped and step == "fix":
                nodes[step] = (name, "skipped", "bad")
            else:
                nodes[step] = (name, "waiting", "idle")
        inv, fix = nodes["investigate"], nodes["fix"]
    patches = st.session_state.get("patches") or []
    if patches:
        pending = [p for p in patches if not (p.get("applied") or p.get("rejected"))]
        review = ("Review", f"{len(pending)} waiting for you" if pending else "decided", "active" if pending else "done")
    else:
        review = ("Review", "-", "idle")
    return [index, retrieve, inv, fix, review]


def _explainer():
    st.markdown(
        f"""
**1. Index (once per project).** Your code is cut into chunks of about {RAG_CHUNK_SIZE} characters (overlap {RAG_CHUNK_OVERLAP}). `{EMBEDDING_MODEL}` turns each chunk into a vector (a list of numbers capturing its meaning), and ChromaDB stores them on disk.

**2. Retrieve (RAG).** Before the agent's first LLM call, your prompt is embedded the same way. ChromaDB returns the closest chunks (up to 3 from your code, 2 from notes of past fixes) and they are pasted into the agent's prompt. The model answers with real code in front of it instead of guessing.

**3. Agent 1, Investigate.** A loop of at most {AGENT_MAX_ITERATIONS} rounds and {AGENT_MAX_TOOL_CALLS} tool calls. Each round the model replies with either a `TOOL_CALL` (list files, read a file, search code, search the index, compile-check...) or a `FINAL_ANSWER`. The app runs the tool and feeds the result back. It is prompted to diagnose only.

**4. Agent 2, Fix.** Same loop, different instructions: it receives your prompt plus Agent 1's findings, reads the file, proposes a diff with `propose_patch` (checked to compile), can run the tests, and may revise up to {MAX_FIX_ATTEMPTS} times.

**5. Review.** Nothing is written to your files by an agent. You read the diff and click Approve (a backup is saved first) or Reject.
""")


# ── Main ──────────────────────────────────────────────────
_status = ("Local" if st.session_state.provider == "ollama" else "Cloud") + f" - {st.session_state.model}"
if THEME_ON:
    st.markdown(taskbar_html(_status), unsafe_allow_html=True)         # fixed strip, never clickable
    st.markdown(hero_html([
        ("Local - private" if st.session_state.provider == "ollama" else "Cloud engine - code leaves this machine",
         ("ok" if ollama_ok else "bad") if st.session_state.provider == "ollama" else ""),
        (f"Model: {st.session_state.model}", ""),
        (f"Workspace: {Path(workspace).name}" if workspace else "No workspace", "ok" if workspace else "bad"),
    ]), unsafe_allow_html=True)
else:
    st.title("CodePilot Local")
    st.caption("Describe the bug once. Watch it get investigated, then fixed.")

run = st.session_state.get("run")
idx = st.session_state.get("idx")
busy = bool((run and run.alive()) or (idx and idx.alive()))

if idx is not None and not idx.alive() and st.session_state.get("pending_run"):   # auto-index finished: start the queued run
    queued = st.session_state.pop("pending_run")
    problem = _start_run(queued, skip_index=True)
    if problem:
        st.error(problem)
    else:
        st.rerun()

if THEME_ON:
    st.markdown(flow_html(_flow_nodes(run, idx)), unsafe_allow_html=True)
    with st.expander("How does this work? (RAG + two agents)"):
        _explainer()

index_slot = None
if THEME_ON:
    with _win("xpwin_index"):
        _bar("Step 0 - Index (RAG)", "search")
        st.caption("Builds the searchable memory of your project. Watch the code get cut up, turned into vectors and stored. "
                   "If nothing is indexed when you start a task, this runs first automatically.")
        if st.button("Re-index workspace" if idx else "Index workspace", disabled=busy or not workspace, key="index_go"):
            _begin_index()
            st.rerun()
        index_slot = st.empty()
        if idx is not None and idx.workspace != workspace and not idx.alive():
            index_slot.caption("The last index was built for a different folder. Re-index to see this project.")
        elif idx is not None and idx.alive():
            index_slot.markdown(_live_index_text(idx), unsafe_allow_html=True)
        elif idx is not None:
            with index_slot.container():
                _show_index(idx)
        else:
            kb = _kb_count()
            index_slot.caption(f"{kb} chunks are already stored from an earlier session. Click Index to rebuild and see how it works."
                               if kb else ("The index is empty." if kb == 0 else "The vector store is unavailable (is chromadb installed?)."))

with _win("xpwin_task"):
    _bar("New task", "app")
    prompt = st.text_area("Describe the bug (paste a traceback too if you have one)", key="prompt", height=110,
                          placeholder="Why does calculate_average in buggy_math.py return wrong results for [2, 4, 6]?")
    can_sample = SAMPLE_WORKSPACE.is_dir()
    b1, b2, b3, b4 = st.columns([1.3, 1.1, 1.1, 1])
    start_error = None
    if b1.button("Investigate & fix", type="primary", disabled=busy or not workspace):
        start_error = _start_run(prompt)
        if not start_error:
            st.rerun()
    b2.button("Run sample", disabled=busy or not can_sample, on_click=_load_sample, args=(True,),
              help="Fills in a sample bug from the bundled test project and runs it straight away (indexing first if needed).")
    b3.button("Load sample", disabled=busy or not can_sample, on_click=_load_sample, args=(False,),
              help="Fills in a sample bug so you can read or edit it first. Click again for the next one.")
    if run and not busy:
        b4.button("Clear", on_click=lambda: st.session_state.pop("run", None))
    if st.session_state.get("sample_label"):
        st.caption(f"Sample: {st.session_state.sample_label} (project: test_workspace)")
    if st.session_state.pop("autorun", False) and not busy:       # "Run sample" was clicked
        start_error = _start_run(st.session_state.get("prompt", "")) if workspace else "Sample project folder not found."
        if not start_error:
            st.rerun()
    if start_error:
        (st.warning if start_error.startswith("Write") else st.error)(start_error)

if run:
    slots = {}
    with _win("xpwin_investigate"):
        _bar("Step 1 - Investigate (Agent 1)", "search")
        st.caption("The detective. First the index is searched for relevant code, then the model loops: think, call a tool, read the result. "
                   "It is prompted to diagnose only and ends with the root cause, file and line citations.")
        slots["investigate"] = st.empty()
    with _win("xpwin_fix"):
        _bar("Step 2 - Fix (Agent 2)", "wrench")
        st.caption("The mechanic. Starts from your prompt plus the findings above, reads the file, proposes a minimal patch, checks that it "
                   f"compiles and can run the tests (up to {MAX_FIX_ATTEMPTS} revisions). Nothing is written yet.")
        slots["fix"] = st.empty()
    with _win("xpwin_review"):
        _bar("Step 3 - Review", "check")
        st.caption("You decide. Applying a patch saves a backup first.")
        review = st.empty()                           # st.empty() replaces its content; st.container() would append

    if run.alive():                                   # live view: poll the background pipeline
        review.caption("Waiting for the fix to be proposed...")      # set once, not inside the loop
        shown = set()
        while run.alive():
            for step in STEPS:
                if run.done(step):
                    if step not in shown:             # draw each finished step once
                        with slots[step].container():
                            _show_result(run, step)
                        shown.add(step)
                elif run.running(step):
                    slots[step].markdown(_live_text(run, step), unsafe_allow_html=True)
                else:
                    slots[step].caption(_waiting_text(run))
            time.sleep(0.4)
        st.rerun()                                    # finished: redraw through the normal path below
    else:
        for step in STEPS:
            with slots[step].container():
                if run.done(step):
                    _show_result(run, step)
                else:
                    st.caption(_waiting_text(run))
        if run.stopped:
            st.warning(run.stopped)
        if not run.patches_taken:                     # hand the Fix agent's patches to the approval list
            st.session_state.patches = list((run.results.get("fix") or {}).get("proposed_patches") or [])
            run.patches_taken = True
            st.rerun()
        with review.container():
            if st.session_state.patches:
                _show_patches(run.workspace_root)
            elif (run.results.get("fix") or {}).get("success"):
                st.info("The Fix agent did not propose a patch. Read the analysis above, or add detail to the prompt.")
            else:
                st.caption("Nothing to review.")

if index_slot is not None and idx is not None and idx.alive():      # live view of the indexer (the run loop above never overlaps it)
    while idx.alive():
        index_slot.markdown(_live_index_text(idx), unsafe_allow_html=True)
        time.sleep(0.4)
    st.rerun()
