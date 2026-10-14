# 🛡️ CodePilot Local

A privacy-first AI debugging assistant. It uses **RAG** over your own Python project to power two
agentic tasks, and ships with a compiler panel for checking code.

| | |
|---|---|
| **Task 1 · Investigate** | The agent retrieves relevant code (ChromaDB + embeddings), reads files, searches, and explains the root cause with file and line citations. Never edits files. |
| **Task 2 · Fix & Validate** | The agent proposes a minimal diff, compile-checks it, and can run your tests. Nothing is written until you click **Approve**; a backup is saved first, and approved fixes are stored as notes the agent retrieves next time. |
| **Compiler panel** | Paste code or pick a workspace file, then **Compile** for syntax errors and undefined names (with line/column and a caret), or **Compile & run** to catch runtime errors. One click sends the errors to Task 2. |

Light and dark mode are one toggle in the header. The pipeline app (`app_pipeline.py`) has a pixel-art pastel Windows-XP look with its own **Dark theme** switch in the sidebar, plus **Run sample** / **Load sample** buttons that fill in a bug from `test_workspace/` (and run it).

**Demo mode (`app_pipeline.py`)** shows what is happening under the hood: a flow strip (Index → Retrieve → Investigate → Fix → Review); a **Step 0 Index** window that walks through loading files, cutting chunks, a real embedding vector, what gets stored in ChromaDB, and a 2-D map of your code where you can search the index and see which chunks a question lands near; the exact chunks retrieved for each run (with distances); and a live trace of each agent (what the model decided, each tool call and its result). If nothing is indexed when you start a task, it indexes first automatically. The trace comes from an optional `on_event` callback on `run_agent_loop` (`app.py` does not use it). Inference is local (Ollama) by default; Groq and
OpenAI-compatible engines are optional. Cloud mode sends prompts and retrieved code to that provider.

## Quick start

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text      # optional: embeddings (falls back to a built-in embedder)
ollama pull qwen2.5-coder:7b      # default local model
streamlit run app.py              # http://localhost:8501
```

Embeddings work with or without Ollama. Generation needs a model, but indexing and retrieval do not: if
`ollama serve` is unreachable (for example on Streamlit Community Cloud) the app falls back to a
dependency-free hashing embedder, so RAG still works with no extra services or model downloads. Force a
back-end with `CODEPILOT_EMBED_BACKEND=ollama` or `CODEPILOT_EMBED_BACKEND=hash`.

Optional: copy `.env.example` to `.env` and add `GROQ_API_KEY`, or paste the key in the sidebar.
Docker is optional; when present, tests and "Compile & run" execute in a disposable container
(no network, memory-limited, read-only mount).

**Try it:** the sidebar's "Use the sample buggy project" loads `test_workspace/` (7 intentional bugs).
Click **Index workspace**, then use the example buttons in either task tab. In the Compiler tab, the
example buttons load a syntax error, an undefined name, and a runtime error.

## How the compiler panel works

Python has no separate compile step, so `compiler.py` does what a compiler front end does:

1. **Compile** byte-compiles the source without executing it → `SyntaxError` / `IndentationError` with line and column.
2. **Static check** runs `pyflakes` → undefined names (errors), unused imports/variables (warnings).
3. **Compile & run** (optional) executes the code in a temp folder with a 10 s timeout and no inherited
   environment variables (so API keys are not exposed) → runtime tracebacks mapped to a line.

The agent also has a `compile_check` tool, and every `propose_patch` result carries a `syntax` check:
a patch that does not compile cannot be approved in the UI.

## Project layout

```
app.py            Streamlit UI (3 task panels + how-it-works, light/dark theme)
app_pipeline.py   Pipeline UI: index -> retrieve -> investigate -> fix -> review, with the demo views
demo_data.py      Instrumented indexing + 2-D embedding map for the demo views
embeddings.py     Embedding back-end: Ollama when reachable, else a built-in hashing embedder
xp_theme.py       Pixel-art XP theme (CSS) | xp_widgets.py: HTML/SVG builders for the demo views
agent.py          Bounded tool-using agent loop, prompts, Ollama/Groq/OpenAI clients
tools.py          Validated workspace tools: read/search/metadata, compile_check, run_tests, patches, audit log
compiler.py       Syntax check, static analysis, sandboxed run, compiler-style reports
rag.py            Document loading, chunking, and Ollama batch embeddings
vector_store.py   ChromaDB collections (workspace_code, reference_docs) with dedup
config.py         Settings and model registry
test_workspace/   Sample project with intentional bugs
tests/            Unit tests (+ a few integration tests that need Ollama)
docs/             Report, presentation outline, implementation notes
```

## Tests

```bash
python -m pytest tests -m "not integration"   # 104 unit tests, no Ollama needed
python -m pytest tests -m integration         # needs Ollama running
```

## Safety

- Paths are confined to the workspace by real path components (not string prefixes), so `..` escapes and look-alike sibling folders are both rejected.
- No workspace write happens without an explicit Approve click; each write is preceded by a timestamped backup in `.codepilot_backups/`.
- Test and run execution has timeouts and uses Docker when available. Without Docker it falls back to a temp-directory subprocess, which is **not** a hard security sandbox.
- Every tool call is appended to `audit_log.jsonl`.


## Pipeline mode (barebones)

`streamlit run app_pipeline.py` - write one prompt and watch the whole process on one page:
**Step 1 Investigate** (live log, then the root-cause analysis) -> **Step 2 Fix** (starts from step 1's
findings, live log, then a compile-checked patch) -> **Step 3 Review** (approve or reject the patch).
If the investigation fails, the fix step is skipped. `pipeline_runner.py` holds the logic and
`tests/test_pipeline.py` tests it with a fake agent. The original tabbed app is still `app.py`.

## Retro theme (Windows XP style)

`app_pipeline.py` wears a Windows-XP look: Bliss-style desktop, blue title-bar windows, green START button,
progress bar while an agent works, and a status taskbar. It is **CSS and static HTML only: no JavaScript**.
- Turn it off completely: `CODEPILOT_NO_THEME=1 streamlit run app_pipeline.py` (plain Streamlit layout, same functionality).
- Your own graphics: put `wallpaper.png` (or .jpg/.webp, up to ~2 MB) in `assets/`; it replaces the built-in hill.
- Files: `xp_theme.py` (CSS, icons drawn in code), `.streamlit/config.toml` (light base theme), `tests/test_xp_theme.py`.
