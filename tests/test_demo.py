"""Demo views: instrumented indexing, embedding map, agent events, HTML builders. No Ollama, ChromaDB or Streamlit."""
import time

import demo_data as dd
import xp_widgets as w
from pipeline_runner import PipelineRun


# ── fakes ────────────────────────────────────────────────
def _docs(_ws):
    return [{"path": "/ws/a.py", "content": "def f():\n    return 1\n" * 30, "extension": ".py"},
            {"path": "/ws/pkg/b.py", "content": "def g():\n    return 2\n" * 10, "extension": ".py"}]


def _chunks(docs):
    out = []
    for d in docs:
        for i in range(3):
            out.append({"text": f"chunk {i} of {d['path']}", "metadata": {"source": d["path"], "chunk_id": f"{d['path']}::chunk_{i}",
                                                                            "start_line": i * 10 + 1, "end_line": i * 10 + 10}})
    return out


class FakeStore:
    def __init__(self):
        self.cleared = []

    def clear_collection(self, name):
        self.cleared.append(name)

    def index_workspace(self, ws):
        return {"total_documents": 2, "total_chunks": 6, "indexed": 6, "skipped_duplicates": 0, "errors": 0}

    def get_collection_stats(self, name):
        return {"name": name, "count": 6}


def _vectors(_store):
    import random
    rnd = random.Random(1)
    metas = [m["metadata"] for m in _chunks(_docs(""))]
    return [[rnd.random() for _ in range(8)] for _ in metas], metas


def _wait(run, timeout=5):
    t0 = time.time()
    while run.alive() and time.time() - t0 < timeout:
        time.sleep(0.01)


def _run(**over):
    kw = dict(loader=_docs, chunker=_chunks, embedder=lambda t: [0.1] * 16, store_factory=FakeStore, fetch_vectors=_vectors)
    kw.update(over)
    run = dd.IndexRun("/ws", **kw).start()
    _wait(run)
    return run


# ── indexing ─────────────────────────────────────────────
def test_index_run_records_every_stage():
    run = _run()
    assert run.ok() and run.done_stages == list(dd.STAGES)
    assert [d["file"] for d in run.docs] == ["a.py", "pkg/b.py"]            # paths are relative to the workspace
    assert len(run.chunks) == 6 and run.chunks[0]["start"] == 1
    assert run.embedding["dim"] == 16 and len(run.embedding["values"]) == 16
    assert run.stats["indexed"] == 6 and run.count == 6
    assert run.map and len(run.map["points"]) == 6 == len(run.map_meta)


def test_index_clears_the_old_project_first():
    store = FakeStore()
    _run(store_factory=lambda: store)
    assert store.cleared == [dd.COLLECTION]


def test_embedding_failure_stops_early_with_a_helpful_error():
    run = _run(embedder=lambda t: None)
    assert not run.ok() and "Ollama" in run.error and run.done_stages == ["load", "chunk"] and not run.alive()


def test_empty_workspace_is_reported():
    run = _run(loader=lambda ws: [])
    assert "No supported source files" in run.error and not run.alive()


def test_map_failure_does_not_fail_the_index():
    def boom(_store):
        raise RuntimeError("no vectors")
    run = _run(fetch_vectors=boom)
    assert run.ok() and run.map is None


# ── projection ───────────────────────────────────────────
def test_projection_is_normalised_and_places_new_points():
    vecs = [[float(i), float(i % 3), 1.0, 0.0] for i in range(10)]
    model = dd.fit_projection(vecs)
    assert len(model["points"]) == 10
    assert all(-1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9 for x, y in model["points"])
    x, y = dd.project_point(model, vecs[3])
    assert abs(x - model["points"][3][0]) < 1e-6 and abs(y - model["points"][3][1]) < 1e-6
    assert dd.project_point(model, [1e6, -1e6, 0, 0]) is not None                  # far-away query is clamped, not an error


def test_projection_needs_two_vectors():
    assert dd.fit_projection([[1.0, 2.0]]) is None and dd.project_point(None, [1]) is None


# ── HTML builders ────────────────────────────────────────
def test_builders_escape_model_and_user_text():
    evil = "<script>alert(1)</script>"
    out = "".join([
        w.trace_html([{"type": "tool", "name": evil, "args": {"a": evil}, "summary": evil, "success": True, "seconds": 0.1, "t": 1},
                      {"type": "llm", "action": "tool_call", "thought": evil, "seconds": 1, "prompt_chars": 5, "t": 2},
                      {"type": "note", "text": evil}, {"type": "patch", "filepath": evil, "stats": {}, "syntax_ok": True}]),
        w.seed_html({"query": evil, "hits": [{"source": evil, "text": evil, "distance": 0.4, "label": evil, "start": 1, "end": 2}]}),
        w.table_html([evil], [[evil]]), w.flow_html([(evil, evil, "done")]),
        w.map_svg([(0.5, 0.5)], [{"file": evil, "chunk_id": "c", "start": 1, "end": 2}], (0.4, 0.4), ["c"]),
    ])
    assert "<script" not in out.lower()


def test_html_has_no_blank_lines_that_would_cut_a_markdown_block():
    for out in (w.trace_html([{"type": "llm", "action": "tool_call", "thought": "a\n\nb", "t": 0}]),
                w.seed_html({"query": "q", "hits": [{"source": "s", "text": "x\n\ny", "distance": 1, "label": "l", "start": 1, "end": 2}]})):
        assert "\n" not in out


def test_trace_describes_each_event_type_and_a_running_marker():
    events = [{"type": "start", "task": "investigate", "model": "m", "t": 0}, {"type": "seed", "hits": [], "used": 0, "t": 0},
              {"type": "iteration", "n": 1, "max": 10}, {"type": "tool", "name": "read_file", "args": {}, "summary": "ok", "success": False, "t": 1},
              {"type": "final", "chars": 10, "t": 2}]
    out = w.trace_html(events, running=True)
    for word in ("START", "RAG", "iteration 1 of 10", "FAIL", "DONE", "working"):
        assert word in out


def test_empty_seed_explains_how_to_get_rag():
    assert "Index the workspace" in w.seed_html({"hits": []})


def test_closeness_is_monotonic():
    assert w.closeness(0) == 1.0 and w.closeness(0.2) > w.closeness(1.5) > w.closeness(None) - 1e-9


# ── pipeline forwards structured events ──────────────────
def test_pipeline_collects_events_when_the_agent_supports_them():
    def agent(user_query, workspace_root, model, task_type, on_status, on_event=None, **kw):
        on_event({"type": "start", "task": task_type})
        return {"success": True, "answer": "x", "proposed_patches": []}
    run = PipelineRun("q", "/ws", "m", agent_fn=agent).start()
    _wait(run)
    assert [e["type"] for e in run.events["investigate"]] == ["start"] and "t" in run.events["investigate"][0]
    assert run.events["fix"][0]["task"] == "fix"


def test_pipeline_still_works_with_an_agent_that_has_no_on_event():
    def agent(user_query, workspace_root, model, task_type, on_status, provider, api_key, base_url):
        return {"success": True, "answer": "x"}
    run = PipelineRun("q", "/ws", "m", agent_fn=agent).start()
    _wait(run)
    assert run.done("fix") and run.events == {"investigate": [], "fix": []}


# ── the real agent loop emits events (LLM and tools faked) ──
def test_agent_loop_emits_events(monkeypatch):
    import agent
    replies = iter(['I should read the file first.\nTOOL_CALL: {"name": "read_file", "arguments": {"filepath": "a.py"}}',
                    "FINAL_ANSWER: the bug is on line 3"])
    monkeypatch.setattr(agent, "call_llm", lambda *a, **k: {"success": True, "response": next(replies), "error": None, "duration": 0.1})
    monkeypatch.setattr(agent, "execute_tool", lambda name, args, ws: {"success": True, "content": "x", "total_lines": 3}
                        if name == "read_file" else {"success": True, "results": [{"source": "a.py", "chunk_id": "a::0", "start_line": 1,
                                                                                  "end_line": 5, "distance": 0.3, "text": "def f(): pass"}]})
    events = []
    out = agent.run_agent_loop("why?", "/ws", model="m", on_event=events.append)
    kinds = [e["type"] for e in events]
    assert out["success"] and kinds[0] == "start" and "seed" in kinds and kinds.count("llm") == 2 and kinds[-1] == "final"
    seed = next(e for e in events if e["type"] == "seed")
    assert seed["hits"][0]["distance"] == 0.3 and seed["hits"][0]["chunk_id"] == "a::0"
    llm = next(e for e in events if e["type"] == "llm")
    assert llm["action"] == "tool_call" and "read the file" in llm["thought"]
    tool = next(e for e in events if e["type"] == "tool")
    assert tool["name"] == "read_file" and tool["success"]


def test_a_broken_event_hook_never_breaks_the_agent(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "call_llm", lambda *a, **k: {"success": True, "response": "FINAL_ANSWER: done", "error": None, "duration": 0})
    monkeypatch.setattr(agent, "execute_tool", lambda *a, **k: {"success": True, "results": []})

    def bad(_event):
        raise ValueError("ui bug")
    assert agent.run_agent_loop("q", "/ws", model="m", on_event=bad)["success"]
