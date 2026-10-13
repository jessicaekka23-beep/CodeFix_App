"""Tests for the Investigate -> Fix pipeline (no Ollama, no Streamlit: the agent is a fake)."""
import time

from pipeline_runner import PipelineRun, build_fix_query


def _fake(calls, fail_step=None, ok=True):
    def agent(user_query, workspace_root, model, task_type, on_status, **kw):
        calls.append((task_type, user_query, time.time()))
        on_status(f"working on {task_type}")
        time.sleep(0.1)
        if task_type == fail_step:
            raise RuntimeError("boom")
        if task_type == "investigate":
            return {"success": ok, "answer": "ROOT CAUSE: divides by 2", "error": None if ok else "no luck"}
        return {"success": True, "answer": "patched", "proposed_patches": [{"filepath": "a.py"}]}
    return agent


def _wait(run, timeout=5):
    t0 = time.time()
    while run.alive() and time.time() - t0 < timeout:
        time.sleep(0.02)


def test_fix_runs_after_investigation_and_receives_its_findings():
    calls = []
    run = PipelineRun("why wrong?", "/ws", "m", agent_fn=_fake(calls)).start()
    _wait(run)
    assert [c[0] for c in calls] == ["investigate", "fix"]
    assert calls[0][1] == "why wrong?"
    assert "ROOT CAUSE: divides by 2" in calls[1][1] and "why wrong?" in calls[1][1]
    assert run.finished["investigate"] <= run.started["fix"]          # strictly sequential
    assert run.results["fix"]["proposed_patches"] and run.stopped is None


def test_failed_investigation_stops_the_pipeline():
    calls = []
    run = PipelineRun("q", "/ws", "m", agent_fn=_fake(calls, ok=False)).start()
    _wait(run)
    assert [c[0] for c in calls] == ["investigate"] and "fix" not in run.results
    assert run.stopped and not run.alive()


def test_crash_is_reported_not_raised():
    run = PipelineRun("q", "/ws", "m", agent_fn=_fake([], fail_step="investigate")).start()
    _wait(run)
    assert "boom" in run.results["investigate"]["error"] and not run.alive()


def test_handoff_without_findings_is_just_the_query():
    assert build_fix_query("q", "  ") == "q"
