"""One prompt -> Investigate -> Fix, as a single visible pipeline.

Pure Python, no Streamlit. The pipeline runs in one background thread and only appends to plain
lists / dicts, so the UI can poll it safely (and re-attach to it after a Streamlit rerun).
"""
import inspect
import threading
import time

STEPS = ("investigate", "fix")
MAX_HANDOFF_CHARS = 3500


def _default_agent():
    from agent import run_agent_loop          # imported lazily so tests can inject a fake
    return run_agent_loop


def _accepts(fn, name):
    """True if fn can take keyword `name` (so older / fake agents without on_event keep working)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def build_fix_query(query: str, investigation_answer: str) -> str:
    """The Fix agent gets the original request plus what the Investigate agent found."""
    findings = (investigation_answer or "").strip()[:MAX_HANDOFF_CHARS]
    if not findings:
        return query
    return (f"{query}\n\n"
            "An investigation of this bug has already been done. Use it as your starting point, "
            "but verify against the real files before proposing a patch.\n"
            f"--- Investigation findings ---\n{findings}\n--- end of findings ---")


class PipelineRun:
    def __init__(self, query, workspace_root, model, provider="ollama", api_key="", base_url="", agent_fn=None):
        self.query, self.workspace_root, self.model = query, workspace_root, model
        self.provider, self.api_key, self.base_url = provider, api_key, base_url
        self.logs = {s: [] for s in STEPS}      # live status messages per step
        self.events = {s: [] for s in STEPS}    # structured trace events per step (seed, llm, tool, ...), for the demo view
        self.queries = {}                       # what each agent was actually asked
        self.results = {}                       # step -> result dict (set when the step ends)
        self.started, self.finished = {}, {}    # step -> timestamp
        self.stopped = None                     # reason the pipeline stopped before the Fix step
        self.ended = False
        self.patches_taken = False              # UI flag
        self._agent_fn = agent_fn

    def start(self):
        self._agent_fn = self._agent_fn or _default_agent()      # import once, on the caller's thread
        threading.Thread(target=self._work, daemon=True, name="codepilot-pipeline").start()
        return self

    def _step(self, step, query):
        self.queries[step] = query
        self.started[step] = time.time()
        t0 = self.started[step]
        extra = {}
        if _accepts(self._agent_fn, "on_event"):
            extra["on_event"] = lambda ev: self.events[step].append({**ev, "t": round(time.time() - t0, 1)})
        try:
            result = self._agent_fn(
                user_query=query, workspace_root=self.workspace_root, model=self.model,
                task_type=step, on_status=self.logs[step].append, provider=self.provider,
                api_key=self.api_key, base_url=self.base_url, **extra)
        except Exception as exc:
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        self.results[step] = result
        self.finished[step] = time.time()       # set last: "finished" implies "result is ready"
        return result

    def _work(self):
        try:
            found = self._step("investigate", self.query)
            if not found.get("success"):
                self.stopped = "The investigation failed, so the fix step was skipped."
                return
            self._step("fix", build_fix_query(self.query, found.get("answer", "")))
        finally:
            self.ended = True

    def done(self, step):
        return step in self.finished

    def running(self, step):
        return step in self.started and step not in self.finished

    def alive(self):
        return not self.ended

    def elapsed(self, step):
        end = self.finished.get(step) or time.time()
        return end - self.started.get(step, end)
