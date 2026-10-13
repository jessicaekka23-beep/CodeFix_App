"""Static HTML/SVG builders for the demo views (index, retrieval, agent trace, flow strip).

Everything user- or model-supplied is escaped, there is no JavaScript, and the markup has no blank lines or leading
indentation (Streamlit's markdown parser would otherwise cut the HTML block short). Styling lives in xp_theme.CSS.
"""
import html
import json

CATEGORY_COLORS = ["#ff8fa3", "#6fb3ff", "#7fdc9a", "#ffd36b", "#b79cff", "#ff9d6b", "#5fd3d0", "#e58fd8"]


def _esc(text) -> str:
    return html.escape(str(text if text is not None else "")).replace("\n", "&#10;")


def closeness(distance) -> float:
    """Turn a vector distance (0 = identical) into a 0..1 'how close' score for display."""
    try:
        return 1.0 / (1.0 + max(float(distance), 0.0))
    except (TypeError, ValueError):
        return 0.0


def _blocks(frac: float, total: int = 10) -> str:
    on = max(0, min(total, round(frac * total)))
    return '<span class="xp-meter">' + "<i class='on'></i>" * on + "<i></i>" * (total - on) + "</span>"


# ── pipeline flow strip ──────────────────────────────────────
def flow_html(nodes) -> str:
    """nodes: iterable of (label, sub, state) with state in {"idle", "active", "done", "bad"}."""
    cells = []
    for label, sub, state in nodes:
        cells.append(f'<div class="xp-node {_esc(state)}"><b>{_esc(label)}</b><span>{_esc(sub)}</span></div>')
    return '<div class="xp-flow">' + '<span class="xp-arrow">&#9654;</span>'.join(cells) + "</div>"


# ── tables ───────────────────────────────────────────────────
def table_html(headers, rows) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="xp-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ── chunking view: one bar per chunk, positioned over the file's line range ──
def chunk_bars_html(chunks, total_lines: int) -> str:
    total = max(int(total_lines), 1)
    rows = []
    for i, c in enumerate(chunks):
        start, end = max(int(c["start"]), 1), max(int(c["end"]), 1)
        left = (start - 1) / total * 100
        width = max((end - start + 1) / total * 100, 1.5)
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        rows.append(f'<div class="xp-chunkrow"><span class="xp-chunklabel">#{i + 1}</span>'
                    f'<div class="xp-chunktrack"><div class="xp-chunkbar" style="left:{left:.2f}%;width:{width:.2f}%;background:{color}"></div></div>'
                    f'<span class="xp-chunklines">lines {start}-{end}</span></div>')
    return ('<div class="xp-chunks">' + "".join(rows) +
            f'<div class="xp-chunknote">file = {total} lines. Where two bars overlap, those lines appear in both chunks.</div></div>')


# ── an embedding vector drawn as a strip of coloured cells ───
def vector_strip_html(values) -> str:
    vals = [float(v) for v in values][:96]
    if not vals:
        return ""
    peak = max(abs(v) for v in vals) or 1.0
    cells = []
    for v in vals:
        a = min(abs(v) / peak, 1.0)
        # positive = warm pink, negative = cool blue; stronger value = stronger colour
        rgb = (255, int(255 - 110 * a), int(255 - 90 * a)) if v >= 0 else (int(255 - 130 * a), int(255 - 70 * a), 255)
        cells.append(f'<i style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})" title="{v:.3f}"></i>')
    return '<div class="xp-vec">' + "".join(cells) + "</div>"


# ── embedding map (2-D projection) ───────────────────────────
def map_svg(points, meta, query_xy=None, hit_ids=(), width=480, height=300) -> str:
    """points: [(x, y)] in 0..1, meta: [{"file","chunk_id"}], query_xy: (x, y) or None, hit_ids: chunk_ids in rank order."""
    pad = 16
    files = []
    for m in meta:
        if m.get("file") not in files:
            files.append(m.get("file"))

    def px(p):
        return pad + p[0] * (width - 2 * pad), pad + (1 - p[1]) * (height - 2 * pad)

    out = [f'<svg class="xp-mapsvg" viewBox="0 0 {width} {height}" shape-rendering="crispEdges" role="img" aria-label="Map of code chunks by meaning">']
    for g in range(1, 8):
        out.append(f'<line class="grid" x1="{g * width / 8:.0f}" y1="0" x2="{g * width / 8:.0f}" y2="{height}"/>')
    for g in range(1, 5):
        out.append(f'<line class="grid" x1="0" y1="{g * height / 5:.0f}" x2="{width}" y2="{g * height / 5:.0f}"/>')
    by_id = {}
    for i, (p, m) in enumerate(zip(points, meta)):
        x, y = px(p)
        by_id[m.get("chunk_id")] = (x, y)
        color = CATEGORY_COLORS[files.index(m.get("file")) % len(CATEGORY_COLORS)]
        out.append(f'<rect x="{x - 3:.0f}" y="{y - 3:.0f}" width="7" height="7" fill="{color}" stroke="#2b2b45" stroke-width="1"><title>{_esc(m.get("file"))} lines {_esc(m.get("start"))}-{_esc(m.get("end"))}</title></rect>')
    qx = qy = None
    if query_xy:
        qx, qy = px(query_xy)
    for rank, cid in enumerate(hit_ids, 1):
        if cid in by_id and qx is not None:
            hx, hy = by_id[cid]
            out.append(f'<line class="link" x1="{qx:.0f}" y1="{qy:.0f}" x2="{hx:.0f}" y2="{hy:.0f}"/>')
    for rank, cid in enumerate(hit_ids, 1):
        if cid in by_id:
            hx, hy = by_id[cid]
            out.append(f'<rect class="ring" x="{hx - 8:.0f}" y="{hy - 8:.0f}" width="16" height="16" fill="none"/>'
                       f'<text class="rank" x="{hx + 11:.0f}" y="{hy - 9:.0f}">{rank}</text>')
    if qx is not None:
        out.append(f'<g class="query"><rect x="{qx - 3:.0f}" y="{qy - 9:.0f}" width="7" height="19"/><rect x="{qx - 9:.0f}" y="{qy - 3:.0f}" width="19" height="7"/></g>'
                   f'<text class="qlabel" x="{min(qx + 12, width - 60):.0f}" y="{max(qy - 10, 12):.0f}">your prompt</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="xp-leg"><i style="background:{CATEGORY_COLORS[i % len(CATEGORY_COLORS)]}"></i>{_esc(f)}</span>' for i, f in enumerate(files))
    return '<div class="xp-map">' + "".join(out) + f'<div class="xp-legend">{legend}</div></div>'


# ── retrieval evidence (what the agent was handed before its first LLM call) ──
def seed_html(event) -> str:
    hits = event.get("hits") or []
    if not hits:
        return ('<div class="xp-note">The index returned nothing for this prompt, so the agent starts without retrieved context. '
                'Index the workspace in the window above to see RAG ground the answer.</div>')
    items = []
    for rank, h in enumerate(hits, 1):
        c = closeness(h.get("distance"))
        d = h.get("distance")
        dist = f"{float(d):.3f}" if isinstance(d, (int, float)) else "?"
        items.append(
            f'<div class="xp-hit"><div class="xp-hithead"><span class="xp-rank">{rank}</span>'
            f'<b>{_esc(_short(h.get("source")))}</b><span class="xp-lines">lines {_esc(h.get("start"))}-{_esc(h.get("end"))}</span>'
            f'<span class="xp-from">{_esc(h.get("label"))}</span>{_blocks(c)}<span class="xp-dist">distance {dist}</span></div>'
            f'<details><summary>show the retrieved text</summary><pre>{_esc(h.get("text"))}</pre></details></div>')
    q = event.get("query", "")
    return (f'<div class="xp-seed"><div class="xp-q">query sent to the index: <code>{_esc(q[:160])}</code></div>' + "".join(items) +
            '<div class="xp-note">Smaller distance = closer in meaning. These chunks were pasted into the agent\'s prompt before it made its first decision.</div></div>')


def _short(path) -> str:
    text = str(path or "?").replace("\\", "/")
    return "/".join(text.split("/")[-2:])


# ── agent trace ──────────────────────────────────────────────
def _args(args) -> str:
    try:
        text = json.dumps(args, separators=(", ", ": "))
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= 150 else text[:147] + "..."


def _row(t, tag, kind, body, extra="") -> str:
    return (f'<div class="xp-ev"><span class="xp-t">{_esc(f"{t:.0f}s" if isinstance(t, (int, float)) else "")}</span>'
            f'<span class="xp-tag {kind}">{_esc(tag)}</span><div class="xp-evbody">{body}{extra}</div></div>')


def trace_html(events, running: bool = False) -> str:
    rows = []
    for ev in events:
        kind, t = ev.get("type"), ev.get("t")
        if kind == "start":
            rows.append(_row(t, "START", "start", f'{_esc(ev.get("task"))} agent on <b>{_esc(ev.get("model"))}</b> '
                             f'<span class="xp-dim">limits: {_esc(ev.get("max_iterations"))} iterations, {_esc(ev.get("max_tool_calls"))} tool calls</span>'))
        elif kind == "seed":
            n = len(ev.get("hits") or [])
            body = (f'Searched the index <b>before</b> the first LLM call: <b>{n}</b> chunk{"s" if n != 1 else ""} retrieved, '
                    f'{_esc(ev.get("used", 0))} put in the prompt')
            rows.append(_row(t, "RAG", "rag", body, f'<details><summary>see them</summary>{seed_html(ev)}</details>'))
        elif kind == "iteration":
            rows.append(f'<div class="xp-iter">iteration {_esc(ev.get("n"))} of {_esc(ev.get("max"))}</div>')
        elif kind == "llm":
            what = {"tool_call": "chose to call a tool", "final_answer": "wrote its final answer", "error": "replied in a format I could not parse"}.get(ev.get("action"), "replied")
            body = f'Model {what} <span class="xp-dim">({_esc(ev.get("seconds"))}s, prompt {_esc(ev.get("prompt_chars"))} chars)</span>'
            if ev.get("thought"):
                body += f'<div class="xp-thought">{_esc(ev["thought"])}</div>'
            rows.append(_row(t, "THINK", "llm", body))
        elif kind == "tool":
            ok = ev.get("success", True)
            body = (f'<code>{_esc(ev.get("name"))}({_esc(_args(ev.get("args")))})</code><div>&#8594; {_esc(ev.get("summary"))} '
                    f'<span class="xp-dim">{_esc(ev.get("seconds"))}s</span></div>')
            rows.append(_row(t, "TOOL" if ok else "FAIL", "tool" if ok else "warn", body))
        elif kind == "patch":
            st = ev.get("stats") or {}
            body = (f'Proposed a patch to <b>{_esc(_short(ev.get("filepath")))}</b> (+{_esc(st.get("additions", 0))} / -{_esc(st.get("deletions", 0))}), '
                    + ("compile check passed" if ev.get("syntax_ok", True) else "<b>does not compile</b>"))
            rows.append(_row(t, "PATCH", "patch" if ev.get("syntax_ok", True) else "warn", body))
        elif kind == "note":
            rows.append(_row(t, "NOTE", "warn", _esc(ev.get("text"))))
        elif kind == "final":
            rows.append(_row(t, "DONE", "ok", f'Final answer written ({_esc(ev.get("chars"))} chars)'))
    if running:
        rows.append('<div class="xp-ev xp-wait"><span class="xp-t"></span><span class="xp-tag idle">...</span><div class="xp-evbody xp-dim">working</div></div>')
    if not rows:
        return '<div class="xp-note">Waiting for the first event...</div>'
    return '<div class="xp-trace">' + "".join(rows) + "</div>"
