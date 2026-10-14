"""Pixel-art Windows-XP-style retro theme for CodePilot Local (pastel light + dark variants).

Pure CSS + static HTML: no JavaScript, nothing that touches Streamlit's own DOM, nothing clickable
(decorations use pointer-events:none). Set CODEPILOT_NO_THEME=1 to switch the whole theme off.

Fonts: Pixelify Sans (text), Silkscreen (titles) and VT323 (code) are loaded from Google Fonts. Offline, the
browser falls back to a generic monospace, so the layout still works.
"""
import base64
import html
import math
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

THEME_ON = os.environ.get("CODEPILOT_NO_THEME") != "1"
ASSETS = Path(__file__).resolve().parent / "assets"
MAX_WALLPAPER_BYTES = 2_000_000


def _uri(svg: str) -> str:
    return "url(\"data:image/svg+xml," + quote(svg, safe="/:=' ,.-()") + "\")"


# ── Pixel icons (16x16 grids drawn in code, rendered with crisp edges) ─────────
def _svg_from_grid(grid, pal):
    rects = []
    for y, row in enumerate(grid):
        x = 0
        while x < len(row):
            ch = row[x]
            if ch == "." or ch not in pal:
                x += 1
                continue
            x2 = x
            while x2 + 1 < len(row) and row[x2 + 1] == ch:
                x2 += 1
            rects.append(f'<rect x="{x}" y="{y}" width="{x2 - x + 1}" height="1" fill="{pal[ch]}"/>')
            x = x2 + 1
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" '
            'shape-rendering="crispEdges">' + "".join(rects) + "</svg>")


def _blank(n=16):
    return [["."] * n for _ in range(n)]


def _outline(g, ch="o"):
    """Turn every filled pixel that touches empty space into an outline pixel."""
    n = len(g)
    out = [row[:] for row in g]
    for y in range(n):
        for x in range(n):
            if g[y][x] == ".":
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < n and 0 <= xx < n) or g[yy][xx] == ".":
                    out[y][x] = ch
                    break
    return out


def _icon_app():
    g = _blank()
    spans = [(2, 13)] * 7 + [(3, 12), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]
    for i, (a, b) in enumerate(spans):
        for x in range(a, b + 1):
            g[1 + i][x] = "b"
    g = _outline(g)
    for y, x in ((3, 4), (4, 4), (3, 5)):
        g[y][x] = "l"
    for y, x in ((8, 5), (9, 6), (10, 7), (9, 8), (8, 9), (7, 10), (6, 11)):
        g[y][x] = "y"
        g[y - 1][x] = "y"
    return _svg_from_grid(["".join(r) for r in g], {"o": "#3b4a8f", "b": "#8ab4f8", "l": "#e3efff", "y": "#ffe27a"})


def _icon_search():
    rows = ["................", "....oooooo......", "...owwgggggo....", "..owgggggggo....", "..oggggggggo....",
            "..oggggggggo....", "..oggggggggo....", "..oggggggggo....", "...oggggggo.....", "....oooooo.h....",
            "...........hh...", "............hh..", ".............hh.", "..............hh", "................",
            "................"]
    return _svg_from_grid(rows, {"o": "#4a5a9a", "g": "#dff0ff", "w": "#ffffff", "h": "#c49a6c"})


def _icon_wrench():
    g = _blank()
    for y in range(16):
        for x in range(16):
            if abs((y + x) - 15) <= 1 and 5 <= y <= 13 and 2 <= x <= 10:
                g[y][x] = "m"
            if math.hypot(y - 5, x - 10) <= 3.7:
                g[y][x] = "m"
    for y in range(16):
        for x in range(16):
            if math.hypot(y - 3, x - 12) <= 2.3:
                g[y][x] = "."
    g = _outline(g)
    g[6][9] = "l"
    g[5][8] = "l"
    return _svg_from_grid(["".join(r) for r in g], {"o": "#59608a", "m": "#c5cbe3", "l": "#ffffff"})


def _icon_check():
    g = _blank()
    for y in range(1, 15):
        for x in range(1, 15):
            if (y in (1, 14) and x in (1, 14)):
                continue
            g[y][x] = "g"
    g = _outline(g)
    for y, x in ((8, 4), (9, 5), (10, 6), (9, 7), (8, 8), (7, 9), (6, 10), (5, 11),
                 (7, 4), (8, 5), (9, 6), (8, 7), (7, 8), (6, 9), (5, 10), (4, 11)):
        g[y][x] = "w"
    return _svg_from_grid(["".join(r) for r in g], {"o": "#3f8f5a", "g": "#8fe0a6", "w": "#ffffff"})


def _icon_flag():
    g = _blank()
    colors = {(0, 0): "r", (0, 1): "g", (1, 0): "b", (1, 1): "y"}
    for (qy, qx), ch in colors.items():
        for dy in range(6):
            for dx in range(6):
                wave = 1 if (dx + qx * 6) in (3, 4, 5, 6, 7) else 0
                y, x = 1 + qy * 7 + dy + wave, 1 + qx * 7 + dx
                if y < 16 and x < 16:
                    g[y][x] = ch
    return _svg_from_grid(["".join(r) for r in g], {"r": "#ff8fa3", "g": "#8fe3a0", "b": "#8fc5ff", "y": "#ffe08a"})


ICONS = {"app": _icon_app(), "search": _icon_search(), "wrench": _icon_wrench(), "check": _icon_check(), "flag": _icon_flag()}


# ── Pixel desktop scene (192x108 grid, scaled up with crisp edges) ─────────────
W, H = 192, 108

SCENES = {
    "light": dict(
        sky=["#a7d3ff", "#b4daff", "#c3dfff", "#d3e3ff", "#e2e4fb", "#efe3f6", "#f9e4f0", "#fde7ea", "#fff0e0"],
        hills=[("#cbc5f2", "#ded9fa", 66, 7, 0.045, 0.3), ("#b9ecd2", "#d2f6e2", 78, 6, 0.06, 2.0),
               ("#9fdc9c", "#c1efb4", 90, 5, 0.08, 4.1)],
        cloud="#ffffff", cloud_shade="#dfe6fb", sun="#fff0a6", sun_edge="#ffd98a", stars=False,
        flowers=["#ff9fb8", "#fff2a0", "#ffffff", "#c8a8f0"]),
    "dark": dict(
        sky=["#14153a", "#1a1b46", "#221f52", "#2c2560", "#3a2b6b", "#4b3375", "#5f3b80", "#78448a", "#8f5191"],
        hills=[("#3b3b86", "#4c4c9c", 66, 7, 0.045, 0.3), ("#2f5a86", "#3f709c", 78, 6, 0.06, 2.0),
               ("#2f6e68", "#42897f", 90, 5, 0.08, 4.1)],
        cloud="#5b5a9e", cloud_shade="#47467f", sun="#fff3c4", sun_edge="#ffe08a", stars=True,
        flowers=["#ffb3c8", "#fff2a0", "#cfd6ff", "#b89cf0"]),
}

CLOUD_SHAPES = {  # rows of (x offset, width)
    "big": [(5, 8), (2, 14), (0, 18), (0, 20), (1, 18)],
    "small": [(3, 6), (1, 10), (0, 12), (1, 10)],
}


def _rect(x, y, w, h, fill):
    return fill, f"M{x} {y}h{w}v{h}h-{w}z"


def _merge(items):
    """Join consecutive same-colour rects into one <path>; raw strings pass through."""
    out, cur, d = [], None, []
    for it in items:
        if isinstance(it, tuple) and it[0] == cur:
            d.append(it[1])
            continue
        if d:
            out.append(f"<path fill='{cur}' d='{''.join(d)}'/>")
            cur, d = None, []
        if isinstance(it, tuple):
            cur, d = it[0], [it[1]]
        else:
            out.append(it)
    if d:
        out.append(f"<path fill='{cur}' d='{''.join(d)}'/>")
    return "".join(out)


def _scene_svg(mode: str) -> str:
    s = SCENES[mode]
    body = []
    n = len(s["sky"])
    band_h = 80 // n + 1
    for i, c in enumerate(s["sky"]):
        y = i * band_h
        body.append(_rect(0, y, W, (H - y) if i == n - 1 else band_h, c))
        if i + 1 < n:   # dithered seam: every other pixel of the next colour
            body.append(f"<rect y='{y + band_h - 1}' width='{W}' height='1' fill='{s['sky'][i + 1]}' mask='url(#m)'/>")
    defs = ("<defs><pattern id='p' width='2' height='1' patternUnits='userSpaceOnUse'><rect width='1' height='1' fill='#fff'/></pattern>"
            f"<mask id='m'><rect width='{W}' height='{H}' fill='url(#p)'/></mask></defs>")
    if s["stars"]:
        for i in range(70):
            x = (i * 53 + 17) % W
            y = (i * 29 + 7) % 56
            c = "#ffffff" if i % 3 else "#ffe9a8"
            body.append(_rect(x, y, 1, 1, c))
            if i % 9 == 0:
                body += [_rect(x - 1, y, 3, 1, c), _rect(x, y - 1, 1, 3, c)]
        for dy, (a, w) in enumerate([(2, 3), (1, 3), (0, 3), (0, 3), (0, 3), (1, 3), (2, 3)]):   # crescent moon
            body.append(_rect(150 + a, 12 + dy, w, 1, s["sun"]))
        body.append(_rect(150, 14, 2, 3, s["sun"]))
    else:
        r = 7
        for dy in range(-r, r + 1):                                                                # pixel sun
            half = int(math.sqrt(max(r * r - dy * dy, 0)) + 0.5)
            body.append(_rect(152 - half, 16 + dy, 2 * half + 1, 1, s["sun"] if abs(dy) < r - 1 else s["sun_edge"]))
    for (cx, cy, kind) in ((18, 14, "big"), (78, 26, "small"), (118, 10, "big"), (168, 40, "small"), (44, 44, "small")):
        for dy, (a, w) in enumerate(CLOUD_SHAPES[kind]):
            last = dy == len(CLOUD_SHAPES[kind]) - 1
            body.append(_rect(cx + a, cy + dy, w, 1, s["cloud_shade"] if last else s["cloud"]))
    for li, (base, top, y0, amp, freq, ph) in enumerate(s["hills"]):
        tops = [int(y0 + amp * math.sin(x * freq + ph) + amp * 0.5 * math.sin(x * freq * 2.7 + ph * 1.7)) for x in range(W)]
        runs, x = [], 0
        while x < W:
            x2 = x
            while x2 + 1 < W and tops[x2 + 1] == tops[x]:
                x2 += 1
            runs.append((x, x2 - x + 1, tops[x]))
            x = x2 + 1
        body += [_rect(x, t, w, H - t, base) for x, w, t in runs]
        body += [_rect(x, t, w, 1, top) for x, w, t in runs]
        if li == 2:                                                                                # pixel flowers
            for i in range(26):
                fx = (i * 37 + 11) % W
                fy = tops[fx] + 3 + (i * 7) % (H - tops[fx] - 5)
                body.append(_rect(fx, fy, 1, 1, s["flowers"][i % len(s["flowers"])]))
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMax slice' "
            f"shape-rendering='crispEdges'>{defs}{_merge(body)}</svg>")


@lru_cache(maxsize=1)
def _wallpaper_uri():
    """User graphics win: assets/wallpaper.(png|jpg|jpeg|webp) if present and not huge."""
    for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp")):
        f = ASSETS / f"wallpaper.{ext}"
        if f.is_file() and f.stat().st_size <= MAX_WALLPAPER_BYTES:
            return f"url(\"data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}\")"
    return None


PALETTES = {
    "light": dict(
        title_top="#d3e2ff", title_bot="#b6cbf5", title_ink="#2b3566", frame="#cbbfe0", face="#faf5ec", face2="#fffdf8",
        ink="#33303f", navy="#3f4d7a", line="#e4d8c4", sel="#b9cdf0", sel_ink="#ffffff", shadow="rgba(120,102,72,.16)",
        side_a="#fbf5ec", side_b="#f3ecf6", btn_top="#ffffff", btn_bot="#f0e9dc", btn_line="#cbbfe0",
        go_top="#dcefcb", go_bot="#b6dd9f", go_line="#86b06f", go_ink="#2f4a2b",
        chip="#fffdf8", muted="#6f6a5e", code_bg="#fffaf0", exp_a="#fffdf8", exp_b="#f0eaf4",
        bar_a="#d3e2ff", bar_b="#b6cbf5", bar_line="#cbbfe0", tray_a="#f0e9dc", tray_b="#e2d8ca", start_ink="#2f4a2b",
        task_a="#b6cbf5", task_b="#d3e2ff", task_ink="#2b3566", prog="#a9d79a", scheme="light",
        t_rag="#e9ddf7", t_llm="#dbe8ff", t_tool="#d9f0d5", t_patch="#fff0b8", t_ok="#c7ecd0", t_warn="#ffd9de", t_start="#f2e7d6",
        radius="16px", radius_sm="10px", card_line="#eadfcd", card_shadow="0 10px 26px rgba(120,102,72,.15)",
        side_grad="linear-gradient(180deg,#fbf5ec 0%,#f3ecf6 100%)"),
    "dark": dict(
        title_top="#8a79c9", title_bot="#6a5aa8", title_ink="#f7f3ff", frame="#5b5170", face="#2a2635", face2="#211e2a",
        ink="#efe9df", navy="#d8c7e8", line="#453e56", sel="#8a79c9", sel_ink="#ffffff", shadow="rgba(0,0,0,.5)",
        side_a="#262231", side_b="#2c2439", btn_top="#3a3448", btn_bot="#2b2637", btn_line="#6b6183",
        go_top="#b6dd9f", go_bot="#7cb063", go_line="#5c8a4a", go_ink="#16280f",
        chip="#211e2a", muted="#b0a89c", code_bg="#1a1722", exp_a="#332d42", exp_b="#43395c",
        bar_a="#6a5aa8", bar_b="#4a3f80", bar_line="#6b6183", tray_a="#3a3448", tray_b="#2b2637", start_ink="#16280f",
        task_a="#4a3f80", task_b="#6a5aa8", task_ink="#f7f3ff", prog="#a9d79a", scheme="dark",
        t_rag="#5b4a8f", t_llm="#3a4f8f", t_tool="#3a6b4a", t_patch="#7a6a2a", t_ok="#3f7a4f", t_warn="#8f4a58", t_start="#5a4a3a",
        radius="16px", radius_sm="10px", card_line="#403950", card_shadow="0 12px 28px rgba(0,0,0,.5)",
        side_grad="linear-gradient(180deg,#262231 0%,#2c2439 100%)"),
}


def theme_css(dark: bool = False) -> str:
    return _theme_css("dark" if dark else "light")


@lru_cache(maxsize=2)
def _theme_css(mode: str) -> str:
    wall = _wallpaper_uri()
    base = (f"{wall} center / cover no-repeat fixed" if wall
            else f"{_uri(_scene_svg(mode))} center bottom / cover no-repeat fixed")
    wash = ("linear-gradient(rgba(255,250,240,.34), rgba(250,244,236,.34))" if mode == "light"
            else "linear-gradient(rgba(28,24,34,.46), rgba(28,24,34,.46))")
    desktop = f"{wash} center / cover no-repeat fixed, {base}"
    icons = "".join(f".xp-ico.{k} {{background-image: {_uri(v)};}}\n" for k, v in ICONS.items())
    variables = "".join(f"--{k.replace('_', '-')}: {v}; " for k, v in PALETTES[mode].items())
    css = CSS.replace("/*VARS*/", variables).replace("/*DESKTOP*/", desktop).replace("/*ICONS*/", icons)
    return f"<style>{css}</style>"


CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;500;700&family=Silkscreen:wght@400;700&family=VT323&display=swap');
:root {/*VARS*/
       --px-body: 'Pixelify Sans', 'Silkscreen', ui-monospace, 'Courier New', monospace;
       --px-head: 'Silkscreen', 'Pixelify Sans', ui-monospace, 'Courier New', monospace;
       --px-code: 'VT323', ui-monospace, Consolas, 'Courier New', monospace;
       color-scheme: var(--scheme);}

/* desktop */
.stApp {background: /*DESKTOP*/ !important; image-rendering: pixelated; font-family: var(--px-body); color: var(--ink);}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {background: transparent !important;}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stHeader"], [data-testid="stHeader"] * {color: var(--ink);}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {display: none !important;}
.block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 80px;}
::selection {background: var(--sel); color: var(--sel-ink);}

/* pixel fonts everywhere except icons and code */
.stApp :where(p, li, h1, h2, h3, h4, h5, h6, label, summary, button, input, textarea, [data-baseweb="select"] *, [data-baseweb="menu"] *)
    :not([data-testid="stIconMaterial"]) {font-family: var(--px-body);}
.stApp :where(p, li, label, summary, button, input, textarea, h1, h2, h3, h4, h5, h6) {font-family: var(--px-body); color: var(--ink);}
.stApp :where(h1, h2, h3, h4, h5, h6) {font-family: var(--px-head); letter-spacing: .5px; color: var(--navy);}
.stApp [data-testid="stMarkdownContainer"] {color: var(--ink);}
.stApp [data-testid="stMarkdownContainer"] a {color: var(--navy);}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {color: var(--muted); font-family: var(--px-body);}
.stApp code, .stApp pre, .stApp [data-testid="stCode"] *, .stApp [data-testid="stCodeBlock"] * {font-family: var(--px-code) !important; font-size: 1.05rem;}
.stApp :not(pre) > code {background: var(--code-bg); color: var(--ink); border: 1px solid var(--line); border-radius: 0; padding: 0 4px;}
.stApp hr {border-color: var(--line);}

/* icons + title bars */
.xp-ico {display: inline-block; width: 16px; height: 16px; flex: none; background: center / contain no-repeat; image-rendering: pixelated;}
/*ICONS*/
.xp-title {display: flex; align-items: center; gap: 8px; height: 32px; padding: 0 5px 0 9px; margin: 0 -14px 6px;
           color: var(--title-ink) !important; font: 700 12px var(--px-head); letter-spacing: .6px;
           background: linear-gradient(180deg, var(--title-top) 0 50%, var(--title-bot) 50% 100%); border-bottom: 2px solid var(--frame);
           box-shadow: inset 0 2px 0 rgba(255,255,255,.55);}
.xp-title.flat {margin: 0 0 10px; border: 2px solid var(--frame);}
.xp-title * {color: var(--title-ink) !important;}
.xp-btns {margin-left: auto; display: flex; gap: 3px; pointer-events: none;}
.xp-btns i {display: grid; place-items: center; width: 20px; height: 20px; font: 700 12px/1 var(--px-head); font-style: normal; color: var(--title-ink);
            background: var(--face); border: 2px solid var(--title-ink); box-shadow: inset 2px 2px 0 #fff, inset -2px -2px 0 rgba(0,0,0,.18);}
.xp-btns i.x {background: #ffb0b8;}

/* windows: containers created with key="xpwin_*" */
[class*="st-key-xpwin"] {background: var(--face); border: 3px solid var(--frame); border-top-width: 0; overflow: hidden; border-radius: 0;
                         padding: 0 14px 14px; box-shadow: 6px 6px 0 var(--shadow); margin-bottom: 22px; image-rendering: auto;}
.xp-hero {background: var(--face); border: 3px solid var(--frame); border-top-width: 0; overflow: hidden; box-shadow: 6px 6px 0 var(--shadow); margin-bottom: 22px;}
.xp-hero .xp-title {margin: 0;}
.xp-hero-body {padding: 14px 18px 12px;}
.xp-brand {font: 700 2rem/1.15 var(--px-head); color: var(--navy) !important; letter-spacing: 1px; text-shadow: 3px 3px 0 var(--title-top);}
.xp-sub {margin-top: 6px; color: var(--muted) !important; font-family: var(--px-body);}
.xp-chips {margin-top: 12px;}
.xp-chip {display: inline-flex; align-items: center; gap: 6px; margin: 0 8px 6px 0; padding: 2px 10px; font: .85rem var(--px-body); background: var(--chip);
          border: 2px solid var(--line); border-radius: 0; color: var(--ink) !important; box-shadow: 2px 2px 0 var(--shadow);}
.xp-chip::before {content: ""; width: 9px; height: 9px; background: #aab0c8; border: 1px solid rgba(0,0,0,.4);}
.xp-chip.ok::before {background: #7fe08f;} .xp-chip.bad::before {background: #ff8d9b;}

/* sidebar = control panel */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background: linear-gradient(180deg, var(--side-a) 0%, var(--side-b) 100%) !important;}
[data-testid="stSidebar"] {border-right: 3px solid var(--frame);}
[data-testid="stSidebarContent"] {padding-bottom: 60px;}
[data-testid="stSidebar"] * {color: var(--ink);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: var(--navy);}
[data-testid="stSidebar"] .xp-title, [data-testid="stSidebar"] .xp-title * {color: var(--title-ink) !important;}
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"] * {color: var(--ink);}

/* inputs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {
    background: var(--face2) !important; border: 2px solid var(--line) !important; border-radius: 0 !important; box-shadow: inset 2px 2px 0 rgba(0,0,0,.1);}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="base-input"] input, .stApp [data-baseweb="textarea"] textarea, .stApp [data-baseweb="select"] * {
    color: var(--ink) !important; -webkit-text-fill-color: var(--ink);}
.stApp [data-baseweb="input"] input::placeholder, .stApp textarea::placeholder {color: var(--muted) !important; -webkit-text-fill-color: var(--muted); opacity: .7;}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within {border-color: var(--navy) !important; box-shadow: 0 0 0 2px var(--navy);}
.stApp [data-testid="stTextArea"] textarea {font-family: var(--px-code); font-size: 1.15rem; line-height: 1.2;}
.stApp [data-testid="stWidgetLabel"] p {font-weight: 700; color: var(--navy);}
.stApp [data-baseweb="select"] svg, .stApp [data-baseweb="input"] svg {fill: var(--ink);}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {background: var(--face2) !important; border: 2px solid var(--frame); border-radius: 0 !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {background: transparent; color: var(--ink) !important; font-family: var(--px-body);}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {background: var(--sel) !important;}
[data-baseweb="popover"] li:hover * {color: var(--sel-ink) !important;}

/* toggle */
.stApp [data-testid="stCheckbox"] label, .stApp [data-testid="stToggle"] label {color: var(--ink);}

/* buttons: bevelled pixel buttons, pastel green for the primary action */
.stApp [data-testid^="stBaseButton"] {border-radius: 0 !important; font-family: var(--px-body); font-weight: 700;}
.stApp [data-testid="stBaseButton-secondary"] {color: var(--ink) !important; border: 2px solid var(--btn-line) !important;
    background: linear-gradient(180deg, var(--btn-top) 0%, var(--btn-bot) 100%) !important;
    box-shadow: inset 2px 2px 0 rgba(255,255,255,.7), inset -2px -2px 0 rgba(0,0,0,.18), 3px 3px 0 var(--shadow);}
.stApp [data-testid="stBaseButton-secondary"] :where(p, span) {color: var(--ink) !important;}
.stApp [data-testid="stBaseButton-secondary"]:hover {outline: 2px solid #ffc857; outline-offset: -4px;}
.stApp [data-testid="stBaseButton-secondary"]:active {transform: translate(2px, 2px); box-shadow: inset 2px 2px 0 rgba(0,0,0,.18);}
.stApp [data-testid="stBaseButton-primary"] {border: 2px solid var(--go-line) !important;
    background: linear-gradient(180deg, var(--go-top) 0%, var(--go-bot) 100%) !important;
    box-shadow: inset 2px 2px 0 rgba(255,255,255,.65), inset -2px -2px 0 rgba(0,0,0,.16), 3px 3px 0 var(--shadow);}
.stApp [data-testid="stBaseButton-primary"] :where(p, span) {color: var(--go-ink) !important;}
.stApp [data-testid="stBaseButton-primary"]:hover {filter: brightness(1.06);}
.stApp [data-testid="stBaseButton-primary"]:active {transform: translate(2px, 2px);}
.stApp [data-testid^="stBaseButton"]:disabled {opacity: .5; filter: grayscale(.4);}

/* messages, code, expanders, patch cards */
.stApp [data-testid="stAlert"] {background: var(--face2); border: 2px solid var(--line); border-radius: 0; box-shadow: 3px 3px 0 var(--shadow);}
.stApp [data-testid="stAlert"], .stApp [data-testid="stAlert"] * {color: var(--ink);}
.stApp [data-testid="stCode"], .stApp [data-testid="stCodeBlock"] {border: 2px solid var(--line); border-radius: 0;}
.stApp [data-testid="stCode"] pre, .stApp [data-testid="stCodeBlock"] pre {background: var(--code-bg) !important;}
.stApp [data-testid="stCode"] pre, .stApp [data-testid="stCode"] code, .stApp [data-testid="stCodeBlock"] code {color: var(--ink);}
.stApp [data-testid="stExpander"] details {background: var(--face2); border: 2px solid var(--line) !important; border-radius: 0 !important; overflow: hidden;}
.stApp [data-testid="stExpander"] summary {background: linear-gradient(90deg, var(--exp-a) 0%, var(--exp-b) 100%);}
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * {color: var(--ink);}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 0;}
.stApp [data-testid="stVerticalBlockBorderWrapper"] > div {border-color: var(--line);}

/* progress bar (pastel green pixel blocks sliding across) */
.xp-prog {position: relative; overflow: hidden; height: 20px; margin: 2px 0 8px; padding: 3px; background: var(--face2); border: 2px solid var(--line);}
.xp-prog::after {content: ""; position: absolute; top: 3px; bottom: 3px; left: -30%; width: 30%;
                 background: repeating-linear-gradient(90deg, var(--prog) 0 8px, transparent 8px 12px); animation: xp-run 1.6s steps(14) infinite;}
@keyframes xp-run {to {left: 100%;}}

/* taskbar: a status strip only. pointer-events:none, so it can never swallow a click */
.xp-taskbar {position: fixed; left: 0; right: 0; bottom: 0; height: 38px; z-index: 999992; display: flex; align-items: center; gap: 8px;
             pointer-events: none; user-select: none; border-top: 3px solid var(--bar-line);
             background: linear-gradient(180deg, var(--bar-a) 0 50%, var(--bar-b) 50% 100%);}
.xp-start {display: flex; align-items: center; gap: 6px; height: 100%; padding: 0 20px 0 12px; font: 700 .95rem var(--px-head); text-transform: lowercase;
           color: var(--start-ink) !important; border-right: 3px solid var(--go-line);
           box-shadow: inset 2px 2px 0 rgba(255,255,255,.6), inset -2px -2px 0 rgba(0,0,0,.14);
           background: linear-gradient(180deg, var(--go-top) 0 50%, var(--go-bot) 50% 100%);}
.xp-start .xp-ico {width: 20px; height: 20px;}
.xp-task {display: flex; align-items: center; gap: 6px; height: 26px; padding: 0 12px; font: 700 .8rem var(--px-body); color: var(--task-ink) !important;
          background: linear-gradient(180deg, var(--task-a), var(--task-b)); border: 2px solid var(--bar-line); box-shadow: inset 2px 2px 0 rgba(0,0,0,.18);}
.xp-tray {margin-left: auto; height: 100%; display: flex; align-items: center; padding: 0 14px; font: .85rem var(--px-body); color: var(--ink) !important;
          background: linear-gradient(180deg, var(--tray-a) 0 50%, var(--tray-b) 50% 100%); border-left: 3px solid var(--bar-line);}

/* tabs */
.stApp [data-baseweb="tab-list"] {gap: 4px; border-bottom: 3px solid var(--frame); background: transparent;}
.stApp button[data-baseweb="tab"] {background: var(--btn-bot); border: 2px solid var(--line); border-bottom: 0; border-radius: 0; margin-bottom: -3px; height: 36px; padding: 0 12px;}
.stApp button[data-baseweb="tab"] p {font-family: var(--px-head); font-size: .72rem; color: var(--ink);}
.stApp button[data-baseweb="tab"][aria-selected="true"] {background: var(--face); border-color: var(--frame);}
.stApp [data-baseweb="tab-highlight"] {background: var(--navy) !important; height: 3px;}
.stApp [data-baseweb="tab-border"] {display: none;}

/* demo: flow strip */
.xp-flow {display: flex; align-items: stretch; gap: 6px; flex-wrap: wrap; margin: 0 0 22px;}
.xp-node {flex: 1 1 120px; min-width: 110px; padding: 8px 10px; background: var(--face); border: 3px solid var(--line); box-shadow: 4px 4px 0 var(--shadow); display: flex; flex-direction: column; gap: 2px;}
.xp-node b {font: 700 .78rem var(--px-head); color: var(--navy); letter-spacing: .5px; text-transform: uppercase;}
.xp-node span {font: .85rem var(--px-body); color: var(--muted);}
.xp-node.done {border-color: var(--go-line); background: linear-gradient(180deg, var(--go-top), var(--go-bot));}
.xp-node.done b, .xp-node.done span {color: var(--go-ink);}
.xp-node.active {border-color: #f0b429; background: var(--t-patch); animation: xp-blink 1s steps(2) infinite;}
.xp-node.bad {border-color: #d6566d; background: var(--t-warn);}
.xp-arrow {align-self: center; color: var(--navy); font-size: .8rem;}
@keyframes xp-blink {50% {box-shadow: 4px 4px 0 var(--shadow), 0 0 0 3px #f0b42966;}}

/* demo: tables, chunks, vectors, map */
.xp-table {width: 100%; border-collapse: collapse; font: .9rem var(--px-body); background: var(--face2); border: 2px solid var(--line); margin: 4px 0 8px;}
.xp-table th {text-align: left; padding: 4px 8px; background: linear-gradient(90deg, var(--exp-a), var(--exp-b)); color: var(--ink); font: 700 .72rem var(--px-head); border-bottom: 2px solid var(--line);}
.xp-table td {padding: 3px 8px; color: var(--ink); border-bottom: 1px dashed var(--line);}
.xp-chunks {display: flex; flex-direction: column; gap: 4px; margin: 6px 0;}
.xp-chunkrow {display: grid; grid-template-columns: 34px 1fr 110px; align-items: center; gap: 8px; font: .85rem var(--px-body); color: var(--ink);}
.xp-chunktrack {position: relative; height: 14px; background: var(--face2); border: 2px solid var(--line);}
.xp-chunkbar {position: absolute; top: 0; bottom: 0; border: 1px solid #2b2b45; opacity: .92;}
.xp-chunknote, .xp-note {margin-top: 6px; font: .85rem var(--px-body); color: var(--muted);}
.xp-vec {display: grid; grid-template-columns: repeat(48, 1fr); gap: 1px; padding: 3px; background: var(--face2); border: 2px solid var(--line); margin: 6px 0;}
.xp-vec i {display: block; aspect-ratio: 1; min-height: 8px;}
.xp-map {background: var(--face2); border: 2px solid var(--line); padding: 6px; margin: 6px 0;}
.xp-mapsvg {width: 100%; max-width: 680px; height: auto; display: block; margin: 0 auto;}
.xp-mapsvg .grid {stroke: var(--line); opacity: .35; stroke-width: 1;}
.xp-mapsvg .link {stroke: var(--navy); stroke-width: 1; stroke-dasharray: 3 3;}
.xp-mapsvg .ring {stroke: var(--ink); stroke-width: 2;}
.xp-mapsvg .query rect {fill: #ffd84a; stroke: #2b2b45; stroke-width: 1;}
.xp-mapsvg text {fill: var(--ink); font: 700 11px var(--px-head);}
.xp-legend {display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 6px; font: .8rem var(--px-body); color: var(--ink);}
.xp-leg i {display: inline-block; width: 10px; height: 10px; margin-right: 5px; border: 1px solid #2b2b45; vertical-align: -1px;}

/* demo: retrieval hits + agent trace */
.xp-seed {display: flex; flex-direction: column; gap: 6px;}
.xp-q {font: .85rem var(--px-body); color: var(--muted);}
.xp-hit {background: var(--face2); border: 2px solid var(--line); padding: 5px 8px;}
.xp-hithead {display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; color: var(--ink); font: .9rem var(--px-body);}
.xp-rank {display: inline-grid; place-items: center; width: 20px; height: 20px; background: var(--t-rag); border: 2px solid var(--line); font: 700 .7rem var(--px-head);}
.xp-from {padding: 0 6px; background: var(--t-llm); border: 1px solid var(--line); font-size: .78rem;}
.xp-lines, .xp-dist, .xp-dim {color: var(--muted);}
.xp-meter {display: inline-flex; gap: 2px;} .xp-meter i {width: 7px; height: 12px; background: var(--face); border: 1px solid var(--line);} .xp-meter i.on {background: var(--prog); border-color: var(--go-line);}
.xp-seed details, .xp-trace details {margin-top: 3px;} .xp-seed summary, .xp-trace summary {cursor: pointer; color: var(--navy); font: .85rem var(--px-body);}
.xp-seed pre, .xp-trace pre {margin: 4px 0 0; padding: 6px 8px; max-height: 220px; overflow: auto; background: var(--code-bg); border: 2px solid var(--line); color: var(--ink); font: 1.02rem/1.05 var(--px-code); white-space: pre;}
.xp-trace {display: flex; flex-direction: column; gap: 5px; margin: 4px 0 8px;}
.xp-ev {display: grid; grid-template-columns: 38px 62px 1fr; gap: 8px; align-items: start; font: .92rem var(--px-body); color: var(--ink);}
.xp-t {color: var(--muted); font-size: .8rem; padding-top: 2px; text-align: right;}
.xp-tag {display: inline-block; text-align: center; padding: 1px 0; border: 2px solid var(--line); font: 700 .66rem var(--px-head); color: var(--ink); background: var(--t-start);}
.xp-tag.rag {background: var(--t-rag);} .xp-tag.llm {background: var(--t-llm);} .xp-tag.tool {background: var(--t-tool);} .xp-tag.patch {background: var(--t-patch);}
.xp-tag.ok {background: var(--t-ok);} .xp-tag.warn {background: var(--t-warn);} .xp-tag.start {background: var(--t-start);}
.xp-evbody code {background: var(--code-bg); border: 1px solid var(--line); padding: 0 4px; font: 1rem var(--px-code); color: var(--ink);}
.xp-thought {margin-top: 3px; padding: 4px 8px; border-left: 4px solid var(--frame); background: var(--face2); color: var(--ink); font-style: italic;}
.xp-iter {margin: 4px 0 0 46px; font: 700 .68rem var(--px-head); color: var(--navy); letter-spacing: .5px; text-transform: uppercase; border-top: 2px dashed var(--line); padding-top: 4px;}
.xp-wait .xp-evbody {animation: xp-blink2 1s steps(2) infinite;} @keyframes xp-blink2 {50% {opacity: .3;}}
@media (max-width: 760px) {.xp-ev {grid-template-columns: 1fr;} .xp-t {text-align: left;} .xp-chunkrow {grid-template-columns: 30px 1fr;} .xp-chunklines {display: none;}}

@media (prefers-reduced-motion: reduce) {.xp-prog::after {animation: none; left: 0; width: 100%;} .xp-node.active, .xp-wait .xp-evbody {animation: none;}}
@media (max-width: 760px) {.xp-brand {font-size: 1.4rem;} .xp-task {display: none;}}

/* ── Retro-Pinterest polish: softer cards, calmer chrome, more breathing room ── */
.block-container {max-width: 1080px; padding-top: 2.4rem; padding-bottom: 108px;}
.stApp [data-testid="stMarkdownContainer"] p {line-height: 1.55;}

/* cards keep their pixel identity, but round off and drop the hard offset shadow */
[class*="st-key-xpwin"], .xp-hero {border: 2px solid var(--card-line) !important; border-radius: var(--radius) !important;
    box-shadow: var(--card-shadow) !important; margin-bottom: 20px;}
[class*="st-key-xpwin"] {padding: 0 18px 18px;}
.xp-hero {padding: 0;}
.xp-title {height: 36px; margin: 0 -18px 12px; padding: 0 8px 0 14px; border-bottom: 0;
    border-radius: var(--radius) var(--radius) 0 0; background: linear-gradient(180deg, var(--title-top), var(--title-bot));
    box-shadow: none; letter-spacing: .5px;}
.xp-hero .xp-title {margin: 0;}
.xp-title.flat {margin: 0 0 14px; border: 2px solid var(--card-line); border-radius: var(--radius-sm);}
.xp-hero-body {padding: 16px 20px 16px;}
.xp-brand {text-shadow: 2px 2px 0 var(--title-top);}
.xp-btns i {width: 18px; height: 18px; border-width: 0; border-radius: 6px; box-shadow: none;
    background: rgba(255,255,255,.45); font-size: 11px;}
.xp-btns i.x {background: #ffc2c9;}

/* chips: pastel pills */
.xp-chip {border: 2px solid var(--card-line); border-radius: 999px; box-shadow: none; padding: 3px 12px 3px 10px;}
.xp-chip::before {border: 0; border-radius: 999px;}

/* buttons: pill shaped, gentle lift on hover */
.stApp [data-testid^="stBaseButton"] {border-radius: 999px !important; padding-left: 1.1rem !important; padding-right: 1.1rem !important;
    transition: transform .12s ease, box-shadow .12s ease;}
.stApp [data-testid="stBaseButton-secondary"] {border: 2px solid var(--card-line) !important; background: var(--face2) !important;
    box-shadow: 0 2px 0 var(--shadow) !important;}
.stApp [data-testid="stBaseButton-secondary"]:hover {transform: translateY(-1px); box-shadow: 0 5px 12px var(--shadow) !important;}
.stApp [data-testid="stBaseButton-secondary"]:active {transform: translateY(1px);}
.stApp [data-testid="stBaseButton-primary"] {border: 2px solid var(--go-line) !important;
    background: linear-gradient(180deg, var(--go-top), var(--go-bot)) !important; box-shadow: 0 3px 0 rgba(0,0,0,.12) !important;}
.stApp [data-testid="stBaseButton-primary"]:hover {transform: translateY(-1px); box-shadow: 0 6px 14px rgba(0,0,0,.18) !important;}
.stApp [data-testid="stBaseButton-primary"]:active {transform: translateY(1px);}

/* inputs: softly rounded, quiet focus ring */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {
    border-radius: 12px !important; border-width: 1.5px !important; box-shadow: none !important;}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within {
    border-color: var(--frame) !important; box-shadow: 0 0 0 3px rgba(160,150,210,.28) !important;}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {border-radius: 12px !important; border-width: 1.5px !important;}

/* surfaces: alerts, code, expanders and the demo widgets share one soft radius */
.stApp [data-testid="stAlert"], .stApp [data-testid="stCode"], .stApp [data-testid="stCodeBlock"],
.stApp [data-testid="stExpander"] details, .xp-table, .xp-map, .xp-vec, .xp-hit, .xp-seed pre, .xp-trace pre,
.xp-thought, .xp-prog {border-radius: var(--radius-sm) !important; border-width: 1.5px !important; box-shadow: none !important;}
.stApp [data-testid="stExpander"] summary {border-radius: var(--radius-sm) var(--radius-sm) 0 0;}
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] {border-radius: var(--radius-sm); overflow: hidden;}
.xp-chunktrack {border-radius: 6px;}
.xp-rank {border-radius: 6px;} .xp-from {border-radius: 6px;}

/* tabs: rounded tops that match the cards */
.stApp button[data-baseweb="tab"] {border-radius: 12px 12px 0 0; border-width: 1.5px;}

/* sidebar + taskbar: tinted panels instead of saturated bars */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background: var(--side-grad) !important;}
[data-testid="stSidebar"] {border-right: 2px solid var(--card-line);}
.xp-taskbar {height: 42px; box-shadow: 0 -6px 18px rgba(0,0,0,.08);}
.xp-task {border-radius: 8px;}
.xp-start {border-top-right-radius: 8px; border-bottom-right-radius: 8px;}
"""


# ── Static HTML snippets (everything user-visible is escaped) ──
def title_bar(text: str, icon: str = "app", flat: bool = False, buttons: bool = True) -> str:
    btns = '<span class="xp-btns"><i>&#8211;</i><i>&#9633;</i><i class="x">&#215;</i></span>' if buttons else ""
    return (f'<div class="xp-title{" flat" if flat else ""}"><span class="xp-ico {html.escape(icon)}"></span>'
            f'<span>{html.escape(text)}</span>{btns}</div>')


def hero_html(chips) -> str:
    """chips: iterable of (text, state) with state in {"ok", "bad", ""}."""
    cells = "".join(f'<span class="xp-chip {html.escape(s)}">{html.escape(t)}</span>' for t, s in chips)
    return ('<div class="xp-hero">' + title_bar("CodePilot Local", "app") +
            '<div class="xp-hero-body"><div class="xp-brand">CodePilot</div>'
            '<div class="xp-sub">Describe the bug once. Watch it get investigated, then fixed.</div>'
            f'<div class="xp-chips">{cells}</div></div></div>')


def progress_html() -> str:
    return '<div class="xp-prog"></div>'


def taskbar_html(status: str = "") -> str:
    return ('<div class="xp-taskbar"><div class="xp-start"><span class="xp-ico flag"></span>start</div>'
            '<div class="xp-task"><span class="xp-ico app"></span>CodePilot.exe</div>'
            f'<div class="xp-tray">{html.escape(status)}</div></div>')
