"""The retro theme must stay presentation-only: no scripts, escaped text, and the CSS hooks the page relies on."""
import re

import xp_theme as t


def test_theme_has_no_script_and_all_placeholders_are_filled():
    css = t.theme_css()
    assert "<script" not in css.lower() and "javascript:" not in css.lower()
    assert "/*DESKTOP*/" not in css and "/*ICONS*/" not in css


def test_page_windows_have_css_hooks():
    css = t.theme_css()
    assert 'class*="st-key-xpwin"' in css                       # the keyed st.container() windows
    for icon in t.ICONS:
        assert f".xp-ico.{icon}" in css


def test_user_text_is_escaped_in_static_html():
    for out in (t.title_bar("<img src=x>", "app"), t.hero_html([("<b>x</b>", "ok")]), t.taskbar_html("<script>1</script>")):
        assert "<script" not in out and "<img" not in out and "<b>" not in out


def test_decorations_never_capture_clicks():
    assert re.search(r"\.xp-taskbar \{[^}]*pointer-events: none", t.theme_css())
    assert re.search(r"\.xp-btns \{[^}]*pointer-events: none", t.theme_css())


def test_light_and_dark_variants_differ_but_share_structure():
    light, dark = t.theme_css(False), t.theme_css(True)
    assert light != dark
    assert "--face: #faf5ec" in light and "--face: #2a2635" in dark
    assert "/*VARS*/" not in light and "/*VARS*/" not in dark
    assert "--scheme: light" in light and "--scheme: dark" in dark
    for css in (light, dark):
        assert "<script" not in css.lower() and "pointer-events: none" in css


def test_pixel_fonts_and_scenes():
    css = t.theme_css()
    assert "Pixelify Sans" in css and "Silkscreen" in css and "VT323" in css
    for mode in ("light", "dark"):
        svg = t._scene_svg(mode)
        assert svg.startswith("<svg") and "crispEdges" in svg and "<script" not in svg
    for name, svg in t.ICONS.items():
        assert svg.count("<rect") > 4, name
