"""Build the Lockoff logo set (SVG) from code, so it is exact and reproducible.

Mark: a horizontal 16:9 source frame with a vertical 9:16 clip locked inside it and
breaking out of it, plus a tally dot (the red "recording" light on a camera). One
solid colour + one accent, no gradients or glow.

usage: ..\\.venv\\Scripts\\python build_logo.py      â†’ branding/*.svg (+ PNG via ffmpeg/librsvg)
"""
from __future__ import annotations

from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).resolve().parent
FONT = HERE.parent / "lockoff" / "assets" / "fonts" / "Geist-Variable.ttf"
NAME = "Lockoff"

INK_DARK = "#ededed"      # mark/wordmark on dark backgrounds (graphite ramp primary text)
INK_LIGHT = "#171717"     # on light backgrounds
TILE = "#141415"          # app-icon tile (graphite page background)
TALLY = "#ff4d2e"         # accent: camera tally light


def mark(ink: str, cut: str | None, x: float = 0, y: float = 0, s: float = 1.0, uid: str = "m") -> str:
    """64x64 mark. `cut` = background colour used to separate the clip from the frame;
    None â†’ a transparent mask (works on any background)."""
    g = f'<g transform="translate({x} {y}) scale({s})">'
    # viewfinder: the whole 16:9 source frame, the 9:16 crop locked inside it
    # (a little right of centre â€” it follows a subject), REC tally top-left
    frame = f'<rect x="3" y="13" width="58" height="38" rx="9" fill="none" stroke="{ink}" stroke-width="4"/>'
    clip = f'<rect x="30" y="19.5" width="14.5" height="25" rx="3.5" fill="{ink}"/>'
    tally = f'<circle cx="12.5" cy="22.5" r="3.2" fill="{TALLY}"/>'
    return f'{g}{frame}{clip}{tally}</g>'


def wordmark_path(weight: int = 560, size: float = 40.0) -> tuple[str, float]:
    font = TTFont(FONT)
    inst = instantiateVariableFont(font, {"wght": weight}, inplace=False)
    gs = inst.getGlyphSet()
    cmap = inst.getBestCmap()
    upm = inst["head"].unitsPerEm
    sc = size / upm
    paths, x = [], 0.0
    for ch in NAME:
        gname = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[gname].draw(TransformPen(pen, (sc, 0, 0, -sc, x, 0)))
        paths.append(pen.getCommands())
        x += gs[gname].width * sc
    return " ".join(paths), x


def svg(w: float, h: float, body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:g} {h:g}" width="{w:g}" height="{h:g}">{body}</svg>\n'


def main() -> None:
    d, tw = wordmark_path()
    cap = 40 * 0.70            # Geist cap height â‰ˆ 0.70 em
    # horizontal lockup: mark 48px tall, wordmark baseline-aligned to the mark's optical centre
    for theme, ink in (("dark", INK_DARK), ("light", INK_LIGHT)):
        h, ms = 56, 56 / 64
        base = h / 2 + cap / 2
        body = mark(ink, None, 0, 0, ms, uid=f"l{theme}") + \
            f'<path d="{d}" fill="{ink}" transform="translate({h + 14} {base:.2f})"/>'
        (HERE / f"lockoff-logo-{theme}.svg").write_text(svg(h + 14 + tw + 2, h, body), encoding="utf-8")
        (HERE / f"lockoff-mark-{theme}.svg").write_text(svg(64, 64, mark(ink, None, uid=f"k{theme}")),
                                                         encoding="utf-8")
    # app icon: mark on a graphite rounded tile
    icon = f'<rect width="512" height="512" rx="112" fill="{TILE}"/>' + mark(INK_DARK, None, 64, 64, 6, uid="ic")
    (HERE / "lockoff-icon.svg").write_text(svg(512, 512, icon), encoding="utf-8")
    # social preview 1280x640 (GitHub repo card): wordmark + one line, quiet
    lw = 56 / 64
    body = (f'<rect width="1280" height="640" fill="{TILE}"/>'
            + mark(INK_DARK, None, 96, 96, 2.0, uid="sp")
            + f'<path d="{d}" fill="{INK_DARK}" transform="translate(96 {360}) scale(2.6)"/>')
    tag_d, _ = _line("Long video in. Hand-edited-looking vertical clips out.", 30, 400)
    tag2_d, _ = _line("Runs on your own PC. Free and open source.", 30, 400)
    body += f'<path d="{tag_d}" fill="#a1a1a6" transform="translate(98 440)"/>'
    body += f'<path d="{tag2_d}" fill="#85858b" transform="translate(98 488)"/>'
    (HERE / "lockoff-social.svg").write_text(svg(1280, 640, body), encoding="utf-8")
    print("wrote", ", ".join(sorted(p.name for p in HERE.glob("lockoff-*"))))


def _line(text: str, size: float, weight: int) -> tuple[str, float]:
    font = TTFont(FONT)
    inst = instantiateVariableFont(font, {"wght": weight}, inplace=False)
    gs, cmap, upm = inst.getGlyphSet(), inst.getBestCmap(), inst["head"].unitsPerEm
    sc = size / upm
    out, x = [], 0.0
    for ch in text:
        g = cmap.get(ord(ch))
        if g is None:
            continue
        pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (sc, 0, 0, -sc, x, 0)))
        out.append(pen.getCommands())
        x += gs[g].width * sc
    return " ".join(out), x


if __name__ == "__main__":
    main()
