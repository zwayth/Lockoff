"""Word-by-word karaoke captions (ASS) + hook title card for the first seconds.

Caption look = the studio's Stream Clips Rule C: Impact, uppercase, white with a
black outline + shadow; the spoken word turns yellow (#FFE500) and pops to 115%.
Words come from the main transcript (no second transcription pass) and are
stored in the job plan, so a misheard word is fixed by editing plan.json and
running `rerender`.
"""
from __future__ import annotations

import re
from pathlib import Path

W, H = 1080, 1920

STYLES = {
    "hormozi": {"font": "Impact", "size": 78, "per": 3, "upper": True},
    "clean": {"font": "Arial", "size": 64, "per": 4, "upper": False},
}


def _t(x: float) -> str:
    cs = max(0, int(round(x * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _esc(s: str) -> str:
    return re.sub(r"[{}\\]", "", s).replace("\n", " ")


def merge_fragments(words: list[dict]) -> list[dict]:
    """Whisper sometimes splits "all-nighters" / "don't" into separate tokens."""
    out: list[dict] = []
    for w in words:
        if out and re.match(r"^[-'’]", w["w"]) and w["s"] - out[-1]["e"] < 0.15:
            out[-1] = {**out[-1], "w": out[-1]["w"] + w["w"], "e": w["e"]}
        else:
            out.append(dict(w))
    return out


def phrases(words: list[dict], per: int) -> list[list[dict]]:
    out, cur = [], []
    for w in merge_fragments(words):
        if cur and (len(cur) >= per or w["s"] - cur[-1]["e"] > 0.6 or re.search(r"[.?!,]$", cur[-1]["w"])):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _wrap_hook(text: str, max_chars: int = 20) -> str:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines[:3])


def write_ass(path: Path, words: list[dict], *, caption_y: int, style: str = "hormozi",
              hook: str | None = None, hook_secs: float = 3.2, hook_y: int = 260,
              duration: float = 0.0) -> int:
    st = STYLES.get(style, STYLES["hormozi"])
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,1,0,1,6,3,5,40,40,0,1
Style: Hook,Arial Black,70,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,3,18,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if hook:
        end = min(hook_secs, duration) if duration else hook_secs
        ev.append(f"Dialogue: 1,{_t(0)},{_t(end)},Hook,,0,0,0,,"
                  f"{{\\pos(540,{hook_y})\\fad(120,200)}}{_wrap_hook(_esc(hook))}")
    norm = (lambda s: s.upper()) if st["upper"] else (lambda s: s)
    phs = phrases(words, st["per"])
    for ph in phs:
        for i, w in enumerate(ph):
            start = w["s"]
            end = ph[i + 1]["s"] if i + 1 < len(ph) else max(w["e"], w["s"] + 0.3)
            parts = []
            for j, x in enumerate(ph):
                txt = norm(_esc(x["w"]))
                parts.append("{\\c&H00E5FF&\\fscx115\\fscy115}" + txt + "{\\c&HFFFFFF&\\fscx100\\fscy100}"
                             if j == i else txt)
            ev.append(f"Dialogue: 0,{_t(start)},{_t(end)},Cap,,0,0,0,,"
                      f"{{\\pos(540,{caption_y})}}" + " ".join(parts))
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")
    return len(phs)
