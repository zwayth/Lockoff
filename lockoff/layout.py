"""Layout decision + ffmpeg filtergraphs for a 1080x1920 output.

  speaker  talking heads / podcasts / IRL: full-height 9:16 crop that follows the
           active speaker (virtual camera operator)
  facecam  gaming streams: small fixed webcam on top, 3:4 gameplay below, blurred
           backdrop (the studio's Stream Clips Rule B)
  general  no usable face: the whole frame fitted to width over a blurred copy
           (openshorts GENERAL) — nothing is cropped away
  fit      source is already vertical: scale/pad only
"""
from __future__ import annotations

W, H = 1080, 1920
TOP = 480
GAME_W, GAME_H = 1080, 1440

CAPTION_Y = {"speaker": 1380, "facecam": 620, "general": 1450, "fit": 1450}
HOOK_Y = {"speaker": 300, "facecam": 260, "general": 300, "fit": 300}


def decide(info: dict, analysis: dict, profile: str, forced: str = "auto") -> tuple[str, str]:
    if forced != "auto":
        return forced, "forced"
    if info["w"] / max(1, info["h"]) <= 0.65:
        return "fit", "source is already vertical"
    tr, p = analysis["tracks"], analysis.get("main")
    if analysis["with_face"] < 0.25 or not p:
        return "general", f"faces in only {analysis['with_face']:.0%} of frames"
    small_static = p["area"] < 0.035 and p["cx_std"] < 0.03 and p["cy_std"] < 0.03 and p["presence"] > 0.6
    off_centre = abs(p["cx"] / info["w"] - 0.5) > 0.22 or abs(p["cy"] / info["h"] - 0.5) > 0.22
    if small_static and (off_centre or profile == "stream"):
        return "facecam", f"small fixed face ({p['area']:.1%} of frame) at an edge → webcam overlay"
    if p["area"] < 0.004:
        return "general", f"main face too small to crop to ({p['area']:.2%} of frame)"
    return "speaker", f"{len(tr)} face track(s), main face {p['area']:.1%} of frame"


def facecam_box(info: dict, analysis: dict) -> tuple[int, int, int, int]:
    """Webcam region around the static face: head-and-shoulders, 16:9, clamped."""
    p = analysis.get("main") or analysis["tracks"][0]
    h = min(info["h"], p["h"] * 3.4)
    w = min(info["w"], h * 16 / 9)
    x = min(max(p["cx"] - w / 2, 0), info["w"] - w)
    y = min(max(p["cy"] - h * 0.45, 0), info["h"] - h)
    ev = lambda v: int(v) // 2 * 2
    return ev(x), ev(y), ev(w), ev(h)


def filtergraph(layout: str, info: dict, *, ass: str, crop: dict | None = None,
                face_box: tuple | None = None) -> str:
    sw, sh = info["w"], info["h"]
    pre = "[0:v]setpts=PTS-STARTPTS"
    bg = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"boxblur=24:2,eq=brightness=-0.15")
    if layout == "speaker":
        c = crop
        fc = [f"{pre},sendcmd=f={c['cmd']},crop@rf=w={c['w']}:h={c['h']}:x={c['x0']}:y={c['y0']},"
              f"scale={W}:{H}:flags=lanczos,setsar=1[comp]"]
    elif layout == "facecam":
        cw = min(sw, round(sh * 3 / 4)) // 2 * 2
        fx, fy, fw, fh = face_box
        fc = [f"{pre},split=3[a][b][c]", f"[a]{bg}[bg]",
              f"[b]crop={cw}:{sh}:{(sw - cw) // 2}:0,scale={GAME_W}:{GAME_H}[game]",
              f"[c]crop={fw}:{fh}:{fx}:{fy},scale={W - 40}:{TOP - 16}:force_original_aspect_ratio=decrease,"
              f"pad=iw+8:ih+8:4:4:color=black[face]",
              f"[bg][game]overlay=0:{TOP}[base]", f"[base][face]overlay=(W-w)/2:({TOP}-h)/2[comp]"]
    else:  # general / fit
        fc = [f"{pre},split=2[a][b]", f"[a]{bg}[bg]",
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg]",
              "[bg][fg]overlay=(W-w)/2:(H-h)/2[comp]"]
    fc.append(f"[comp]ass={ass},format=yuv420p[v]")
    return ";".join(fc)
