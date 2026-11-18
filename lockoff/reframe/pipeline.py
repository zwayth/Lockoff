"""Speaker reframe: detections → identity repair → virtual camera plan → ffmpeg sendcmd.

Identity repair (track stitching, mouth-activity smoothing) and the
detection approach come from the author's earlier Clip Studio prototype. The camera itself
is smooth.py: one offline plan per shot, held still by default, eased moves,
cuts only at scene cuts or speaker switches. (The zip's per-2.5 s re-solve
produced 30-50 visible jumps per clip on the TED test.)
"""
from __future__ import annotations

from pathlib import Path

from ..util import log
from . import detect as D
from .smooth import crop_window_size
from .smooth import FramePath, motion_report, plan
from .tracking import smooth_mouth_activity, stitch_tracks


def analyze_clip(video: Path, info: dict, analysis: dict, target_w: int, target_h: int,
                 fps: float = 30.0, director=None) -> tuple[FramePath, list[str]]:
    """director: a reframe.director.Director, or None (lip motion + size decide alone)."""
    sw, sh, dur = info["w"], info["h"], info["duration"]
    cw, ch = crop_window_size(sw, sh, target_w, target_h)
    shots = D.detect_shots(video, dur)
    people = analysis["boxes"]
    if people:
        raw = len({s.track_id for s in people})
        people = smooth_mouth_activity(stitch_tracks(people, frame_w=sw, frame_h=sh))
        log(f"  reframe: {len(shots)} shot(s), {raw} raw face tracks → {len({s.track_id for s in people})} stitched")
    path = plan(info, people, shots, cw, ch, saliency=lambda shot: D.saliency_centers(video, shot), fps=fps,
                director=director)
    if director is not None:
        log(f"  AI director: {director.calls} Claude call(s) this render "
            f"({'multi-person shots found' if director.calls or director.cache else 'single-speaker clip — no calls needed'})")
    return path, path.modes


def sendcmd(path: FramePath, duration: float, fps: float, target: str = "crop@rf") -> tuple[str, int, int, int, int]:
    """Per-frame crop commands, deduplicated. Returns (text, crop_w, crop_h, x0, y0)."""
    cw = int(round(path.crop_w / 2) * 2)
    ch = int(round(path.crop_h / 2) * 2)
    max_x, max_y = path.source_w - cw, path.source_h - ch
    lines, last = [], None
    x0 = y0 = 0
    for i in range(int(duration * fps) + 2):
        x, y, _, _ = path.at(i / fps)
        xi = int(min(max(round(x), 0), max_x))
        yi = int(min(max(round(y), 0), max_y))
        if i == 0:
            x0, y0 = xi, yi
        if (xi, yi) != last:
            lines.append(f"{i / fps:.3f} {target} x {xi}, {target} y {yi};")
            last = (xi, yi)
    return "\n".join(lines) + "\n", cw, ch, x0, y0


def face_coverage(path: FramePath, analysis: dict) -> float:
    """QA: share of sampled moments where a PROMINENT face (≥60% of the biggest
    face's area) is inside the crop. Crowd shots (5+ faces) are skipped."""
    tot = ok = 0
    main = analysis.get("main")
    floor = (main["area"] * 0.35) if main else 0.0
    for t, faces in zip(analysis["sampled"]["times"], analysis["sampled"]["faces"]):
        if not faces or len(faces) >= 5:
            continue
        big_a = max(f["w"] * f["h"] for f in faces)
        if big_a / (path.source_w * path.source_h) < floor:
            continue
        tot += 1
        x, y, w, h = path.at(t)
        if any(x <= f["x"] + f["w"] / 2 <= x + w and y <= f["y"] + f["h"] / 2 <= y + h
               for f in faces if f["w"] * f["h"] >= 0.6 * big_a):
            ok += 1
    return round(ok / tot, 3) if tot else 1.0


__all__ = ["analyze_clip", "sendcmd", "face_coverage", "motion_report"]
