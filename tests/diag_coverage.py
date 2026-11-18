"""Diagnostic: list sampled moments where the main face is outside the reframe crop.

usage: .venv\\Scripts\\python tests\\diag_coverage.py Jobs\\<job>\\_work\\c02_slice.mp4
"""
import sys
from pathlib import Path

from lockoff.reframe import detect as D
from lockoff.reframe import pipeline as RP
from lockoff.util import probe

v = Path(sys.argv[1])
info = probe(v)
a = D.analyse(v, info)
p, modes = RP.analyze_clip(v, info, a, 1080, 1920)
print("shots", [round(s.start, 1) for s in D.detect_shots(v, info["duration"])], modes)
floor = a["main"]["area"] * 0.35
for t, fs in zip(a["sampled"]["times"], a["sampled"]["faces"]):
    b = max(fs, key=lambda f: f["w"] * f["h"], default=None)
    if not b or b["w"] * b["h"] / (info["w"] * info["h"]) < floor:
        continue
    x, y, w, h = p.at(t)
    cx = b["x"] + b["w"] / 2
    if not x <= cx <= x + w:
        print(f"t={t:5.1f} face cx={cx:6.0f} fw={b['w']:4.0f} crop=[{x:6.0f},{x + w:6.0f}] faces={len(fs)}")
