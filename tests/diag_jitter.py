"""Camera-motion report for rendered speaker clips, from their sendcmd crop files.

usage: .venv\\Scripts\\python tests\\diag_jitter.py Jobs\\_e2e-ted
Per clip: jumps (frame-to-frame moves > 3% of crop width; hard cuts that are
NOT on a scene cut look like glitches), direction reversals within 1 s (wobble),
and mean |acceleration| (lower = smoother).
"""
import re
import sys
from pathlib import Path

import numpy as np


def series(cmd: Path, fps=30.0):
    pts = []
    for line in cmd.read_text().splitlines():
        m = re.match(r"([\d.]+) crop@rf x (\d+)", line)
        if m:
            pts.append((float(m.group(1)), int(m.group(2))))
    n = int(pts[-1][0] * fps) + 1
    x = np.zeros(n)
    j = 0
    for i in range(n):
        while j + 1 < len(pts) and pts[j + 1][0] <= i / fps + 1e-6:
            j += 1
        x[i] = pts[j][1]
    return x


def report(x, crop_w=608):
    v = np.diff(x)
    jumps = int(np.sum(np.abs(v) > crop_w * 0.03))
    moving = np.sign(v[np.abs(v) >= 1])
    rev = int(np.sum(moving[1:] != moving[:-1])) if moving.size > 1 else 0
    acc = float(np.mean(np.abs(np.diff(v)))) if v.size > 1 else 0.0
    still = float(np.mean(np.abs(v) < 0.5))
    return {"frames": len(x), "jumps": jumps, "reversals": rev, "mean_abs_acc_px": round(acc, 2),
            "still_share": round(still, 2)}


if __name__ == "__main__":
    for cmd in sorted(Path(sys.argv[1]).glob("_work/c*.cmd")):
        print(cmd.stem, report(series(cmd)))
