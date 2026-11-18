"""AI director: Claude decides WHO to frame, only where it matters.

Idea from the author's earlier Clip Studio research (experiment-4): show the model a frame
with every person drawn and NUMBERED, and have it pick one by number, so the
answer binds to a real track instead of invented coordinates.

Differences, all to save credits and fix its known weaknesses:
- Called ONLY for shots with 2+ significant faces. Single-speaker shots cost 0.
- Beats ≤ 3 s, capped per clip (default 12 calls); beyond the cap beats get longer.
- Each person's measured lip motion and the words spoken in the beat go into the
  prompt, so the model is not guessing who talks from one silent frame (the
  zip's biggest error source).
- The answer is a strong PRIOR on the camera's subject choice, not a hard
  override; the smooth camera still decides how to move.
- Cached per clip (_work/cNN.director.json): re-renders never re-spend credits.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..util import log, open_video

BEAT_MAX = 3.0
SIGNIFICANT_PRESENCE = 0.4    # on screen ≥40% of the shot
SIGNIFICANT_AREA = 0.25       # face ≥25% the area of the biggest face in the shot

PROMPT = """You are directing a vertical (9:16) crop of a horizontal video. Look at the image {img}.

Every person that matters is drawn with a numbered box ON THEIR FACE: {labels}.
Measured lip movement during this {dur:.1f}s moment (higher = more likely talking): {lips}.
Words spoken during this moment: "{words}"

Reply with ONLY a JSON object, no prose, no code fence:
{{"follow": <the number of the person the vertical crop should centre on>,
 "keep_both": <true if two people must stay in frame together, e.g. a close exchange, else false>,
 "reason": "<short reason>"}}

Prefer whoever is speaking or carrying the moment (use the lip numbers and words), not merely
the biggest or most central person. Pick a listener only if their reaction IS the moment."""


def significant_tracks(by: dict, shot) -> list[int]:
    in_shot = {k: [b for t, b in v.items() if shot.start <= t < shot.end] for k, v in by.items()}
    in_shot = {k: v for k, v in in_shot.items() if v}
    if len(in_shot) < 2:
        return []
    n_t = max(len(v) for v in in_shot.values())
    areas = {k: float(np.median([b.area for b in v])) for k, v in in_shot.items()}
    top = max(areas.values()) or 1.0
    keep = [k for k, v in in_shot.items() if len(v) >= SIGNIFICANT_PRESENCE * n_t and areas[k] >= SIGNIFICANT_AREA * top]
    return sorted(keep, key=lambda k: float(np.median([b.face_cx for b in in_shot[k]]))) if len(keep) >= 2 else []


def _beats(shot, n_cap: int) -> list[tuple[float, float]]:
    dur = shot.end - shot.start
    n = max(1, min(n_cap, int(round(dur / BEAT_MAX + 0.49))))
    step = dur / n
    return [(shot.start + i * step, shot.start + (i + 1) * step) for i in range(n)]


def _ask(img: Path, prompt: str, model: str, timeout: int = 180) -> dict:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found")
    r = subprocess.run([exe, "-p", prompt, "--allowedTools", "Read", "--model", model], cwd=str(img.parent),
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, stdin=subprocess.DEVNULL)
    txt = r.stdout.strip()
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e < 0:
        raise ValueError(f"no JSON in reply: {txt[:160]!r}")
    return json.loads(txt[s:e + 1])


class Director:
    def __init__(self, video: Path, cache: Path, words: list[dict] | None, model: str = "sonnet",
                 max_calls: int = 12):
        self.video, self.cache_path, self.words, self.model = video, cache, words or [], model
        self.left = max_calls
        self.cache = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        self.calls = 0

    def _frame(self, t: float, people: list, out: Path) -> bool:
        cap = open_video(self.video)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        cap.release()
        if not ok:
            return False
        for num, b in people:
            fw = (b.x2 - b.x1) / 2.8
            x1, y1 = int(b.face_cx - fw * 0.75), int(b.face_cy - fw * 0.9)
            x2, y2 = int(b.face_cx + fw * 0.75), int(b.face_cy + fw * 0.9)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 255), 3)
            lab = str(num)
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.rectangle(img, (x1, max(0, y1 - th - 14)), (x1 + tw + 14, y1), (0, 0, 0), -1)
            cv2.putText(img, lab, (x1 + 7, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 230, 255), 3)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True

    def decide(self, shot, by: dict) -> list[tuple[float, float, int | None, bool]]:
        """[(t0, t1, track_id or None, keep_both)] for a multi-person shot, else []."""
        sig = significant_tracks(by, shot)
        if not sig:
            return []
        out = []
        for t0, t1 in _beats(shot, max(1, min(4, self.left))):
            key = f"{t0:.2f}-{t1:.2f}:{','.join(map(str, sig))}"
            hit = self.cache.get(key)
            if hit is None:
                if self.left <= 0:
                    continue
                # representative moment: the sample in the beat where most significant people are visible
                ts = sorted({t for k in sig for t in by[k] if t0 <= t < t1})
                if not ts:
                    continue
                tm = max(ts, key=lambda t: (sum(1 for k in sig if t in by[k]), -abs(t - (t0 + t1) / 2)))
                people = [(i + 1, by[k][tm]) for i, k in enumerate(sig) if tm in by[k]]
                if len(people) < 2:  # only one person visible in this beat → nothing to decide, no credits
                    self.cache[key] = {"track": None, "keep_both": False, "reason": "one person visible"}
                    continue
                lips = []
                for i, k in enumerate(sig):
                    a = [b.mouth_act or 0.0 for t, b in by[k].items() if t0 <= t < t1]
                    lips.append(f"#{i + 1}: {np.mean(a) * 100:.1f}" if a else f"#{i + 1}: n/a")
                said = " ".join(w["w"] for w in self.words if t0 <= w["s"] < t1)[:300] or "(no speech)"
                img = self.cache_path.parent / f"{self.cache_path.stem}_{t0:.1f}.jpg"
                if not self._frame(tm, people, img):
                    continue
                prompt = PROMPT.format(img=img.name, labels=", ".join(f"#{n}" for n, _ in people),
                                       lips=", ".join(lips), words=said.replace('"', "'"), dur=t1 - t0)
                self.left -= 1
                self.calls += 1
                try:
                    ans = _ask(img, prompt, self.model)
                    num = ans.get("follow")
                    tid = sig[int(num) - 1] if isinstance(num, (int, float)) and 1 <= int(num) <= len(sig) else None
                    hit = {"track": tid, "keep_both": bool(ans.get("keep_both")), "reason": ans.get("reason", "")}
                    log(f"    director {t0:.1f}-{t1:.1f}s: follow #{num} ({hit['reason'][:60]})")
                except Exception as e:  # noqa: BLE001 — no answer → camera decides alone
                    log(f"    director {t0:.1f}-{t1:.1f}s failed ({str(e)[:80]}) — lip motion decides")
                    hit = {"track": None, "keep_both": False, "reason": "failed"}
                self.cache[key] = hit
                self.cache_path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
            out.append((t0, t1, hit["track"], hit["keep_both"]))
        return out
