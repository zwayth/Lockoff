"""Virtual camera operator — plans one smooth crop path per shot, offline.

Replaces per-2.5 s re-solving (which snapped at every segment boundary and
flipped between single and group framing — measured: 30-50 jumps per clip).
Because the whole clip is known in advance, the camera can behave like a
calm human operator:

  1. WHO   pick the subject per moment: speaking (lip motion) + face size,
           with hysteresis — a challenger must clearly beat the current
           subject for 1.5 s, and never sooner than 3 s after the last switch.
           Two people who fit in frame together and are both active → frame
           the pair (decided once per subject segment, never flickering).
  2. CLEAN median filter kills detector spikes before the camera sees them.
  3. HOLD  lazy follow: the camera only moves when the face would leave a
           comfort zone around the current framing (dead zone).
  4. EASE  zero-phase Gaussian smoothing turns every move into an eased
           pan that starts slightly before the subject moves (anticipation).
  5. KEEP  containment: if a fast move would push the face out, the smallest
           smoothed correction is added — never a snap.
  Hard cuts happen only at scene cuts or a speaker switch across the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import BoxSample, Shot

DEAD_ZONE = 0.12         # of crop width: face may drift this far before the camera moves
CUT_REFRAME = 0.45        # a reframe further than this (of crop width) is a cut, never a whip pan
MAX_PAN = 0.55            # peak pan speed, crop widths per second
RECENTER = 0.15          # after a reframe the subject sits 15% of the dead zone off-centre, not at its edge
SMOOTH_SIGMA_S = 0.45    # ease length (Gaussian sigma, seconds)
MEDIAN_S = 0.7            # spike filter window
CONTAIN_MARGIN = 0.14     # face centre kept inside [14%, 86%] of the crop
SWITCH_MARGIN = 1.3       # challenger must beat the subject by 30% …
SWITCH_SUSTAIN_S = 1.5    # … for this long …
MIN_HOLD_S = 3.0          # … and not sooner than this after the last switch
CUT_JUMP = 0.35           # subject switch moving the frame further than this (of crop w) → cut
HEADROOM = 0.38           # face centre at 38% from the top when the crop is shorter than the source
MIN_TRACK_SAMPLES = 5


@dataclass
class FramePath:
    """Per-frame crop origin. Same role as the zip's CropPath, without keyframe limits."""
    source_w: int
    source_h: int
    crop_w: float
    crop_h: float
    fps: float
    xs: np.ndarray
    ys: np.ndarray
    cuts: list[float] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)

    def at(self, t: float) -> tuple[float, float, float, float]:
        i = int(min(max(round(t * self.fps), 0), len(self.xs) - 1))
        return float(self.xs[i]), float(self.ys[i]), self.crop_w, self.crop_h

    def to_json(self) -> dict:
        return {"source": [self.source_w, self.source_h], "crop": [round(self.crop_w), round(self.crop_h)],
                "fps": self.fps, "cuts": [round(c, 2) for c in self.cuts], "modes": self.modes,
                "x": np.round(self.xs).astype(int).tolist(), "y": np.round(self.ys).astype(int).tolist()}


def crop_window_size(source_w: int, source_h: int, target_w: int, target_h: int) -> tuple[float, float]:
    """Largest window of the target aspect ratio that fits inside the source frame."""
    if source_w / source_h >= target_w / target_h:
        return float(source_h) * target_w / target_h, float(source_h)
    return float(source_w), float(source_w) * target_h / target_w


# ------------------------------------------------------------------ signal helpers
def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _median(x: np.ndarray, win: int) -> np.ndarray:
    if win < 3 or len(x) < 3:
        return x.copy()
    h = win // 2
    pad = np.pad(x, h, mode="edge")
    return np.array([np.median(pad[i:i + win]) for i in range(len(x))])


def _fill(x: np.ndarray) -> np.ndarray:
    """Hold the nearest valid value across gaps (NaN)."""
    x = x.copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(ok)
    x[:first] = x[first]
    return x


def _camera_axis(target: np.ndarray, fps: float, crop: float, lo: float, hi: float,
                 dead: float = DEAD_ZONE, sigma_s: float = SMOOTH_SIGMA_S,
                 subj: np.ndarray | None = None, cuts_out: list | None = None) -> np.ndarray:
    """target = desired crop ORIGIN per frame. Returns a calm, eased origin path."""
    t_step = _median(_fill(target), max(3, int(MEDIAN_S * fps) | 1))  # median keeps true steps sharp
    t = _gauss(t_step, 0.15 * fps)    # tame detector noise before deciding anything
    n = len(t)
    dz = dead * crop
    # 1. decide HOLD positions: stay put until the subject leaves the comfort zone,
    #    then one decisive reframe that lands them near the centre
    holds: list[tuple[int, float]] = []
    cur = float(np.clip(np.median(t[: max(1, int(3 * fps))]), lo, hi))  # opening framing: where they are over 3 s
    holds.append((0, cur))
    for i in range(n):
        d = t[i] - cur
        if abs(d) > dz:
            cur = float(np.clip(t[i] - np.sign(d) * dz * RECENTER, lo, hi))
            if abs(cur - holds[-1][1]) >= 2:
                holds.append((i, cur))
    # 2. merge reframes closer than 0.8 s into one move (no stutter-steps)
    # no new move in the last 0.6 s of a shot: the cut reframes anyway, and a
    # pan squeezed against a cut looks like a stumble
    holds = [h for h in holds if h[0] == 0 or h[0] < n - 0.6 * fps]
    merged = [holds[0]]
    for i, v in holds[1:]:
        if len(merged) > 1 and i - merged[-1][0] < 0.8 * fps:
            merged[-1] = (merged[-1][0], v)
        else:
            merged.append((i, v))
    # 3. render: exact constant holds joined by S-curve (cosine) moves that start
    #    before the breach (anticipation) — still is perfectly still, moves are smooth
    c = np.full(n, merged[0][1])
    prev_end = 0
    for k in range(1, len(merged)):
        i, b = merged[k]
        a = merged[k - 1][1]
        # cut only if the subject TELEPORTED: the target jumps within ~0.12 s, which
        # only happens when the subject changes (detector/identity switch). A real
        # person, even stepping fast, takes several tenths of a second to cross —
        # that is followed with a pan, because a cut inside one continuous shot
        # reads as a jump cut (head-to-head test, 17 Nov: fast step at 1.8 s).
        w = max(1, int(0.12 * fps))
        lo_i, hi_i = max(0, i - int(fps)), min(n - 1, i + int(0.5 * fps))
        jump = float(np.max(np.abs(t_step[lo_i + w:hi_i + 1] - t_step[lo_i:hi_i + 1 - w]))) \
            if hi_i - lo_i > w else 0.0
        if abs(b - a) > CUT_REFRAME * crop and jump > CUT_REFRAME * crop * 0.8:
            c[i:] = b
            prev_end = i
            if cuts_out is not None:
                cuts_out.append(i)
            continue
        # S-curve peak speed = π/2·Δ/dur. Normally ≤ MAX_PAN; when the subject is
        # moving fast, match their speed (up to 1.3 crop widths/s) so the pan keeps
        # up smoothly instead of lagging and losing the face.
        w3 = max(1, int(0.5 * fps))
        v_subj = abs(t_step[min(n - 1, i + w3)] - t_step[max(0, i - w3)]) / (2 * w3 / fps)
        speed = float(np.clip(max(MAX_PAN * crop, 1.15 * v_subj), MAX_PAN * crop, 1.3 * crop))
        dur = int(np.clip(abs(b - a) * np.pi / 2 / speed * fps, 0.4 * fps, 2.5 * fps))
        s0 = max(prev_end, i - int(dur * 0.6))
        s1 = min(n, s0 + dur)
        if s1 <= s0:
            c[s0:] = b
            continue
        u = np.arange(s1 - s0) / max(1, s1 - s0 - 1)
        c[s0:s1] = a + (b - a) * (1 - np.cos(np.pi * u)) / 2
        c[s1:] = b
        prev_end = s1
    # 4. safety: if the subject still escapes (very fast move), follow it tightly
    #    for just those frames, eased in/out — the face is never lost
    if subj is not None:
        s = _fill(subj)
        m = CONTAIN_MARGIN * crop
        over = np.maximum(s - (c + crop - m), 0) - np.maximum((c + m) - s, 0)
        if np.max(np.abs(over)) >= 1:
            c = c + _gauss(over, 0.2 * fps)
    return np.clip(c, lo, hi)


# ------------------------------------------------------------------ who to frame
DIRECTOR_BONUS = 0.6   # how strongly an AI-director pick outweighs lip motion + size (scores are 0..1)


def _choose_subjects(samples: list[BoxSample], times: np.ndarray, prior_fn=None
                     ) -> tuple[list[int | None], dict, list]:
    by: dict[int, dict[float, BoxSample]] = {}
    for s in samples:
        by.setdefault(s.track_id, {})[round(s.t, 3)] = s
    by = {k: v for k, v in by.items() if len(v) >= MIN_TRACK_SAMPLES}
    if not by:
        return [None] * len(times), by, []
    prior = prior_fn(by) if prior_fn else []   # [(t0, t1, track, keep_both)] from the AI director
    max_area = max(b.area for v in by.values() for b in v.values()) or 1.0
    # speaking: per-track rolling mean of lip motion over ±1 s
    acts = {}
    for k, v in by.items():
        ts = np.array(sorted(v))
        a = np.array([v[t].mouth_act if v[t].mouth_act is not None else 0.0 for t in ts])
        acts[k] = (ts, a)
    peak = max((float(a.max()) for _, a in acts.values() if a.size), default=0.0) or 1.0

    def score(k: int, t: float) -> float:
        b = by[k].get(round(t, 3))
        if b is None:
            return 0.0
        ts, a = acts[k]
        win = a[(ts >= t - 1.0) & (ts <= t + 1.0)]
        speak = float(win.mean()) / peak if win.size else 0.0
        bonus = DIRECTOR_BONUS if any(p[2] == k and p[0] <= t < p[1] for p in prior) else 0.0
        return 0.5 * speak + 0.5 * (b.area / max_area) ** 0.5 + bonus

    sc = {k: np.array([score(k, t) for t in times]) for k in by}
    # opening subject: best average over the first second
    first = times <= times[0] + 1.0
    cur = max(sc, key=lambda k: (sc[k][first].mean(), -k))
    out: list[int | None] = []
    last_switch = times[0] - MIN_HOLD_S
    lead_since: dict[int, float] = {}
    i = 0
    while i < len(times):
        t = times[i]
        inc = sc[cur][i]
        best = max(sc, key=lambda k: (sc[k][i], -k))
        if best != cur and sc[best][i] > max(inc, 0.05) * SWITCH_MARGIN:
            lead_since.setdefault(best, t)
        for k in list(lead_since):
            if k == cur or sc[k][i] <= max(inc, 0.05) * SWITCH_MARGIN:
                lead_since.pop(k)
        ready = [k for k, s in lead_since.items() if t - s >= SWITCH_SUSTAIN_S and s - last_switch >= MIN_HOLD_S]
        if ready:
            new = max(ready, key=lambda k: sc[k][i])
            start = lead_since[new]
            j = int(np.searchsorted(times, start))       # switch where they STARTED leading
            out = out[:j]
            i = j
            cur, last_switch = new, start
            lead_since.clear()
            continue
        out.append(cur)
        i += 1
    return out, by, prior


# ------------------------------------------------------------------ plan
def plan(info: dict, boxes: list[BoxSample], shots: list[Shot], crop_w: float, crop_h: float,
         saliency=None, fps: float = 30.0, director=None) -> FramePath:
    sw, sh, dur = info["w"], info["h"], info["duration"]
    n = int(np.ceil(dur * fps)) + 1
    xs, ys = np.zeros(n), np.zeros(n)
    max_x, max_y = max(0.0, sw - crop_w), max(0.0, sh - crop_h)
    cuts: list[float] = []
    modes: list[str] = []

    def solve(fa: int, fb: int, cx: np.ndarray, cy: np.ndarray, calm: bool = False,
              face_w: float = 0.0) -> None:
        tx = cx - crop_w / 2
        ty = cy - crop_h * HEADROOM if max_y > 0 else np.zeros_like(cy)
        # close-ups: a head turn moves a big face a lot — widen the comfort zone with face size
        dead = max(DEAD_ZONE, 0.45 * face_w / crop_w) if face_w else DEAD_ZONE
        kw = {"dead": 0.25, "sigma_s": 1.0} if calm else {"dead": min(dead, 0.25)}
        local: list[int] = []
        xs[fa:fb] = _camera_axis(tx, fps, crop_w, 0, max_x, subj=None if calm else cx, cuts_out=local, **kw)
        cuts.extend((fa + i) / fps for i in local)
        ys[fa:fb] = _camera_axis(ty, fps, crop_h, 0, max_y, subj=None if calm or max_y == 0 else cy, **kw) \
            if max_y > 0 else 0.0

    for sh_ in shots:
        fa, fb = int(round(sh_.start * fps)), min(n, int(round(sh_.end * fps)) + 1)
        if fb - fa < 2:
            continue
        if fa > 0:
            cuts.append(sh_.start)
        frame_t = np.arange(fa, fb) / fps
        ins = [b for b in boxes if sh_.start <= b.t < sh_.end and b.face_cx is not None]
        times = np.array(sorted({round(b.t, 3) for b in ins}))
        prior_fn = (lambda by_, s=sh_: director.decide(s, by_)) if director else None
        subj, by, prior = _choose_subjects(ins, times, prior_fn) if len(times) else ([], {}, [])
        if not by:
            pts = saliency(sh_) if saliency else []
            if pts:
                st = np.array([p[0] for p in pts])
                cx = np.interp(frame_t, st, [p[1] for p in pts])
                cy = np.interp(frame_t, st, [p[2] for p in pts])
                modes.append("saliency")
            else:
                cx, cy = np.full(len(frame_t), sw / 2), np.full(len(frame_t), sh / 2)
                modes.append("center")
            solve(fa, fb, cx, cy, calm=True)
            continue

        # subject segments → target per sample; pair framing decided per segment
        seg_bounds = [0] + [i for i in range(1, len(subj)) if subj[i] != subj[i - 1]] + [len(subj)]
        tcx = np.full(len(times), np.nan)
        tcy = np.full(len(times), np.nan)
        tfw = np.full(len(times), np.nan)   # face width of the framed subject (body box ≈ 2.8 face widths)
        seg_info = []
        for a, b in zip(seg_bounds, seg_bounds[1:]):
            k = subj[a]
            seg_t = times[a:b]
            main = [by[k].get(t) for t in seg_t]
            # partner: another track present most of the segment whose face fits in frame with ours
            pair = None
            for o in by:
                if o == k:
                    continue
                both = [(m, by[o].get(t)) for m, t in zip(main, seg_t) if m is not None and by[o].get(t) is not None]
                if len(both) < 0.6 * len(seg_t):
                    continue
                fits = [abs(m.face_cx - p.face_cx) + (m.x2 - m.x1) * 0.35 < crop_w * (1 - 2 * CONTAIN_MARGIN)
                        for m, p in both]
                act_o = np.mean([p.mouth_act or 0 for _, p in both])
                act_k = np.mean([m.mouth_act or 0 for m, _ in both]) or 1e-6
                # the AI director asked to keep both people in frame during this segment
                wants_both = any(p[3] and p[0] < seg_t[-1] and seg_t[0] < p[1] for p in prior)
                if np.mean(fits) > 0.85 and (act_o > 0.5 * act_k or wants_both):
                    pair = o
                    break
            for j, t in enumerate(seg_t):
                m = main[j]
                if m is None:
                    continue
                if pair is not None and (p := by[pair].get(t)) is not None:
                    tcx[a + j] = (m.face_cx + p.face_cx) / 2
                else:
                    tcx[a + j] = m.face_cx
                tcy[a + j] = m.face_cy
                tfw[a + j] = (m.x2 - m.x1) / 2.8
            seg_info.append((times[a], k, pair))

        # split at speaker switches that move the frame a lot → clean cut, else smooth pan
        split_frames = [fa]
        filled = _fill(tcx)
        for (t0, _, _), (t1, _, _) in zip(seg_info, seg_info[1:]):
            i1 = int(np.searchsorted(times, t1))
            before = np.nanmedian(filled[max(0, i1 - 10):i1]) if i1 > 0 else filled[0]
            after = np.nanmedian(filled[i1:i1 + 10])
            if abs(after - before) > CUT_JUMP * crop_w:
                split_frames.append(fa + int(round((t1 - sh_.start) * fps)))
                cuts.append(t1)
        split_frames.append(fb)
        for s0, s1 in zip(split_frames, split_frames[1:]):
            if s1 - s0 < 2:
                continue
            ft = np.arange(s0, s1) / fps
            # strictly inside this piece: a sample from across the cut would make
            # the camera drift toward the next speaker just before cutting to them
            sel = (times >= s0 / fps - 1e-3) & (times < s1 / fps - 1e-3) & ~np.isnan(tcx)
            if not sel.any():
                xs[s0:s1] = xs[s0 - 1] if s0 > 0 else max_x / 2
                continue
            cx = np.interp(ft, times[sel], tcx[sel])
            cy = np.interp(ft, times[sel], _fill(tcy)[sel])
            fw = tfw[sel]
            solve(s0, s1, cx, cy, face_w=float(np.nanmedian(fw)) if np.any(~np.isnan(fw)) else 0.0)
        base = "pair" if any(p is not None for _, _, p in seg_info) else "subject"
        modes.append(base + ("+director" if prior else ""))
    if n > 1 and fb < n:  # tail after the last shot: hold
        xs[fb:] = xs[fb - 1]
        ys[fb:] = ys[fb - 1]
    return FramePath(sw, sh, crop_w, crop_h, fps, xs, ys, sorted(set(round(c, 3) for c in cuts)), modes)


def motion_report(path: FramePath) -> dict:
    """QA numbers for camera motion: glitch jumps (not at a cut), wobble, smoothness."""
    v = np.diff(path.xs)
    cut_frames = {int(round(c * path.fps)) for c in path.cuts}
    near_cut = lambda i: any(abs(i + 1 - c) <= 2 for c in cut_frames)
    # a glitch is a sudden CHANGE of speed (jerk), not a fast smooth pan
    a = np.diff(v)
    big = [i for i in np.where(np.abs(a) > path.crop_w * 0.013)[0] if not near_cut(i) and not near_cut(i + 1)]
    keep = np.array([not near_cut(i) for i in range(len(v))], dtype=bool)
    vk = v[keep] if len(v) else v
    mv = np.sign(vk[np.abs(vk) >= 0.5])
    rev = int(np.sum(mv[1:] != mv[:-1])) if mv.size > 1 else 0
    acc = np.abs(np.diff(v))
    acc = np.delete(acc, [i for i in range(len(acc)) if near_cut(i) or near_cut(i + 1)]) if len(acc) else acc
    return {"glitch_jumps": len(big), "cuts": len(path.cuts),
            "reversals_per_min": round(rev / max(1e-6, len(path.xs) / path.fps / 60), 1),
            "mean_abs_acc_px": round(float(acc.mean()) if acc.size else 0.0, 3),
            "still_share": round(float(np.mean(np.abs(vk) < 0.5)) if len(vk) else 1.0, 2),
            "max_speed_cropw_per_s": round(float(np.abs(vk).max() * path.fps / path.crop_w) if len(vk) else 0, 2)}
