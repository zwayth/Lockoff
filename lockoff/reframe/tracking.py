"""Track post-processing — robustness layer over raw ByteTrack output.

ByteTrack fragments one person into many short-lived IDs through occlusion,
brief detection loss, and low-light dropouts (a 20s clip yielded 23 IDs in
the field diagnostics). Fragmented identity silently breaks everything
downstream: persistence is undercounted, incumbency resets mid-shot ("wrong
face" / instability), and the speaker signal can't stay attached to a
person. This module repairs identity BEFORE ranking sees it.

Everything here is pure and deterministic (N4): sorted iteration, no
randomness, no model calls.
"""
from __future__ import annotations

import dataclasses

from .types import BoxSample

# Stitching thresholds. Two fragments are the same person when the later one
# starts soon after the earlier ends, near where it left off, at a similar
# size. Generous on gap/distance (fragmentation is common) but guarded by
# size + no temporal overlap so distinct people are never merged.
STITCH_GAP_SECONDS = 1.2
STITCH_DIST_FRAC = 0.18     # of frame width, between fragment endpoints
STITCH_SIZE_RATIO = 2.2     # max box-area ratio to still be "same person"
MOUTH_SMOOTH_WINDOW = 2     # ± samples for rolling mouth-activity mean


def _find(parent: dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def stitch_tracks(samples: list[BoxSample], *, frame_w: int,
                  frame_h: int) -> list[BoxSample]:
    """Merge fragmented track IDs into continuous identities.

    For each track (by first appearance) find the best earlier track whose
    end connects to this one's start — small time gap, small spatial jump,
    similar size, no temporal overlap. Union-find so chains collapse
    (A→B→C become one). Returns samples with rewritten track_ids.
    """
    if not samples:
        return samples

    by_track: dict[int, list[BoxSample]] = {}
    for s in samples:
        by_track.setdefault(s.track_id, []).append(s)
    for ss in by_track.values():
        ss.sort(key=lambda b: b.t)

    tracks = sorted(by_track.items(), key=lambda kv: (kv[1][0].t, kv[0]))
    ends = {tid: ss[-1] for tid, ss in tracks}
    starts = {tid: ss[0] for tid, ss in tracks}

    parent = {tid: tid for tid, _ in tracks}
    max_dist = frame_w * STITCH_DIST_FRAC

    for i, (tid, ss) in enumerate(tracks):
        start = starts[tid]
        best_pred, best_cost = None, None
        for ptid, _ in tracks[:i]:
            end = ends[ptid]
            gap = start.t - end.t
            if gap <= 0 or gap > STITCH_GAP_SECONDS:
                continue  # overlap in time, or too far apart
            dist = ((start.cx - end.cx) ** 2 + (start.cy - end.cy) ** 2) ** 0.5
            if dist > max_dist:
                continue
            a1, a2 = start.area, end.area
            if max(a1, a2) > STITCH_SIZE_RATIO * max(1.0, min(a1, a2)):
                continue
            # Prefer the smallest gap, then smallest spatial jump.
            cost = (gap, dist)
            if best_cost is None or cost < best_cost:
                best_pred, best_cost = ptid, cost
        if best_pred is not None:
            parent[_find(parent, tid)] = _find(parent, best_pred)

    return [dataclasses.replace(s, track_id=_find(parent, s.track_id))
            for s in samples]


def smooth_mouth_activity(samples: list[BoxSample], *,
                          window: int = MOUTH_SMOOTH_WINDOW) -> list[BoxSample]:
    """Rolling-mean the per-sample mouth-activity within each track.

    Raw lip-motion is spiky — a single chew/blink/compression artefact can
    spike it, and a speaker who pauses for one sample dips to zero. Averaging
    over ±window samples makes "who is speaking" stable across a segment
    instead of flickering frame to frame.
    """
    if not samples:
        return samples
    by_track: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_track.setdefault(s.track_id, []).append(i)

    out = list(samples)
    for idxs in by_track.values():
        idxs.sort(key=lambda i: samples[i].t)
        acts = [samples[i].mouth_act for i in idxs]
        for j, i in enumerate(idxs):
            lo, hi = max(0, j - window), min(len(idxs), j + window + 1)
            vals = [a for a in acts[lo:hi] if a is not None]
            if vals:
                out[i] = dataclasses.replace(samples[i], mouth_act=sum(vals) / len(vals))
    return out
