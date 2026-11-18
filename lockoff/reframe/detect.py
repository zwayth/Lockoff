"""Face sampling for reframing + layout decisions.

YuNet (OpenCV, 230 KB ONNX, CPU-fast) replaces the zip's YOLO+InsightFace stack:
one detector gives face box + 5 landmarks, so the mouth patch is located from the
real mouth corners (the zip had to guess it from the person box). Faces are
tracked with a greedy distance/size matcher, then turned into the zip's
BoxSample contract (a body-sized box around each face, face centre, mouth
activity) so its stitching, ranking and camera-path solver run unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np

from ..util import MODELS, log, open_video
from .types import BoxSample, Shot

ANALYSIS_W = 640
SCORE_MIN = 0.6
DARK_LUMA = 70.0
MATCH_GAP_S = 1.0


def _enhance(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    if float(yuv[:, :, 0].mean()) >= DARK_LUMA:
        return frame
    yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def sample_faces(video: Path, *, fps: float, sample_fps: float = 10.0) -> dict:
    """Detect faces on a fixed time grid. Returns {"times": [...], "faces": [[face,...],...]}
    with every face = {x,y,w,h,score,mouth:[mx,my,mw], landmarks} in SOURCE pixels."""
    det = None
    cap = open_video(video)
    stride = max(1, round(fps / sample_fps))
    times, faces_all = [], []
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % stride:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            idx += 1
            h, w = frame.shape[:2]
            scale = min(1.0, ANALYSIS_W / w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
            small = _enhance(small)
            if det is None:
                det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "",
                                                (small.shape[1], small.shape[0]), SCORE_MIN, 0.3, 50)
            det.setInputSize((small.shape[1], small.shape[0]))
            _, res = det.detect(small)
            found = []
            gray = None
            for r in (res if res is not None else []):
                x, y, fw, fh = (float(v) / scale for v in r[:4])
                lm = [float(v) / scale for v in r[4:14]]
                if fw < 12 or fh < 12:
                    continue
                if gray is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found.append({"x": x, "y": y, "w": fw, "h": fh, "score": float(r[14]), "lm": lm,
                              "patch": _mouth_patch(gray, fh, lm)})
            times.append(t)
            faces_all.append(found)
    finally:
        cap.release()
    return {"times": times, "faces": faces_all}


def _mouth_patch(gray, fh: float, lm: list[float]):
    """24x16 lighting-normalised patch around the mouth corners (YuNet landmarks 3,4)."""
    H, W = gray.shape
    mrx, mry, mlx, mly = lm[6], lm[7], lm[8], lm[9]
    cx, cy = (mrx + mlx) / 2, (mry + mly) / 2
    half_w = max(6.0, abs(mlx - mrx) * 0.9)
    half_h = max(5.0, fh * 0.16)
    x1, x2, y1, y2 = int(cx - half_w), int(cx + half_w), int(cy - half_h), int(cy + half_h)
    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        return None
    p = cv2.resize(gray[y1:y2, x1:x2], (24, 16)).astype(np.float32) / 255.0
    return p - p.mean()


def mouth_activity(tracks: dict[int, list[tuple[float, dict]]]) -> dict:
    """Lip-patch change between consecutive samples of a track (talking mouths
    move). Patches were taken in the detection pass — no second decode."""
    acts: dict[tuple[int, float], float] = {}
    for tid, items in tracks.items():
        prev = None
        for t, f in items:
            p = f.get("patch")
            if p is not None and prev is not None:
                acts[(tid, round(t, 3))] = float(np.mean(np.abs(p - prev)))
            prev = p
    return acts


def track(sampled: dict, frame_w: int) -> dict[int, list[tuple[float, dict]]]:
    """Greedy identity: match each face to the nearest live track of similar size."""
    tracks: dict[int, list[tuple[float, dict]]] = {}
    live: dict[int, tuple[float, dict]] = {}
    nid = 0
    for t, faces in zip(sampled["times"], sampled["faces"]):
        used: set[int] = set()
        for f in sorted(faces, key=lambda f: -f["w"] * f["h"]):
            cx, cy = f["x"] + f["w"] / 2, f["y"] + f["h"] / 2
            best, best_d = None, None
            for tid, (lt, lf) in live.items():
                if tid in used or t - lt > MATCH_GAP_S:
                    continue
                lcx, lcy = lf["x"] + lf["w"] / 2, lf["y"] + lf["h"] / 2
                d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                ratio = max(f["w"], lf["w"]) / max(1.0, min(f["w"], lf["w"]))
                if d < max(lf["w"] * 1.2, frame_w * 0.04) and ratio < 1.8 and (best_d is None or d < best_d):
                    best, best_d = tid, d
            if best is None:
                best, nid = nid, nid + 1
            used.add(best)
            live[best] = (t, f)
            tracks.setdefault(best, []).append((t, f))
    return tracks


def to_box_samples(tracks: dict, acts: dict, src_w: int, src_h: int) -> list[BoxSample]:
    """Face → body-sized box (ranking's size/persistence terms expect people)."""
    out = []
    for tid, items in tracks.items():
        for t, f in items:
            x1 = max(0.0, f["x"] - f["w"] * 0.9)
            x2 = min(float(src_w), f["x"] + f["w"] * 1.9)
            y1 = max(0.0, f["y"] - f["h"] * 0.5)
            y2 = min(float(src_h), f["y"] + f["h"] * 4.0)
            out.append(BoxSample(t=t, track_id=tid, x1=x1, y1=y1, x2=x2, y2=y2, conf=f["score"],
                                 has_face=True, face_cx=f["x"] + f["w"] / 2, face_cy=f["y"] + f["h"] / 2,
                                 mouth_act=acts.get((tid, round(t, 3)))))
    out.sort(key=lambda b: (b.t, b.track_id))
    return out


def detect_shots(video: Path, duration: float) -> list[Shot]:
    try:
        from scenedetect import ContentDetector, detect
        scenes = detect(str(video), ContentDetector(threshold=27.0, min_scene_len=15))
        shots = [Shot(s.get_seconds(), e.get_seconds()) for s, e in scenes]
        if shots:
            shots[-1] = Shot(shots[-1].start, max(shots[-1].end, duration))
            return shots
    except Exception as e:  # noqa: BLE001
        log(f"shot detection failed ({e}) — single shot")
    return [Shot(0.0, duration)]


def saliency_centers(video: Path, shot: Shot, sample_fps: float = 4.0) -> list[tuple[float, float, float]]:
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    except Exception:  # noqa: BLE001
        return []
    cap = open_video(video)
    out = []
    t = shot.start
    try:
        while t < shot.end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok:
                ok2, m = sal.computeSaliency(frame)
                if ok2:
                    _, _, _, loc = cv2.minMaxLoc(cv2.GaussianBlur(m, (31, 31), 0))
                    out.append((t, float(loc[0]), float(loc[1])))
            t += 1.0 / sample_fps
    finally:
        cap.release()
    return out


def analyse(video: Path, info: dict) -> dict:
    """Everything the layout decision + reframe need, computed once per slice."""
    sampled = sample_faces(video, fps=info["fps"])
    trk = track(sampled, info["w"])
    acts = mouth_activity(trk)
    for faces in sampled["faces"]:  # patches are analysis-only; keep the result small
        for f in faces:
            f.pop("patch", None)
    boxes = to_box_samples(trk, acts, info["w"], info["h"])
    n = max(1, len(sampled["times"]))
    with_face = sum(1 for f in sampled["faces"] if f) / n
    stats = []
    for tid, items in trk.items():
        cx = np.array([f["x"] + f["w"] / 2 for _, f in items])
        cy = np.array([f["y"] + f["h"] / 2 for _, f in items])
        area = np.array([f["w"] * f["h"] for _, f in items]) / (info["w"] * info["h"])
        a = [acts.get((tid, round(t, 3))) for t, _ in items]
        a = [v for v in a if v is not None]
        stats.append({"id": tid, "presence": round(len(items) / n, 3),
                      "cx": float(np.median(cx)), "cy": float(np.median(cy)),
                      "cx_std": float(cx.std() / info["w"]), "cy_std": float(cy.std() / info["h"]),
                      "area": float(np.median(area)),
                      "w": float(np.median([f["w"] for _, f in items])),
                      "h": float(np.median([f["h"] for _, f in items])),
                      "mouth": round(float(np.mean(a)), 4) if a else 0.0})
    stats.sort(key=lambda s: -s["presence"])
    # Main face = biggest face that is on screen a meaningful share of the time
    # (audience members and posters are small; the speaker / webcam is not).
    persistent = [s for s in stats if s["presence"] >= 0.3] or stats
    main = max(persistent, key=lambda s: s["area"]) if persistent else None
    return {"sampled": sampled, "boxes": boxes, "with_face": round(with_face, 3), "tracks": stats,
            "main": main}


def replace_ids(boxes: list[BoxSample], mapping: dict[int, int]) -> list[BoxSample]:
    return [dataclasses.replace(b, track_id=mapping.get(b.track_id, b.track_id)) for b in boxes]
