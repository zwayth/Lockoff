"""Sources: a local video file or a YouTube/Twitch/any yt-dlp URL.

URLs never download the whole video: metadata + audio only for analysis, then
just the chosen windows. yt-dlp section downloads start on a keyframe, not on
the exact second, so every fetched window is re-aligned to the master audio by
cross-correlating loudness envelopes — captions and cuts then land exactly.
Every slice is re-timed so audio and video both start at 0.000 (Rule A).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .signals import read_wav
from .util import YTDLP, StageError, ff, ffmpeg, intermediate_encoder, log, probe, run

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi", ".flv", ".ts"}


def is_url(s: str) -> bool:
    return bool(re.match(r"https?://", s or ""))


def resolve(source: str, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "source.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source") == source and Path(meta["wav"]).exists():
            return meta
    wav = work / "audio16k.wav"
    if is_url(source):
        log("fetching metadata…")
        m = json.loads(run([YTDLP, "-J", "--no-warnings", "--no-playlist", source], timeout=180).stdout)
        meta = {k: m.get(k) for k in ("title", "uploader", "channel", "webpage_url", "duration",
                                      "extractor", "upload_date", "width", "height")}
        meta["heatmap"] = m.get("heatmap") or []
        log(f"{meta['title']} — {meta['uploader']} — {int(meta['duration'] or 0)}s; downloading audio only…")
        run([YTDLP, "-f", "ba[ext=m4a]/ba/b", "-x", "--audio-format", "m4a", "--no-playlist",
             "-o", str(work / "audio.%(ext)s"), "--no-warnings", source], timeout=3600)
        audio = next(work.glob("audio.*"))
        run([ffmpeg(), "-v", "error", "-y", "-i", audio, "-ac", "1", "-ar", "16000", wav], timeout=3600)
        audio.unlink(missing_ok=True)
        meta.update(kind="url", source=source, wav=str(wav))
    else:
        src = Path(source).resolve()
        if not src.exists():
            raise StageError(f"source not found: {src}")
        info = probe(src)
        if not info["has_audio"]:
            raise StageError("source has no audio track")
        log(f"extracting audio from {src.name} ({int(info['duration'])}s)…")
        run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav], timeout=3600)
        meta = {"kind": "file", "source": source, "path": str(src), "title": src.stem, "uploader": None,
                "duration": info["duration"], "width": info["w"], "height": info["h"], "heatmap": [],
                "wav": str(wav)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def _normalise(src: Path, out: Path, ss: float, dur: float) -> None:
    """Frame-accurate cut + Rule A re-timing to an intermediate (CRF 16)."""
    run([*ff("-ss", f"{max(0.0, ss):.3f}", "-i", src), "-t", f"{dur:.3f}",
         "-filter_complex",
         "[0:v]setpts=PTS-STARTPTS,fps=30[v];[0:a]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a]",
         "-map", "[v]", "-map", "[a]", *intermediate_encoder(), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out], timeout=1800)


def _envelope(x: np.ndarray, hop: int = 160) -> np.ndarray:
    n = len(x) // hop
    e = np.abs(x[: n * hop]).reshape(n, hop).mean(axis=1)
    e = np.log1p(e * 50)
    return (e - e.mean()) / (e.std() + 1e-6)


def align_offset(piece_wav: np.ndarray, master: np.ndarray, guess: float, search: float = 15.0) -> float:
    """Master-timeline second at which `piece` starts (envelope cross-correlation)."""
    sr = 16000
    a = max(0, int((guess - search) * sr))
    b = min(len(master), int((guess + search + 25) * sr))
    m = _envelope(master[a:b])
    p = _envelope(piece_wav[: 20 * sr])
    if len(p) < 200 or len(m) <= len(p):
        return guess
    corr = np.correlate(m, p, mode="valid") / len(p)
    lag = int(np.argmax(corr))
    if corr[lag] < 0.3:
        log(f"  alignment weak (r={corr[lag]:.2f}) — trusting section start")
        return guess
    return a / sr + lag / 100.0


def fetch_slice(meta: dict, work: Path, name: str, start: float, end: float) -> Path:
    out = work / f"{name}_slice.mp4"
    if out.exists():
        return out
    dur = end - start
    if meta["kind"] == "file":
        _normalise(Path(meta["path"]), out, start, dur)
    else:
        pad = 6.0
        a, b = max(0.0, start - pad), end + pad
        raw = work / f"{name}_raw.%(ext)s"
        run([YTDLP, "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--download-sections", f"*{a}-{b}",
             "--force-keyframes-at-cuts", "--merge-output-format", "mkv", "--no-playlist",
             "-o", str(raw), "--no-warnings", meta["source"]], timeout=1800)
        src = next(work.glob(f"{name}_raw.*"))
        pw = work / f"{name}_raw16k.wav"
        run([ffmpeg(), "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", pw])
        at = align_offset(read_wav(pw), read_wav(Path(meta["wav"])), a)
        log(f"  {name}: section starts at {at:.2f}s on the master timeline (asked {a:.2f}s)")
        _normalise(src, out, start - at, dur)
        pw.unlink(missing_ok=True)
        src.unlink(missing_ok=True)
    info = probe(out)
    if abs(info["v_start"] - info["a_start"]) > 0.001:
        raise StageError(f"{out.name}: audio/video start mismatch")
    return out
