"""Multi-signal "interestingness" curve (podcli saliency design + streamclip scan).

Channels, all on a 1-second grid and normalised PER VIDEO (robust z → 0..1):
  energy    loudness spikes vs the video's own baseline
  reaction  YAMNet laughter / cheering / screaming probability (ONNX, no torch)
  words     reaction words in the transcript ("no way", "let's go", [laughter]…)
  rate      speech-rate bursts (excited talking speeds up)
  heat      YouTube "most replayed" heatmap, when the source has one
A profile is a weight vector over these. Peaks feed the LLM as anchors and are
also the candidate source when no LLM is available.
"""
from __future__ import annotations

import csv
import re
import wave
from pathlib import Path

import numpy as np

from .util import MODELS, log

REACTION_WORDS = re.compile(
    r"\b(oh my god|omg|no way|what the|wtf|holy|let'?s go+|clutch|insane|crazy|unbelievable|"
    r"wait what|are you serious|bro+|dude|ha(ha)+|lmao|lol|scream|damn|wow+|shit|f+u+c+k+)\b|"
    r"\[(laughter|laughs|applause|screams?|cheering)\]", re.I)

PROFILES = {
    # talking-head / podcast / interview: words matter most, LLM leads
    "podcast": {"energy": 0.20, "reaction": 0.35, "words": 0.10, "rate": 0.15, "heat": 0.20},
    # gaming / IRL streams: loud moments and reactions lead
    "stream": {"energy": 0.35, "reaction": 0.25, "words": 0.20, "rate": 0.05, "heat": 0.15},
}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def energy_db(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    n = len(wav) // sr
    if n == 0:
        return np.zeros(0)
    x = wav[: n * sr].reshape(n, sr)
    rms = np.sqrt(np.mean(x * x, axis=1) + 1e-10)
    return 20 * np.log10(rms + 1e-10)


def reactions(wav: np.ndarray, n_sec: int) -> np.ndarray | None:
    """Per-second max(laughter, cheering, screaming) from YAMNet, or None."""
    model, cmap = MODELS / "yamnet.onnx", MODELS / "yamnet_class_map.csv"
    if not model.exists() or not cmap.exists():
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    names = {}
    with open(cmap, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            names[int(row[0])] = row[2]
    keys = [i for i, n in names.items()
            if any(k in n.lower() for k in ("laugh", "giggle", "chuckle", "snicker"))
            or n in ("Cheering", "Applause", "Whoop", "Screaming", "Crowd")]
    from .util import THREADS
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = np.zeros(n_sec, dtype=np.float32)
    win = 16000 * 300
    for off in range(0, len(wav), win):
        chunk = wav[off: off + win]
        if len(chunk) < 16000:
            continue
        scores = sess.run(None, {inp: chunk})[0]
        if scores.shape[0] == 0:
            continue
        hop = (len(chunk) / 16000) / scores.shape[0]
        lvl = scores[:, keys].max(axis=1)
        for i, v in enumerate(lvl):
            s = int(off / 16000 + i * hop)
            if s < n_sec:
                out[s] = max(out[s], float(v))
    return out


def robust01(x: np.ndarray) -> np.ndarray:
    """Per-video robust z (median/MAD), clipped to [0, 4] and scaled to 0..1."""
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-6:
        mad = np.std(x) or 1.0
    return np.clip((x - med) / mad, 0, 4) / 4.0


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if x.size < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def build(wav_path: Path, transcript: dict, heatmap: list | None, profile: str) -> dict:
    wav = read_wav(wav_path)
    db = energy_db(wav)
    n = len(db)
    ch: dict[str, np.ndarray] = {}
    base = float(np.median(db)) if n else -60.0
    ch["energy"] = robust01(smooth(np.maximum(db, base - 20), 2))

    words = transcript.get("words", [])
    rate = np.zeros(n)
    hits = np.zeros(n)
    for w in words:
        i = int(w["s"])
        if 0 <= i < n:
            rate[i] += 1
    for seg in transcript.get("segments", []):
        k = len(REACTION_WORDS.findall(seg["text"]))
        i = int(seg["start"])
        if k and 0 <= i < n:
            hits[i] += k
    ch["rate"] = robust01(smooth(rate, 5))
    ch["words"] = np.clip(smooth(hits, 5) * 2.5, 0, 1)

    log("signals: running YAMNet reaction detector…")
    react = reactions(wav, n)
    if react is not None:
        ch["reaction"] = np.clip(smooth(react, 3) * 2.0, 0, 1)
    if heatmap:
        heat = np.zeros(n)
        for h in heatmap:
            a, b = int(h.get("start_time", 0)), int(h.get("end_time", 0))
            heat[max(0, a): min(n, max(a + 1, b))] = float(h.get("value", 0))
        ch["heat"] = robust01(heat)

    weights = {k: v for k, v in PROFILES[profile].items() if k in ch}
    total = sum(weights.values()) or 1.0
    fused = sum(ch[k] * (v / total) for k, v in weights.items())
    fused = smooth(fused, 5)
    return {"n": n, "baseline_db": round(base, 1), "channels": {k: v.round(3).tolist() for k, v in ch.items()},
            "fused": np.round(fused, 4).tolist(), "weights": weights, "profile": profile}


def peaks(fused: list[float], k: int = 20, min_gap: int = 30) -> list[dict]:
    """Greedy NMS peak-pick on the fused curve, adaptive floor (top 15th pct)."""
    x = np.asarray(fused)
    if x.size == 0:
        return []
    floor = float(np.percentile(x, 85))
    order = np.argsort(-x)
    picked: list[int] = []
    for i in order:
        if x[i] < floor and len(picked) >= 3:
            break
        if all(abs(int(i) - p) >= min_gap for p in picked):
            picked.append(int(i))
        if len(picked) >= k:
            break
    return [{"t": p, "score": round(float(x[p]), 3)} for p in sorted(picked)]


def window_score(sig: dict, start: float, end: float) -> float:
    """Signal strength of a clip window: 0.6·peak + 0.4·mean of the fused curve."""
    x = sig.get("fused") or []
    a, b = max(0, int(start)), min(len(x), int(end) + 1)
    if b <= a:
        return 0.0
    seg = x[a:b]
    return round(0.6 * max(seg) + 0.4 * (sum(seg) / len(seg)), 4)


def describe_peak(sig: dict, t: int) -> str:
    parts = []
    for name, vals in sig["channels"].items():
        if t < len(vals) and vals[t] >= 0.45:
            parts.append(f"{name} {vals[t]:.2f}")
    return ", ".join(parts) or "mixed"
