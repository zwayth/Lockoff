"""Word-level transcription with faster-whisper (GPU when CUDA libs load, else CPU int8).

Output: {"language", "duration", "segments": [{start,end,text}], "words": [{w,s,e}]}
All times are seconds on the SOURCE timeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .util import log


def _add_cuda_dll_dirs() -> None:
    """pip's nvidia-cublas/cudnn wheels put DLLs in site-packages/nvidia/*/bin."""
    if sys.platform != "win32":
        return
    for base in sys.path:
        nv = Path(base) / "nvidia"
        if nv.is_dir():
            for b in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(b))
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass


def _load_model(model_name: str, device: str):
    from faster_whisper import WhisperModel
    _add_cuda_dll_dirs()
    tries = []
    if device in ("auto", "cuda"):
        tries += [("cuda", "int8_float16"), ("cuda", "int8")]
    if device in ("auto", "cpu"):
        tries += [("cpu", "int8")]
    last = None
    for dev, ct in tries:
        try:
            from .util import THREADS
            m = WhisperModel(model_name, device=dev, compute_type=ct, cpu_threads=THREADS)
            log(f"whisper {model_name} on {dev}/{ct}")
            return m, dev
        except Exception as e:  # noqa: BLE001 — try the next backend
            last = e
            log(f"whisper {dev}/{ct} unavailable: {str(e)[:120]}")
    raise RuntimeError(f"no whisper backend could load: {last}")


def transcribe(wav: Path, model_name: str = "small", language: str | None = None,
               device: str = "auto") -> dict:
    from .signals import read_wav
    model, dev = _load_model(model_name, device)
    audio = read_wav(wav)  # 16 kHz mono float32 — skips PyAV decoding entirely
    try:
        segs_iter, info = model.transcribe(
            audio, language=language, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400}, beam_size=5,
            condition_on_previous_text=False)
        segments, words = [], []
        last_pct = -10
        for s in segs_iter:
            segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
            for w in s.words or []:
                t = w.word.strip()
                if t:
                    words.append({"w": t, "s": round(w.start, 3), "e": round(w.end, 3)})
            pct = int(100 * s.end / max(1.0, info.duration))
            if pct >= last_pct + 10:
                log(f"  transcribed {pct}%")
                last_pct = pct
    except RuntimeError as e:
        if dev == "cuda" and device == "auto":  # CUDA loaded but failed mid-run (VRAM / cuDNN)
            log(f"GPU transcription failed ({str(e)[:100]}) — retrying on CPU")
            return transcribe(wav, model_name, language, "cpu")
        raise
    return {"language": info.language, "duration": round(info.duration, 2),
            "segments": segments, "words": words, "model": model_name, "device": dev}


def words_in(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.3]


def text_in(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["w"] for w in words_in(words, start, end))
