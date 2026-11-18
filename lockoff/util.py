"""Shared helpers: subprocess, ffprobe, time formatting, JSON cache, encoder pick."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
JOBS = ROOT / "Jobs"

_T0 = time.time()

# The venv's yt-dlp ships with yt-dlp-ejs (YouTube JS challenge solver, uses Node);
# the bare winget build gets HTTP 403 on many YouTube formats without it.
_VENV_YTDLP = Path(sys.executable).parent / ("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")
YTDLP = str(_VENV_YTDLP) if _VENV_YTDLP.exists() else "yt-dlp"


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


class StageError(RuntimeError):
    """A pipeline stage failed in a way the user should see as-is."""


def run(args, cwd=None, timeout=None, check=True, stdin_text=None) -> subprocess.CompletedProcess:
    r = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, input=stdin_text)
    if check and r.returncode != 0:
        cmd = " ".join(str(a) for a in args)
        raise StageError(f"command failed ({r.returncode}): {cmd[:400]}\n{(r.stderr or '')[-2000:]}")
    return r


def secs(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    v = 0.0
    for p in str(t).split(":"):
        v = v * 60 + float(p)
    return v


def hms(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return s[:n].strip("-") or "clip"


def probe(path) -> dict:
    """{w, h, fps, duration, v_start, a_start, has_audio} via ffprobe."""
    out = run(["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,width,height,start_time,avg_frame_rate",
               "-of", "json", path]).stdout
    data = json.loads(out)
    info = {"duration": float(data["format"].get("duration") or 0), "has_audio": False,
            "w": 0, "h": 0, "fps": 30.0, "v_start": 0.0, "a_start": 0.0}
    for s in data.get("streams", []):
        if s["codec_type"] == "video" and not info["w"]:
            num, _, den = (s.get("avg_frame_rate") or "30/1").partition("/")
            den = float(den or 1) or 1.0
            info.update(w=int(s["width"]), h=int(s["height"]), fps=(float(num) / den) or 30.0,
                        v_start=float(s.get("start_time") or 0))
        elif s["codec_type"] == "audio" and not info["has_audio"]:
            info.update(has_audio=True, a_start=float(s.get("start_time") or 0))
    return info


# ------------------------------------------------------------------ power / CPU-GPU balance
# eco:      2 CPU threads, idle priority. PC stays fully usable; slowest.
# balanced: half the cores, below-normal priority, GPU does decode+encode+whisper. (default)
# max:      all cores, normal priority.
POWER_MODES = {"eco": (0.34, "idle"), "balanced": (0.5, "below"), "max": (1.0, "normal")}
POWER = "balanced"
THREADS = 3


def set_power(mode: str | None = None) -> None:
    """Cap CPU threads for every stage and lower process priority. Child processes
    (ffmpeg, yt-dlp) inherit an idle/below-normal priority class on Windows."""
    import os
    global POWER, THREADS
    POWER = mode or os.environ.get("CF_POWER") or "balanced"
    share, prio = POWER_MODES.get(POWER, POWER_MODES["balanced"])
    THREADS = max(2, int(round((os.cpu_count() or 4) * share)))
    os.environ["CF_POWER"] = POWER
    # OpenCV's built-in FFmpeg decoder otherwise spawns one thread per core.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"threads;{THREADS}"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(THREADS)
    try:
        import cv2
        cv2.setNumThreads(THREADS)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        cls = {"idle": 0x40, "below": 0x4000, "normal": 0x20}[prio]
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), cls)


# GPU-capable FFmpeg: FFmpeg 9 needs NVIDIA driver 610+ for NVENC; this PC has 560.
# The bundled Gyan 7.0.2 build (NVENC API 12.2) works with 560, so it does all
# encoding/decoding; the system ffmpeg is only the fallback.
_GPU_FF = ROOT / "tools" / "ffmpeg-gpu" / "7.0.2" / "bin" / "ffmpeg.exe"
_FF: str | None = None
_ENCODER: list[str] | None = None
_HWDEC: list[str] = []


def _encoder_ok(binary: str, codec: str) -> bool:
    try:
        r = subprocess.run([binary, "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.2",
                            "-c:v", codec, "-f", "null", "-"], capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# Tried in order; the first that actually encodes on THIS machine wins. A new GPU
# (or driver) is picked up on the next run with no config: NVIDIA → NVENC,
# Intel → Quick Sync, AMD → AMF, nothing → CPU x264.
_GPU_ENCODERS = [
    ("h264_nvenc", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "8M", "-maxrate", "12M",
                    "-bufsize", "16M"], ["-hwaccel", "cuda"]),
    ("h264_qsv", ["-preset", "medium", "-global_quality", "21", "-look_ahead", "0"], ["-hwaccel", "d3d11va"]),
    ("h264_amf", ["-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"], ["-hwaccel", "d3d11va"]),
]


def _pick_ffmpeg() -> None:
    global _FF, _ENCODER, _HWDEC
    if _FF is not None:
        return
    binaries = ["ffmpeg"] + ([str(_GPU_FF)] if _GPU_FF.exists() else [])
    for codec, opts, hw in _GPU_ENCODERS:
        for b in binaries:
            if _encoder_ok(b, codec):
                _FF, _ENCODER, _HWDEC = b, ["-c:v", codec, *opts], hw
                return
    _FF = "ffmpeg"
    _ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-threads", str(THREADS)]
    _HWDEC = []


def gpu_info() -> dict:
    """NVIDIA GPU name + VRAM (GB) via nvidia-smi; {} when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        name, mem = out[0].rsplit(",", 1)
        return {"name": name.strip(), "vram_gb": round(float(mem) / 1024, 1)}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {}


def auto_whisper_model() -> str:
    """Bigger GPU → more accurate transcription, no config needed."""
    v = gpu_info().get("vram_gb", 0)
    if v >= 10:
        return "large-v3"
    if v >= 6:
        return "large-v3-turbo"
    return "small"


def open_video(path):
    """cv2.VideoCapture with GPU (D3D11/NVDEC) decoding when OpenCV can, else CPU."""
    import cv2
    try:
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    except Exception:  # noqa: BLE001
        pass
    return cv2.VideoCapture(str(path))


def ffmpeg() -> str:
    _pick_ffmpeg()
    return _FF


def encoder() -> list[str]:
    """NVENC (GPU) when any available ffmpeg can use it, else CPU x264 capped to THREADS."""
    _pick_ffmpeg()
    return _ENCODER


def intermediate_encoder() -> list[str]:
    """Near-lossless encode for the cut slices that feed analysis + final render."""
    _pick_ffmpeg()
    codec = _ENCODER[1]
    if codec == "h264_nvenc":
        return ["-c:v", codec, "-preset", "p4", "-rc", "constqp", "-qp", "16"]
    if codec == "h264_qsv":
        return ["-c:v", codec, "-global_quality", "16"]
    if codec == "h264_amf":
        return ["-c:v", codec, "-rc", "cqp", "-qp_i", "16", "-qp_p", "16"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-threads", str(THREADS)]


def ff(*args) -> list:
    """ffmpeg command head: chosen binary, GPU decode when available, capped filter threads."""
    _pick_ffmpeg()
    return [_FF, "-v", "error", "-y", "-filter_threads", str(THREADS),
            "-filter_complex_threads", str(THREADS), *_HWDEC, *args]


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cached(path: Path, build, fresh: bool = False):
    """Stage cache: reuse the JSON on disk unless --fresh. Makes runs resumable."""
    if not fresh:
        hit = read_json(path)
        if hit is not None:
            return hit
    data = build()
    write_json(path, data)
    return data


def die(msg: str) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)
