"""Lockoff CLI.

  lockoff run <file-or-url> [--clips 5] [--length auto|short|medium|long] [--profile auto|podcast|stream]
                       [--llm claude|claude:sonnet|ollama:qwen2.5:7b|none] [--layout auto|speaker|facecam|general]
                       [--whisper small|medium|large-v3] [--lang en] [--credit @handle]
                       [--captions hormozi|clean|off] [--no-hook] [--job NAME] [--fresh]
  lockoff rerender <jobDir> [--clip 2,3] [--layout speaker]   rebuild after editing plan.json
  lockoff watch [--inbox DIR] [run options]                     process every video dropped in Inbox/
  lockoff batch <list.txt> [run options]                        one source (file or URL) per line
  lockoff clean <jobDir>                                        delete _work, keep final clips + clips.md
  lockoff doctor                                                check the toolchain
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import factory as F
from .ingest import VIDEO_EXT
from . import util as U
from .util import JOBS, ROOT, StageError, encoder, log, set_power


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--clips", type=int)
    p.add_argument("--length", choices=["auto", "short", "medium", "long"])
    p.add_argument("--profile", choices=["auto", "podcast", "stream"])
    p.add_argument("--llm")
    p.add_argument("--layout", choices=["auto", "speaker", "facecam", "general", "fit"])
    p.add_argument("--whisper")
    p.add_argument("--lang")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--credit")
    p.add_argument("--captions", choices=["hormozi", "clean", "off"])
    p.add_argument("--no-hook", dest="hook", action="store_false", default=None)
    p.add_argument("--job")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--power", choices=["eco", "balanced", "max"],
                   help="CPU use: eco (light, slow), balanced (default), max (all cores)")
    p.add_argument("--director", choices=["on", "off"],
                   help="AI director: Claude picks who to frame, only in shots with 2+ people (uses credits)")
    p.add_argument("--director-model", dest="director_model", help="Claude model for the director (default sonnet)")
    p.add_argument("--director-max", dest="director_max", type=int, help="max Claude calls per clip (default 12)")


def _kw(a) -> dict:
    keys = ("clips", "length", "profile", "llm", "layout", "whisper", "lang", "device", "credit",
            "captions", "hook", "job", "fresh", "director", "director_model", "director_max")
    return {k: getattr(a, k, None) for k in keys}


def doctor() -> int:
    ok = True
    for tool in ("ffmpeg", "ffprobe", "yt-dlp", "claude"):
        path = shutil.which(tool)
        print(f"{'OK ' if path else '-- '} {tool:8} {path or 'not found'}")
        ok &= bool(path) or tool == "claude"
    for mod in ("cv2", "numpy", "onnxruntime", "faster_whisper", "scenedetect"):
        try:
            __import__(mod)
            print(f"OK  {mod}")
        except ImportError as e:
            print(f"--  {mod}: {e}")
            ok = False
    for m in ("face_detection_yunet_2023mar.onnx", "yamnet.onnx", "yamnet_class_map.csv"):
        print(f"{'OK ' if (ROOT / 'models' / m).exists() else '-- '} models/{m}")
    try:
        import ctranslate2
        print(f"OK  CUDA devices for whisper: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:  # noqa: BLE001
        print(f"--  ctranslate2: {e}")
    g = U.gpu_info()
    print(f"OK  GPU: {g.get('name', 'no NVIDIA GPU')} {g.get('vram_gb', '')} GB → whisper '{U.auto_whisper_model()}'")
    enc = encoder()[1]
    print(f"OK  video encoder: {enc} ({'GPU' if 'nvenc' in enc else 'CPU'}) via {U.ffmpeg()}")
    print(f"OK  power mode: {U.POWER} — {U.THREADS} CPU threads, lowered priority")
    return 0 if ok else 1


def watch(inbox: Path, kw: dict, interval: int = 20) -> None:
    done, failed = inbox / "done", inbox / "failed"
    for d in (inbox, done, failed):
        d.mkdir(parents=True, exist_ok=True)
    log(f"watching {inbox} — drop videos (or .url/.txt files with links) in it. Ctrl+C to stop.")
    sizes: dict[Path, int] = {}
    while True:
        for f in sorted(inbox.iterdir()):
            if not f.is_file():
                continue
            size = f.stat().st_size
            if sizes.get(f) != size:  # still being copied — wait for a stable size
                sizes[f] = size
                continue
            sources = [f.read_text(encoding="utf-8").strip()] if f.suffix.lower() in (".url", ".txt") else \
                [str(f)] if f.suffix.lower() in VIDEO_EXT else []
            if not sources:
                continue
            for s in sources:
                s = s.replace("URL=", "").splitlines()[-1].strip() if f.suffix.lower() == ".url" else s
                try:
                    F.run_job(s, **kw)
                    dest = done
                except Exception as e:  # noqa: BLE001 — keep the factory running
                    log(f"FAILED {f.name}: {e}")
                    dest = failed
            shutil.move(str(f), str(dest / f.name))
            sizes.pop(f, None)
        time.sleep(interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lockoff", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("source")
    _run_args(p)
    p = sub.add_parser("rerender")
    p.add_argument("job")
    p.add_argument("--clip")
    p.add_argument("--layout", choices=["auto", "speaker", "facecam", "general", "fit"])
    p.add_argument("--power", choices=["eco", "balanced", "max"])
    p.add_argument("--director", choices=["on", "off"])
    p = sub.add_parser("watch")
    p.add_argument("--inbox", default=str(ROOT / "Inbox"))
    _run_args(p)
    p = sub.add_parser("batch")
    p.add_argument("list")
    _run_args(p)
    p = sub.add_parser("clean")
    p.add_argument("job")
    sub.add_parser("doctor")
    a = ap.parse_args(argv)
    set_power(getattr(a, "power", None))
    try:
        if a.cmd == "run":
            F.run_job(a.source, **_kw(a))
        elif a.cmd == "rerender":
            only = [int(x) for x in a.clip.split(",")] if a.clip else None
            job = Path(a.job) if Path(a.job).exists() else JOBS / a.job
            F.rerender(job, only, a.layout, a.director)
        elif a.cmd == "watch":
            watch(Path(a.inbox), _kw(a))
        elif a.cmd == "batch":
            for line in Path(a.list).read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    try:
                        F.run_job(line.strip(), **_kw(a))
                    except Exception as e:  # noqa: BLE001
                        log(f"FAILED {line.strip()}: {e}")
        elif a.cmd == "clean":
            F.clean(Path(a.job) if Path(a.job).exists() else JOBS / a.job)
        elif a.cmd == "doctor":
            return doctor()
    except (StageError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped — run the same command again to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
