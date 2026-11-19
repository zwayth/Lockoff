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

  lockoff live <url> [--workspace NAME] [--buffer 15] [--model small]
                                                                record a live stream into a rolling buffer
  lockoff live-status                                           show live sessions and their queued clips
  lockoff live-clip <session> [seconds]                          clip the last N seconds out of the buffer
  lockoff live-stop <session>                                   stop a recording
  lockoff workspace <name> [--channel URL] [--credit @h] [--triggers "clip it,GG"]
                                                                create/update a creator workspace
  lockoff workspaces                                            list workspaces, or `lockoff activity`
  lockoff queue [--dry-run] [--max 10] [--mode addToQueue|shareNow]
                                                                publish finished clips through Buffer
  lockoff refresh-urls [--dry-run]                              refresh media URLs on soon-to-publish posts
  lockoff post --setup                                          store the Buffer token for this product
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
from .social import BufferError
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

    # -- live buffer / workspaces / publishing -------------------------------
    p = sub.add_parser("live", help="record a stream into a rolling buffer")
    p.add_argument("url")
    p.add_argument("--workspace", default="default")
    p.add_argument("--buffer", type=int, default=15, help="minutes of stream kept in the buffer")
    p.add_argument("--name")
    p.add_argument("--model", default="small")
    p.add_argument("--lang")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--credit")
    p.add_argument("--no-auto-clip", dest="auto_clip", action="store_false", default=True)
    p = sub.add_parser("live-status", help="show live sessions")
    p = sub.add_parser("live-stop")
    p.add_argument("session")
    p = sub.add_parser("live-clip")
    p.add_argument("session")
    p.add_argument("seconds", type=float, nargs="?", default=30)
    p = sub.add_parser("workspace")
    p.add_argument("name")
    p.add_argument("--channel", default="")
    p.add_argument("--credit", default="")
    p.add_argument("--triggers", default="")
    p.add_argument("--buffer", dest="buffer_min", type=int, default=15)
    p.add_argument("--no-auto-clip", dest="auto_clip", action="store_false", default=True)
    sub.add_parser("workspaces")
    sub.add_parser("activity")
    p = sub.add_parser("queue", help="publish rendered clips through Buffer")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--mode", choices=["addToQueue", "shareNow"], default="addToQueue")
    p = sub.add_parser("refresh-urls")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("post", help="publish a finished job's clips")
    p.add_argument("job", nargs="?")
    p.add_argument("--clip")
    p.add_argument("--mode", choices=["shareNow", "addToQueue"], default="shareNow")
    p.add_argument("--credit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--setup", action="store_true")
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
        elif a.cmd == "live":
            from . import live as L
            s = L.start_session(a.url, a.workspace, buffer_min=a.buffer, name=a.name,
                                model=a.model, language=a.lang, device=a.device,
                                credit=a.credit, auto_clip=a.auto_clip)
            log(f"recording {s.url} → {s.dir}")
            log("press Ctrl+C to stop (the rolling buffer is kept on disk)")
            try:
                while s.name in L.SESSIONS and not s.stop_flag.wait(2):
                    pass
            except KeyboardInterrupt:
                s.stop()
        elif a.cmd == "live-status":
            from . import live as L
            sessions = L.status_all()
            if not sessions:
                print("no live sessions yet — run `lockoff live <url>`")
            for s in sessions:
                print(f"{s['name']}  {s['status']:9} {s.get('elapsed', 0):>6}s  "
                      f"buffer {s.get('buffer_sec', 0):>5}s  words {s.get('words', 0):>4}  "
                      f"clips {len(s.get('clips', []))}  [{s.get('workspace')}]")
                for c in s.get("clips", []):
                    print(f"    {c['id']}  {c['status']:5}  {c.get('why', '')}  {c.get('title', '')[:60]}")
        elif a.cmd == "live-stop":
            from . import live as L
            print("stopped" if L.stop_session(a.session) else f"session {a.session} not running")
        elif a.cmd == "live-clip":
            from . import live as L
            req = L.clip_last(a.session, a.seconds)
            log(f"clip {req['id']} requested from the last {a.seconds:g}s")
        elif a.cmd == "workspace":
            from . import workspaces as W
            w = W.upsert(a.name, a.channel, a.credit, a.triggers, a.buffer_min, a.auto_clip)
            print(f"workspace saved: {w['name']} ({w['channel'] or 'no channel'})")
        elif a.cmd == "workspaces":
            from . import workspaces as W
            for name, w in sorted(W.all_creators().items()):
                print(f"{name:20} {w.get('channel', ''):45} credit {w.get('credit', '')}")
        elif a.cmd == "activity":
            from . import workspaces as W
            for e in W.activity(30):
                print(f"{time.strftime('%m-%d %H:%M', time.localtime(e['t']))}  "
                      f"{e['creator']:18} {e['message']}")
        elif a.cmd == "queue":
            from . import queue as Q
            Q.queue(max_scheduled=a.max, mode=a.mode, dry_run=a.dry_run)
        elif a.cmd == "refresh-urls":
            from . import queue as Q
            n = Q.refresh_due_urls(dry_run=a.dry_run)
            log(f"refreshed {n} scheduled post URL(s)")
        elif a.cmd == "post":
            from . import social as S
            if a.setup:
                cfg = S.load_config()
                cfg["buffer_token"] = input("Buffer API key: ").strip()
                org = cfg.get("organization_id")
                if not org:
                    try:
                        org = S.Buffer(cfg["buffer_token"]).organizations()[0]["id"]
                    except Exception:  # noqa: BLE001
                        org = ""
                cfg["organization_id"] = input(f"Buffer organization id [{org}]: ").strip() or org
                S.save_config(cfg)
                print(f"saved to {S.CONFIG}")
            elif not a.job:
                ap.error("lockoff post needs a job directory (or --setup)")
            else:
                job = Path(a.job) if Path(a.job).exists() else JOBS / a.job
                no = [int(x) for x in a.clip.split(",")] if a.clip else [1]
                for i in no:
                    print(S.post_clip(job, i, mode=a.mode, credit=a.credit, dry_run=a.dry_run))
    except (StageError, BufferError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped — run the same command again to resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
