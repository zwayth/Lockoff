"""Rolling live buffer: never miss a moment.

This is the PeakCut feature that matters most locally. Lockoff records a live
stream continuously into short segments, keeps only the last N minutes, and
transcribes it as it goes. When the crowd gets loud, someone says "clip that",
or you press *Clip last 30s*, the moment is rendered straight out of the
buffer through Lockoff's normal reframe/caption/QA pipeline — even though the
stream never finished downloading.

Unlike PeakCut this needs no server, sign-in, public HTTPS address or Meta app.
Everything stays in Jobs/_live/<session>/ on this PC.

    lockoff live <url> --workspace mycreator --buffer 15 --model small
    lockoff live-status
    lockoff live-clip <session> 30
    lockoff live-stop <session>
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import factory as F
from . import workspaces as W
from .util import JOBS, ROOT, YTDLP, StageError, ffmpeg, log, read_json, run

LIVE_DIR = JOBS / "_live"
SEG_SEC = 6           # length of each rolling buffer piece
PAD = 8               # seconds kept either side of a clip so it can be re-trimmed
SR = 16000
CLIP_GAP = 45         # least seconds between auto-triggered clips
SESSIONS: dict[str, "Session"] = {}
_LOCK = threading.RLock()


# ------------------------------------------------------------------ helpers

def normalize(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        url = "https://" + url
    m = re.match(r"^(https?://(?:www\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+))/?(?:[?#].*)?$", url)
    return m.group(1) + "/live" if m else url


def _resolve(url: str) -> dict:
    """Ask yt-dlp for stream details. Raises StageError when offline."""
    import json as _json
    try:
        meta = _json.loads(run([YTDLP, "-J", "--no-warnings", "--no-playlist", url], timeout=180).stdout)
    except Exception as e:  # noqa: BLE001
        raise StageError(f"could not read the stream: {str(e)[-200:]}") from e
    if not meta.get("is_live"):
        raise StageError("that channel is not live right now")
    urls = [ln.strip() for ln in run(
        [YTDLP, "-g", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--no-warnings",
         "--no-playlist", url], timeout=180).stdout.splitlines() if ln.strip().startswith("http")]
    if not urls:
        raise StageError("yt-dlp returned no playable stream URL")
    return {"title": meta.get("title") or "Live stream",
            "uploader": meta.get("uploader") or meta.get("channel") or "",
            "urls": urls}


def _quiet_cut(x: np.ndarray, lo: float = 4.0, hi: float = 6.5) -> int:
    """Sample index near the quietest point in [lo, hi] so no word is split."""
    n = len(x)
    a, b = int(lo * SR), min(n, int(hi * SR))
    if b - a < SR:
        return n
    w = SR // 2
    e = np.array([np.abs(x[i:i + w]).mean() for i in range(a, b - w, w)])
    return int(a + int(np.argmin(e)) * w) if len(e) else n


class _LiveTranscriber:
    """One whisper model, reused for every window (loading each time is slow)."""

    def __init__(self, model_name: str, language: str | None, device: str):
        self.model_name, self.language, self.device = model_name, language, device
        self._model = None
        self._dev = None

    def _load(self):
        if self._model is not None:
            return
        from .transcribe import _add_cuda_dll_dirs
        from faster_whisper import WhisperModel
        _add_cuda_dll_dirs()
        tries = ([("cuda", "int8_float16"), ("cuda", "int8")]
                 if self.device in ("auto", "cuda") else []) + [("cpu", "int8")]
        last = None
        for dev, ct in tries:
            try:
                self._model = WhisperModel(self.model_name, device=dev, compute_type=ct,
                                           cpu_threads=4)
                self._dev = dev
                log(f"live whisper {self.model_name} on {dev}/{ct}")
                return
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"no whisper backend for live captions: {last}")

    def transcribe(self, audio: np.ndarray) -> list[dict]:
        self._load()
        segs, _ = self._model.transcribe(
            audio, language=self.language, word_timestamps=True, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400}, beam_size=5,
            condition_on_previous_text=False)
        out = []
        for s in segs:
            for w in s.words or []:
                t = w.word.strip()
                if t:
                    out.append({"w": t, "s": round(w.start, 3), "e": round(w.end, 3)})
        return out


# ------------------------------------------------------------------ session

class Session:
    def __init__(self, url: str, workspace: str = "default", buffer_min: int = 15,
                 name: str | None = None, model: str = "small", language: str | None = None,
                 device: str = "auto", credit: str | None = None, auto_clip: bool = True):
        self.url = normalize(url)
        self.workspace = workspace or "default"
        self.buffer_min = max(3, int(buffer_min))
        self.name = name or time.strftime("live-%Y%m%d-%H%M%S")
        self.dir = LIVE_DIR / self.name
        self.buf = self.dir / "buffer"
        self.clipdir = self.dir / "clips"
        self.state_path = self.dir / "session.json"
        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.ended = False
        self.proc: subprocess.Popen | None = None
        self.pending = bytearray()
        self.pending_base = 0
        self.samples = 0
        self.db: list[float] = []
        self.words: list[dict] = []
        self.segments: list[tuple[Path, float, float]] = []
        self.seg_clock = 0.0
        self.csv_seen = 0
        self.part = 0
        self.n = 0
        self.last_trigger = -999.0
        self.last_spike = -999.0
        self.spikes = 0
        self.triggers = W.get(self.workspace).get("triggers", "")
        self.credit = credit or W.get(self.workspace).get("credit") or None
        self.auto_clip = auto_clip
        self.title = self.workspace + " live"
        self.uploader = ""
        self.error = ""
        self.started = time.time()
        self.clips: list[dict] = []
        for d in (self.buf, self.clipdir, self.dir / "_work"):
            d.mkdir(parents=True, exist_ok=True)
        self.tx = _LiveTranscriber(model, language, device)

    # -------------------------------------------------------------- recording
    def start(self) -> None:
        SESSIONS[self.name] = self
        W.log(self.workspace, f"recording started: {self.url}")
        threading.Thread(target=self._ingest, daemon=True, name="live-ingest").start()
        threading.Thread(target=self._analyze, daemon=True, name="live-analyze").start()

    def stop(self, note: str = "Stopped by you") -> None:
        self.stop_flag.set()
        self.ended = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
        self._save("done", note)
        W.log(self.workspace, f"recording stopped: {note}")

    def _ingest(self) -> None:
        try:
            self._ingest_loop()
        except Exception as e:  # noqa: BLE001
            self.error = f"Recording stopped: {str(e)[-200:]}"
            self._save("error", self.error)
        finally:
            self.ended = True

    def _ingest_loop(self) -> None:
        info = _resolve(self.url)
        self.title, self.uploader = info["title"], info["uploader"]
        cmd = [ffmpeg(), "-v", "error", "-reconnect", "1", "-reconnect_streamed", "1",
               "-reconnect_delay_max", "10"]
        for u in info["urls"]:
            cmd += ["-i", u]
        a_in = "1:a:0" if len(info["urls"]) > 1 else "0:a:0"
        cmd += ["-map", "0:v:0", "-map", a_in, "-c", "copy",
                "-f", "segment", "-segment_time", str(SEG_SEC), "-reset_timestamps", "1",
                "-segment_format", "mpegts", "-segment_list", "buffer.csv",
                "-segment_list_type", "csv", "buffer_%05d.ts",
                "-map", a_in, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
        with open(self.dir / "ffmpeg.log", "ab") as errlog:
            self.proc = subprocess.Popen(cmd, cwd=self.buf, stdout=subprocess.PIPE,
                                         stderr=errlog,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while not self.stop_flag.is_set():
                data = self.proc.stdout.read(SR)     # one second of 16-bit mono
                if not data:
                    break
                data = data[: len(data) // 2 * 2]
                x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                level = float(20 * np.log10(np.sqrt((x ** 2).mean() + 1e-10))) if len(x) else -100.0
                with self.lock:
                    self.pending += data
                    self.samples += len(data) // 2
                    self.db.append(level)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
        if not self.stop_flag.is_set():
            self._save("live", "stream ended — finishing queued clips")

    def _scan_segments(self) -> None:
        csv = self.buf / "buffer.csv"
        if not csv.exists():
            return
        with self.lock:
            rows = [r.split(",") for r in csv.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if r.count(",") >= 2]
            for name, _a, b in (r[:3] for r in rows[self.csv_seen:]):
                try:
                    dur = max(0.0, float(b) - float(a))
                except ValueError:
                    continue
                self.segments.append((self.buf / name.strip(), self.seg_clock, self.seg_clock + dur))
                self.seg_clock += dur
            self.csv_seen = len(rows)
            keep_from = self.seg_clock - self.buffer_min * 60
            while self.segments and self.segments[0][2] < keep_from:
                self.segments.pop(0)[0].unlink(missing_ok=True)

    # --------------------------------------------------------------- analysis
    def _transcribe_step(self) -> bool:
        with self.lock:
            n = len(self.pending) // 2
            hurry = self.ended or bool([c for c in self.clips if c["status"] == "queued"])
            if n < (SR if hurry else SR * 7):
                return False
            audio = np.frombuffer(bytes(self.pending[: n * 2]), dtype=np.int16).astype(np.float32) / 32768.0
            base = self.pending_base
        cut = n if hurry or n > SR * 26 else _quiet_cut(audio)
        try:
            words = self.tx.transcribe(audio[:cut])
        except Exception as e:  # noqa: BLE001
            log(f"live transcription unavailable: {str(e)[-120:]}")
            self.tx = None  # type: ignore[assignment]
            return False
        off = base / SR
        for w in words:
            w["s"] = round(w["s"] + off, 2)
            w["e"] = round(w["e"] + off, 2)
        with self.lock:
            del self.pending[: cut * 2]
            self.pending_base += cut
            self.words += words
        self._find_triggers(words)
        self._save("live", f"{len(self.words)} words heard")
        return True

    def _find_triggers(self, words: list[dict]) -> None:
        clip_word = re.compile(r"\bclip(\s+(it|that|this))?\b", re.I)
        phrases = [p.strip().lower() for p in (self.triggers or "").split(",") if p.strip()]
        for w in words:
            clean = re.sub(r"[^a-z' ]", "", w["w"].lower())
            why = None
            if clip_word.search(clean):
                why = 'someone said "clip"'
            elif phrases:
                key = re.sub(r"[^a-z0-9']", "", clean)
                for p in phrases:
                    parts = [re.sub(r"[^a-z0-9']", "", x) for x in p.split()]
                    if key and key in (" ".join(parts), parts[-1] if parts else ""):
                        why = f'trigger "{p}"'
                        break
            if why and w["s"] - self.last_trigger >= CLIP_GAP:
                self.last_trigger = w["s"]
                self._queue_clip(max(0.0, w["s"] - 30), w["s"] + 5, why)

    def _watch_spikes(self) -> None:
        if not self.auto_clip:
            return
        with self.lock:
            n = len(self.db)
            if n < 60:
                return
            db = np.array(self.db[max(0, n - 190):], dtype=np.float32)
        amp = 10 ** (db / 20)
        base, cur = float(amp[:-10].mean()), float(amp[-5:].mean())
        if cur / max(base, 1e-6) - 1 < 0.6 or n - self.last_spike < CLIP_GAP:
            return
        self.last_spike = n
        self.spikes += 1
        self._queue_clip(max(0.0, n - 33), float(n + 2), "loudness jump")

    def _queue_clip(self, start: float, end: float, why: str) -> None:
        with self.lock:
            self.n += 1
            req = {"id": f"c{self.n:03d}", "start": round(start, 2), "end": round(end, 2),
                   "why": why, "status": "queued", "made": time.time()}
            self.clips.append(req)
        W.log(self.workspace, f"clip {req['id']} queued ({why})")
        self._save("live", f"clip {req['id']} queued")
        threading.Thread(target=self._render, args=(req,), daemon=True, name="live-render").start()

    def clip_last(self, seconds: float = 30, why: str = "Manual") -> dict:
        end = self.samples / SR
        if end < 5:
            raise StageError("nothing recorded yet")
        self._queue_clip(max(0.0, end - seconds), end, why)
        return {"id": self.clips[-1]["id"], "start": self.clips[-1]["start"]}

    # ------------------------------------------------------------------ render
    def _render(self, req: dict) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            self._scan_segments()
            if self.ended or (self.segments and self.segments[-1][2] >= req["end"] + 2):
                break
            time.sleep(1)
        with self.lock:
            segs = [s for s in self.segments
                    if s[2] > req["start"] - PAD and s[1] < req["end"] + PAD and s[0].exists()]
        if not segs:
            req.update(status="error", error="that moment is no longer in the buffer")
            self._save("live", f"clip {req['id']} failed: outside buffer")
            return
        base = segs[0][1]
        cstart = max(0.0, req["start"] - base)
        cend = min(segs[-1][2] - base, req["end"] - base)
        if cend - cstart < 2:
            req.update(status="error", error="buffer segment too short")
            self._save("live", f"clip {req['id']} failed: too short")
            return
        cid = req["id"]
        lst = self.clipdir / f"{cid}.txt"
        lst.write_text("".join(f"file '{s[0].as_posix()}'\n" for s in segs), encoding="utf-8")
        src = self.clipdir / f"{cid}.src.ts"
        try:
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                 "-c", "copy", str(src)], timeout=180)
        except StageError as e:
            req.update(status="error", error=str(e)[:200])
            self._save("live", f"clip {cid} failed to stitch")
            return
        finally:
            lst.unlink(missing_ok=True)

        with self.lock:
            local_words = [{"w": w["w"], "s": round(w["s"] - base, 3),
                            "e": round(w["e"] - base, 3)}
                           for w in self.words if w["s"] >= base - 0.05 and w["e"] <= segs[-1][2] + 0.3]
        text = " ".join(w["w"] for w in local_words
                        if w["s"] >= cstart - 0.05 and w["e"] <= cend + 0.3)
        title = (text[:60].strip() or self.title)
        opts = {**F.DEFAULTS, "profile_used": "stream", "captions": "hormozi",
                "layout": "auto", "llm": "none", "credit": self.credit,
                "transcript": {"words": local_words}}
        clip = {"start": cstart, "end": cend, "title": title,
                "hook_text": text[:80] or None, "score": 0.0}
        meta = {"kind": "file", "source": str(src), "path": str(src), "title": title,
                "wav": ""}
        try:
            made = F.render_clip(clip, self.n, meta, self.dir, opts)
        except Exception as e:  # noqa: BLE001
            req.update(status="error", error=str(e)[:200])
            self._save("live", f"clip {cid} render failed")
            return
        final = self.clipdir / f"{cid}.mp4"
        rendered = self.dir / made["file"]
        if rendered.exists():
            rendered.replace(final)
        caption = f"{title}\n\nCredit: {self.credit}" if self.credit else title
        final.with_suffix(".txt").write_text(caption, encoding="utf-8")
        req.update(status="done", file=str(final), title=title,
                   qa=made.get("qa", {}).get("pass", False))
        W.log(self.workspace, f"clip {cid} rendered → {final.name}")
        self._save("live", f"clip {cid} ready")

    # ------------------------------------------------------------------- state
    def _save(self, status: str, stage: str = "") -> None:
        with self.lock:
            data = {"name": self.name, "workspace": self.workspace, "url": self.url,
                    "status": status, "stage": stage, "title": self.title,
                    "uploader": self.uploader, "started": self.started,
                    "elapsed": round(self.samples / SR, 1),
                    "buffer_sec": round(self.segments[-1][2] - self.segments[0][1], 1)
                    if len(self.segments) > 1 else 0,
                    "words": len(self.words), "spikes": self.spikes,
                    "error": self.error, "clips": list(self.clips),
                    "last_lines": [" ".join(w["w"] for w in self.words[-12:])[:200]]}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def status(self) -> dict:
        return read_json(self.state_path, {"name": self.name, "status": "unknown"})


# ------------------------------------------------------------------ module API

def start_session(url: str, workspace: str = "default", **kw) -> Session:
    s = Session(url, workspace, **kw)
    s._save("starting", "connecting to stream")
    s.start()
    return s


def status_all() -> list[dict]:
    out = []
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for d in sorted(LIVE_DIR.iterdir(), reverse=True):
        st = read_json(d / "session.json", None)
        if isinstance(st, dict) and "status" in st:
            s = SESSIONS.get(st["name"])
            if s:
                st = s.status()
            out.append(st)
    return out


def get_session(name: str) -> Session | None:
    s = SESSIONS.get(name)
    if s:
        return s
    st = read_json(LIVE_DIR / name / "session.json", None)
    if not isinstance(st, dict):
        return None
    return Session(st["url"], st.get("workspace") or "default", name=name,
                   buffer_min=st.get("buffer_min", 15))


def stop_session(name: str) -> bool:
    s = SESSIONS.get(name)
    if not s:
        return False
    s.stop()
    return True


def clip_last(name: str, seconds: float = 30) -> dict:
    s = SESSIONS.get(name)
    if not s:
        raise StageError(f"session {name} is not running")
    return s.clip_last(seconds)


def clips(name: str) -> list[dict]:
    st = read_json(LIVE_DIR / name / "session.json", None)
    return (st or {}).get("clips", [])
