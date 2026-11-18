"""Lockoff local web UI — http://127.0.0.1:8765 (this PC only).

Standard library only. Jobs run one at a time in a queue, each as a separate
`python -m lockoff ...` process, so the UI stays responsive and a crash in
a job never takes the UI down. Videos are served with HTTP Range support so the
browser player can seek.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .util import JOBS, ROOT, read_json, write_json

HOST, PORT = "127.0.0.1", 8765
PAGE = Path(__file__).with_name("ui.html")
LOGS = ROOT / "Jobs" / "_logs"

# ------------------------------------------------------------------ job queue
_q: "queue.Queue[dict]" = queue.Queue()
_state = {"current": None, "queued": [], "history": []}
_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def _args_for(task: dict) -> list[str]:
    o = task["opts"]
    if task["kind"] == "rerender":
        a = ["rerender", task["job"]]
        if o.get("clip"):
            a += ["--clip", str(o["clip"])]
        if o.get("layout") and o["layout"] != "keep":
            a += ["--layout", o["layout"]]
    else:
        a = ["run", task["source"]]
        for k in ("clips", "length", "profile", "layout", "llm", "whisper", "lang", "credit", "captions"):
            v = o.get(k)
            if v not in (None, "", "auto") or (k in ("layout", "profile", "length") and v):
                a += [f"--{k}", str(v)]
        if o.get("hook") is False:
            a.append("--no-hook")
        if o.get("director") is True:
            a += ["--director", "on"]
        if o.get("fresh"):
            a.append("--fresh")
    a += ["--power", o.get("power") or "balanced"]
    return a


def _worker() -> None:
    global _proc
    LOGS.mkdir(parents=True, exist_ok=True)
    while True:
        task = _q.get()
        with _lock:
            _state["queued"] = [t for t in _state["queued"] if t["id"] != task["id"]]
            task.update(status="running", started=time.time())
            _state["current"] = task
        log = LOGS / f"{task['id']}.log"
        task["log"] = str(log)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ cf " + " ".join(_args_for(task)) + "\n")
            fh.flush()
            _proc = subprocess.Popen([sys.executable, "-m", "lockoff", *_args_for(task)], cwd=ROOT,
                                     stdout=fh, stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            code = _proc.wait()
        with _lock:
            task.update(status="done" if code == 0 else ("stopped" if task.get("stop") else "failed"),
                        code=code, ended=time.time())
            _state["history"].insert(0, task)
            _state["history"] = _state["history"][:30]
            _state["current"] = None
            _proc = None


def enqueue(task: dict) -> dict:
    task.update(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{len(_state['history']) + _q.qsize()}",
                status="queued", created=time.time())
    with _lock:
        _state["queued"].append(task)
    _q.put(task)
    return task


def stop_current() -> bool:
    with _lock:
        if _proc is None or _state["current"] is None:
            return False
        _state["current"]["stop"] = True
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(_proc.pid)], capture_output=True)
        return True


# ------------------------------------------------------------------ data helpers
def _tail(path: str | None, n: int = 60) -> str:
    if not path or not Path(path).exists():
        return ""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in lines if not re.search(r"HF_TOKEN|WARN:0|Targets are not supported", ln)]
    return "\n".join(lines[-n:])


def jobs() -> list[dict]:
    out = []
    for d in sorted(JOBS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("_logs"):
            continue
        plan = read_json(d / "plan.json")
        src = read_json(d / "_work" / "source.json") or {}
        out.append({"name": d.name, "title": (plan or {}).get("source", src).get("title") or d.name,
                    "clips": len((plan or {}).get("clips", [])), "has_plan": bool(plan),
                    "made": (plan or {}).get("made"), "profile": (plan or {}).get("profile"),
                    "passed": sum(1 for c in (plan or {}).get("clips", []) if c.get("qa", {}).get("pass"))})
    return out


def job_detail(name: str) -> dict | None:
    d = _job_path(name)
    plan = read_json(d / "plan.json") if d else None
    if not plan:
        return None
    for c in plan["clips"]:
        c.pop("words", None)  # large; edited via the captions endpoint
    return plan


def _job_path(name: str) -> Path | None:
    d = (JOBS / name).resolve()
    return d if d.parent == JOBS.resolve() and d.is_dir() else None


def save_clip(name: str, idx: int, patch: dict) -> dict:
    d = _job_path(name)
    plan = read_json(d / "plan.json")
    c = plan["clips"][idx - 1]
    for k in ("title", "hook_text", "description", "status", "note"):
        if k in patch:
            c[k] = patch[k]
    if "hashtags" in patch:
        c["hashtags"] = [t if t.startswith("#") else f"#{t}" for t in re.split(r"[\s,]+", patch["hashtags"]) if t]
    for k in ("start", "end"):
        if k in patch and patch[k] not in (None, ""):
            c[k] = round(float(patch[k]), 2)
    if patch.get("layout") and patch["layout"] != "auto":
        c["layout"] = patch["layout"]
    elif patch.get("layout") == "auto":
        c.pop("layout", None)
    write_json(d / "plan.json", plan)
    return c


def captions(name: str, idx: int, text: str | None = None) -> dict:
    """GET: caption words as editable text. POST: replace words 1:1 (same count keeps timing)."""
    d = _job_path(name)
    plan = read_json(d / "plan.json")
    c = plan["clips"][idx - 1]
    words = c.get("words") or []
    if text is None:
        return {"text": " ".join(w["w"] for w in words), "count": len(words)}
    new = text.split()
    if len(new) == len(words):
        for w, t in zip(words, new):
            w["w"] = t
    else:  # different count: spread the new words over the same time span
        if not words:
            return {"error": "no caption timing for this clip"}
        s, e = words[0]["s"], words[-1]["e"]
        step = (e - s) / max(1, len(new))
        c["words"] = [{"w": t, "s": round(s + i * step, 3), "e": round(s + (i + 1) * step - 0.02, 3)}
                      for i, t in enumerate(new)]
    write_json(d / "plan.json", plan)
    return {"ok": True, "count": len(new)}


def signals_for(name: str) -> dict:
    """Moments map data: the fused interest curve (≤600 points) + clip ranges."""
    d = _job_path(name)
    if not d:
        return {"error": "not found"}
    plan = read_json(d / "plan.json") or {}
    sig = None
    for f in sorted((d / "_work").glob("signals-*.json")):
        sig = read_json(f)
        if plan.get("profile") and f.stem.endswith(plan["profile"]):
            break
    fused = (sig or {}).get("fused") or []
    n = len(fused)
    step = max(1, n // 600)
    curve = [round(max(fused[i:i + step]), 3) for i in range(0, n, step)] if n else []
    clips = [{"i": i + 1, "start": c["start"], "end": c["end"], "title": c.get("title", ""),
              "status": c.get("status", "")} for i, c in enumerate(plan.get("clips", []))]
    return {"duration": float((plan.get("source") or {}).get("duration") or n), "step": step,
            "curve": curve, "clips": clips}


ASSETS = Path(__file__).with_name("assets")
BRANDING = ROOT / "branding"


def pick_file() -> str:
    """Native Windows 'Open' dialog (this server runs on the user's own PC)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="Choose a video for Lockoff", filetypes=[
        ("Video", "*.mp4 *.mkv *.mov *.webm *.m4v *.avi *.flv *.ts"), ("All files", "*.*")])
    root.destroy()
    return path or ""


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet console
        pass

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path: Path):
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        p = unquote(u.path)
        if p == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/status":
            with _lock:
                cur = dict(_state["current"]) if _state["current"] else None
                hist = [dict(h) for h in _state["history"][:8]]
                qd = [dict(t) for t in _state["queued"]]
            if cur:
                cur["tail"] = _tail(cur.get("log"))
            if hist:
                hist[0]["tail"] = _tail(hist[0].get("log"), 25)
            self._json({"current": cur, "queued": qd, "history": hist})
        elif p == "/api/jobs":
            self._json(jobs())
        elif p.startswith("/api/job/"):
            plan = job_detail(p.split("/")[3])
            self._json(plan or {"error": "not found"}, 200 if plan else 404)
        elif p.startswith("/api/captions/"):
            _, _, _, name, idx = p.split("/")
            self._json(captions(name, int(idx)))
        elif p == "/api/pick":
            self._json({"path": pick_file()})
        elif p.startswith("/api/signals/"):
            self._json(signals_for(p.split("/")[3]))
        elif p.startswith("/assets/") or p.startswith("/brand/"):
            root = ASSETS if p.startswith("/assets/") else BRANDING
            f = (root / p.split("/", 2)[2]).resolve()
            if f.is_file() and root.resolve() in f.parents and f.suffix.lower() in (".ttf", ".woff2", ".svg", ".png"):
                self.send_response(200)
                self.send_header("Content-Type", {".ttf": "font/ttf", ".woff2": "font/woff2", ".svg": "image/svg+xml",
                                                  ".png": "image/png"}[f.suffix.lower()])
                self.send_header("Cache-Control", "max-age=86400")
                body = f.read_bytes()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        elif p.startswith("/media/"):
            parts = p.split("/", 3)
            d = _job_path(parts[2]) if len(parts) > 3 else None
            f = (d / parts[3]).resolve() if d else None
            if f and f.parent == d and f.is_file() and f.suffix.lower() in (".mp4", ".jpg", ".md"):
                self._file(f)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        p = urlparse(self.path).path
        b = self._body()
        if p == "/api/run":
            src = (b.get("source") or "").strip().strip('"')
            if not src:
                return self._json({"error": "paste a link or choose a file"}, 400)
            if not re.match(r"https?://", src) and not Path(src).is_file():
                return self._json({"error": f"file not found: {src}"}, 400)
            self._json(enqueue({"kind": "run", "source": src, "opts": b.get("opts") or {}}))
        elif p == "/api/rerender":
            if not _job_path(b.get("job", "")):
                return self._json({"error": "no such job"}, 404)
            self._json(enqueue({"kind": "rerender", "job": b["job"], "source": b["job"],
                                "opts": {"clip": b.get("clip"), "layout": b.get("layout"),
                                         "power": b.get("power")}}))
        elif p == "/api/stop":
            self._json({"stopped": stop_current()})
        elif p.startswith("/api/clip/"):
            _, _, _, name, idx = p.split("/")
            self._json(save_clip(name, int(idx), b))
        elif p.startswith("/api/captions/"):
            _, _, _, name, idx = p.split("/")
            self._json(captions(name, int(idx), b.get("text", "")))
        elif p.startswith("/api/open/"):
            d = _job_path(p.split("/")[3])
            if d:
                os.startfile(d)  # noqa: S606 — opens Explorer on this PC
            self._json({"ok": bool(d)})
        else:
            self.send_error(404)


def serve(open_browser: bool = True) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"Lockoff UI running at {url}  (close this window to stop it)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_current()
