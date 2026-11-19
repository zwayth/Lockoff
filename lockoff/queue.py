"""Lockoff's post queue: publish rendered clips and keep their URLs alive.

Shares the proven Buffer mechanics with Clip Factory but has its own state
file, so scheduling the two products can never double-post or step on each
other. The URL refresh is the fix we proved in Clip Factory: uguu.se links die
in ~3 h, so any post due within 2 h is re-uploaded and swapped in with
`editPost` before Buffer fetches it.

    lockoff queue [--dry-run] [--max 10]
    lockoff refresh-urls [--dry-run]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import social as S
from .util import JOBS, ROOT, log, read_json

STATE = ROOT / "lockoff_queue.json"


def _state() -> dict:
    st = read_json(STATE, None)
    if not isinstance(st, dict):
        st = {"posted": [], "media": {}}
    st.setdefault("posted", [])
    st.setdefault("media", {})
    return st


def _save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


POSTS_Q = """query($o: OrganizationId!, $n: Int) {
  posts(first: $n, input: { organizationId: $o }) {
    edges { node { id status dueAt text
      assets { __typename ... on VideoAsset { video { title } } }
    } }
  }
}"""


def candidates() -> list[dict]:
    """Every rendered clip Lockoff owns, newest first."""
    out: list[dict] = []
    for mp4 in sorted((JOBS / "_live").glob("*/clips/*.mp4"), reverse=True):
        sidecar = mp4.with_suffix(".txt")
        caption = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else mp4.stem
        out.append({"key": str(mp4), "file": mp4, "title": mp4.stem,
                    "caption": caption, "source": "live"})
    for plan in sorted(JOBS.glob("*/plan.json"), reverse=True):
        data = read_json(plan, {}) or {}
        for i, c in enumerate(data.get("clips", []), 1):
            f = plan.parent / str(c.get("file") or "")
            if f.exists() and c.get("qa", {}).get("pass", True):
                out.append({"key": str(f), "file": f, "title": str(c.get("title") or f.stem),
                            "caption": S.build_caption(c, data.get("opts", {}).get("credit", "")),
                            "source": "job"})
    return out


def scheduled_count(buf: S.Buffer, cfg: dict) -> int:
    data = buf.gql(POSTS_Q, {"o": cfg["organization_id"], "n": 50})["posts"]
    return sum(1 for e in data["edges"] if e["node"]["status"] == "scheduled")


def queue(max_scheduled: int = 10, mode: str = "addToQueue",
          dry_run: bool = False) -> list[str]:
    cfg = S.load_config()
    buf = S.Buffer(cfg["buffer_token"])
    channel = S.resolve_channel(buf, cfg)   # also fills cfg["organization_id"]
    st = _state()
    posted = set(st["posted"])
    todo = [c for c in candidates() if c["key"] not in posted]
    log(f"{len(todo)} clip(s) waiting, {scheduled_count(buf, cfg)}/{max_scheduled} scheduled")

    sent: list[str] = []
    for c in todo:
        room = max_scheduled - scheduled_count(buf, cfg)
        if room <= 0 and mode == "addToQueue":
            log("Buffer schedule full — remaining clips stay queued locally")
            break
        url = S.upload_public(c["file"]) if not dry_run else "https://example.invalid/dry.mp4"
        if dry_run:
            log(f"[dry-run] would queue {c['file'].name}: {c['caption'].splitlines()[0][:60]}")
            sent.append(c["file"].name)
            continue
        try:
            p = buf.create_post(channel["id"], url, c["caption"], mode=mode,
                                title=c["title"])
        except S.BufferError as e:
            log(f"could not queue {c['file'].name}: {e}")
            continue
        posted.add(c["key"])
        st["posted"] = sorted(posted)
        st["media"][p["id"]] = {"file": str(c["file"]), "title": c["title"],
                                "refreshed_at": time.time()}
        _save(st)
        sent.append(p["id"])
        log(f"queued {c['file'].name}: post {p['id']} ({p['status']})")
    return sent


def refresh_due_urls(lookahead_hours: float = 2, dry_run: bool = False) -> int:
    cfg = S.load_config()
    buf = S.Buffer(cfg["buffer_token"])
    posts = [e["node"] for e in buf.gql(
        POSTS_Q, {"o": cfg["organization_id"], "n": 50})["posts"]["edges"]]
    now = time.time()
    soon = []
    for p in posts:
        if p["status"] != "scheduled" or not p.get("dueAt"):
            continue
        try:
            when = time.mktime(time.strptime(p["dueAt"], "%Y-%m-%dT%H:%M:%S.%fZ")) - time.timezone
        except ValueError:
            try:
                when = time.mktime(time.strptime(p["dueAt"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            except ValueError:
                continue
        if (when - now) / 3600 <= lookahead_hours:
            soon.append(p)
    st = _state()
    changed = False
    count = 0
    for p in soon:
        rec = st["media"].get(p["id"]) or {}
        if 0 < now - rec.get("refreshed_at", 0) < 3600 and rec.get("file"):
            continue
        f = Path(rec["file"]) if rec.get("file") else None
        if not f or not f.exists():
            log(f"no local file for post {p['id']} — cannot refresh")
            continue
        if dry_run:
            log(f"[dry-run] would refresh {p['id']}: {f.name}")
            count += 1
            continue
        try:
            url = S.upload_public(f)
            buf.edit_post(p["id"], url, rec.get("title") or "")
        except S.BufferError as e:
            log(f"could not refresh {p['id']}: {e}")
            continue
        st["media"][p["id"]] = {**rec, "refreshed_at": now}
        changed = True
        count += 1
        log(f"refreshed URL for post {p['id']} (due {p['dueAt']})")
    if changed:
        _save(st)
    return count
