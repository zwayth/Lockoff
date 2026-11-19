"""Per-creator workspaces and a local activity log.

PeakCut keeps every creator separate (their own live sessions, settings and
history). Lockoff does the same locally: a workspace is just a named channel
with its own credit line and live preferences. Everything lives in one small
JSON file at the repo root, so no server or sign-in is needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .util import ROOT, read_json

STATE = ROOT / "lockoff_workspaces.json"


def _state() -> dict:
    st = read_json(STATE, None)
    if not isinstance(st, dict):
        st = {"creators": {}, "activity": []}
    st.setdefault("creators", {})
    st.setdefault("activity", [])
    return st


def _save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def all_creators() -> dict:
    return _state()["creators"]


def get(name: str) -> dict:
    return _state()["creators"].get(name, {})


def upsert(name: str, channel: str = "", credit: str = "", triggers: str = "",
           buffer_min: int = 15, auto_clip: bool = True) -> dict:
    st = _state()
    cur = st["creators"].get(name, {})
    cur.update(name=name, channel=channel.strip(), credit=credit.strip(),
               triggers=triggers.strip(), buffer_min=int(buffer_min),
               auto_clip=bool(auto_clip), updated=time.time())
    st["creators"][name] = cur
    _save(st)
    log(name, f"workspace saved ({channel or 'no channel'})")
    return cur


def remove(name: str) -> bool:
    st = _state()
    ok = st["creators"].pop(name, None) is not None
    if ok:
        _save(st)
    return ok


def log(creator: str, message: str) -> None:
    st = _state()
    st["activity"].append({"t": time.time(), "creator": creator, "message": str(message)[:300]})
    if len(st["activity"]) > 500:      # keep the last 500 events
        del st["activity"][:-500]
    _save(st)


def activity(limit: int = 50) -> list[dict]:
    return list(reversed(_state()["activity"][:limit]))
