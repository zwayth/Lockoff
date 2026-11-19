"""Publish rendered clips to Instagram through Buffer's GraphQL API.

This is the proven Clip Factory Buffer client, kept as its own copy so Lockoff
has separate config, state and credentials. Buffer has no file upload endpoint:
it fetches the video from a public URL. So a finished clip is pushed to a
temporary public host first, then handed to Buffer.

Config lives in `lockoff_social.json` at the repo root (created by `lockoff post --setup`):

    {
      "buffer_token": "...",
      "organization_id": "...",
      "channel_id": "...",          // optional: first Instagram channel is used
      "credit": "@handle",          // default credit line
      "host": "catbox"              // catbox | none
    }

Environment variables BUFFER_TOKEN / BUFFER_ORG / BUFFER_CHANNEL override the file.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .util import ROOT, log

API = "https://api.buffer.com"
UGUU = "https://uguu.se/upload.php"
CONFIG = ROOT / "lockoff_social.json"

CAPTION_LIMIT = 2200
MAX_HASHTAGS = 30


# ---------------------------------------------------------------- config ----

def load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["buffer_token"] = os.environ.get("BUFFER_TOKEN") or cfg.get("buffer_token", "")
    cfg["organization_id"] = os.environ.get("BUFFER_ORG") or cfg.get("organization_id", "")
    cfg["channel_id"] = os.environ.get("BUFFER_CHANNEL") or cfg.get("channel_id", "")
    cfg.setdefault("credit", "")
    cfg.setdefault("host", "uguu")
    return cfg


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"wrote {CONFIG}")


# ---------------------------------------------------------------- buffer ----

class BufferError(RuntimeError):
    pass


class Buffer:
    def __init__(self, token: str):
        if not token:
            raise BufferError("no Buffer token — run `lockoff post --setup` or set BUFFER_TOKEN")
        self.token = token

    def gql(self, query: str, variables: dict | None = None) -> dict:
        r = requests.post(
            API,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            timeout=60,
        )
        try:
            body = r.json()
        except ValueError:
            raise BufferError(f"HTTP {r.status_code}: {r.text[:300]}") from None
        if body.get("errors"):
            msg = "; ".join(e.get("message", "?") for e in body["errors"])
            raise BufferError(msg)
        return body.get("data") or {}

    # -- reads -------------------------------------------------------------
    def organizations(self) -> list[dict]:
        return self.gql("query { account { organizations { id name } } }")["account"]["organizations"]

    def channels(self, org_id: str) -> list[dict]:
        q = """query($o: OrganizationId!) {
                 channels(input: { organizationId: $o }) { id name service avatar }
               }"""
        return self.gql(q, {"o": org_id})["channels"]

    def posts(self, org_id: str, first: int = 20) -> list[dict]:
        q = """query($o: OrganizationId!, $n: Int) {
                 posts(first: $n, input: { organizationId: $o }) {
                   edges { node { id text status sentAt channelId } }
                 }
               }"""
        data = self.gql(q, {"o": org_id, "n": first})["posts"]
        return [e["node"] for e in data.get("edges", [])]

    # -- write -------------------------------------------------------------
    def create_post(self, channel_id: str, video_url: str, text: str,
                    mode: str = "shareNow", due_at: str | None = None,
                    title: str | None = None, first_comment: str | None = None,
                    share_to_feed: bool = True) -> dict:
        mutation = """
        mutation($input: CreatePostInput!) {
          createPost(input: $input) {
            __typename
            ... on PostActionSuccess { post { id status shareMode sentAt dueAt text } }
            ... on InvalidInputError { message }
            ... on LimitReachedError { message }
            ... on UnauthorizedError { message }
            ... on NotFoundError { message }
            ... on UnexpectedError { message }
            ... on RestProxyError { message }
          }
        }"""
        variables = {
            "input": {
                "channelId": channel_id,
                "text": text,
                "assets": [{"video": {"url": video_url, "metadata": {"title": title or ""}}}],
                "mode": mode,
                "schedulingType": "automatic",
                "needsApproval": False,
                "metadata": {
                    "instagram": {
                        "type": "reel",
                        "shouldShareToFeed": share_to_feed,
                        **({"firstComment": first_comment} if first_comment else {}),
                    }
                },
                **({"dueAt": due_at} if due_at else {}),
            }
        }
        out = self.gql(mutation, variables)["createPost"]
        if out.get("__typename") != "PostActionSuccess":
            raise BufferError(out.get("message") or f"createPost failed: {out}")
        return out["post"]

    def edit_post(self, post_id: str, video_url: str, title: str = "") -> dict:
        """Swap the media URL on an existing post (fresh URL before it publishes)."""
        mutation = """
        mutation($input: EditPostInput!) {
          editPost(input: $input) {
            __typename
            ... on PostActionSuccess { post { id status dueAt } }
            ... on InvalidInputError { message }
            ... on LimitReachedError { message }
            ... on UnauthorizedError { message }
            ... on NotFoundError { message }
            ... on UnexpectedError { message }
            ... on RestProxyError { message }
          }
        }"""
        out = self.gql(mutation, {"input": {
            "id": post_id,
            "assets": [{"video": {"url": video_url, "metadata": {"title": title}}}],
        }})["editPost"]
        if out.get("__typename") != "PostActionSuccess":
            raise BufferError(out.get("message") or f"editPost failed: {out}")
        return out["post"]

    def delete_post(self, post_id: str) -> bool:
        mutation = """
        mutation($input: DeletePostInput!) {
          deletePost(input: $input) {
            __typename
            ... on DeletePostSuccess { id }
            ... on VoidMutationError { message }
          }
        }"""
        out = self.gql(mutation, {"input": {"id": post_id}})["deletePost"]
        return out.get("__typename") == "DeletePostSuccess"


# --------------------------------------------------------------- hosting ----

def upload_public(path: Path, timeout: int = 600) -> str:
    """Push a clip to a public host and return a direct URL Buffer can fetch.

    Buffer pulls the file itself, so the host has to serve raw bytes with Range
    support. uguu.se does (files live ~3 h — plenty, Buffer fetches within
    seconds). catbox.moe and tmpfiles.org both fail Buffer's fetch, so they are
    not used.
    """
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    size = path.stat().st_size
    log(f"uploading {path.name} ({size / 1e6:.1f} MB)")
    with path.open("rb") as fh:
        r = requests.post(
            UGUU,
            files={"files[]": (path.name, fh, mime)},
            headers={"User-Agent": "Mozilla/5.0 (ClipFactory)"},
            timeout=timeout,
        )
    if r.status_code != 200:
        raise BufferError(f"uguu upload failed: HTTP {r.status_code} {r.text[:200]}")
    try:
        url = r.json()["files"][0]["url"]
    except (ValueError, KeyError, IndexError):
        raise BufferError(f"uguu upload failed: {r.text[:200]}") from None
    check = requests.get(url, headers={"Range": "bytes=0-1023"}, timeout=60)
    if check.status_code not in (200, 206):
        raise BufferError(f"uploaded file not readable at {url} (HTTP {check.status_code})")
    return url


# --------------------------------------------------------------- captions ---

def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_caption(clip: dict, credit: str = "", extra: str = "") -> str:
    parts: list[str] = []
    title = _clean(clip.get("title", ""))
    hook = _clean(clip.get("hook_text", ""))
    desc = _clean(clip.get("description", ""))

    if title:
        parts.append(title)
    if desc and desc.lower() != title.lower():
        parts.append(desc)
    if extra:
        parts.append(_clean(extra))

    tags = [t if t.startswith("#") else f"#{t}" for t in (clip.get("hashtags") or [])][:MAX_HASHTAGS]
    if tags:
        parts.append(" ".join(tags))
    if credit:
        parts.append(f"Credit: {credit}" if not credit.lower().startswith("credit") else credit)

    text = "\n\n".join(p for p in parts if p)
    if not text:
        text = hook or "Clip"
    return text[:CAPTION_LIMIT]


# ------------------------------------------------------------------ flow ----

def resolve_channel(buf: Buffer, cfg: dict) -> dict:
    org = cfg.get("organization_id") or (buf.organizations()[0]["id"] if buf.organizations() else "")
    if not org:
        raise BufferError("no Buffer organization on this account")
    cfg["organization_id"] = org
    chans = [c for c in buf.channels(org) if c["service"] == "instagram"]
    if not chans:
        raise BufferError("no Instagram channel connected in Buffer")
    wanted = cfg.get("channel_id")
    for c in chans:
        if c["id"] == wanted:
            return c
    return chans[0]


def _load_plan(job_dir: Path) -> dict:
    p = Path(job_dir) / "plan.json"
    if not p.exists():
        raise BufferError(f"no plan.json in {job_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def post_clip(job_dir: Path, clip_no: int, mode: str = "shareNow", dry_run: bool = False,
              credit: str | None = None, due_at: str | None = None,
              first_comment: str | None = None) -> dict:
    plan = _load_plan(job_dir)
    clips = plan["clips"]
    if clip_no < 1 or clip_no > len(clips):
        raise BufferError(f"clip {clip_no} out of range (1..{len(clips)})")
    clip = clips[clip_no - 1]
    video = job_dir / clip["file"]
    if not video.exists():
        raise BufferError(f"missing render: {video}")

    cfg = load_config()
    credit = credit if credit is not None else cfg.get("credit", "")
    text = build_caption(clip, credit)

    log(f"clip {clip_no}: {clip.get('title', '')[:60]}")
    log(f"caption:\n{text}")
    if dry_run:
        return {"dry_run": True, "file": str(video), "caption": text, "mode": mode}

    if cfg.get("host") == "none":
        raise BufferError("host=none: Buffer needs a public URL for the video")

    buf = Buffer(cfg["buffer_token"])
    channel = resolve_channel(buf, cfg)
    url = upload_public(video)
    log(f"public url: {url}")

    post = buf.create_post(
        channel_id=channel["id"],
        video_url=url,
        text=text,
        mode=mode,
        due_at=due_at,
        title=_clean(clip.get("title", ""))[:100],
        first_comment=first_comment,
    )
    log(f"queued to {channel['name']} ({channel['service']}): post {post['id']} — {post['status']}")

    # remember it, so `cf autopost` never sends the same clip twice
    st = _state()
    st.setdefault("posted", {})[f"{Path(job_dir).name}/{clip.get('file', clip_no)}"] = {
        "post_id": post["id"],
        "at": post.get("sentAt") or post.get("dueAt") or time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_state(st)

    return {"post": post, "channel": channel, "url": url, "caption": text}


def post_job(job_dir: Path, clip_nos: list[int] | None = None, mode: str = "shareNow",
             dry_run: bool = False, credit: str | None = None, gap_minutes: int = 0) -> list[dict]:
    plan = _load_plan(job_dir)
    n = len(plan["clips"])
    clip_nos = clip_nos or list(range(1, n + 1))
    out = []
    for i, no in enumerate(clip_nos):
        if i and gap_minutes and not dry_run:
            time.sleep(gap_minutes * 60)
        out.append(post_clip(job_dir, no, mode=mode, dry_run=dry_run, credit=credit))
    return out


# ------------------------------------------------------------- automation ---

STATE = ROOT / "lockoff_social_state.json"


def _state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"posted": {}}


def _save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def autopost(mode: str = "addToQueue", per_job: int = 1, dry_run: bool = False,
             credit: str | None = None, skip_unapproved: bool = True) -> list[dict]:
    """Post clips from finished jobs that have not been posted yet.

    Runs once; call it from a loop or a scheduled task for hands-off operation.
    Newest job first, at most `per_job` clips from each, QA failures skipped.
    """
    from .util import JOBS

    st = _state()
    posted = st.setdefault("posted", {})
    results: list[dict] = []

    jobs = sorted((p for p in JOBS.iterdir() if p.is_dir() and (p / "plan.json").exists()),
                  key=lambda p: (p / "plan.json").stat().st_mtime, reverse=True)

    for job in jobs:
        try:
            plan = _load_plan(job)
        except BufferError:
            continue
        sent_here = 0
        for i, clip in enumerate(plan.get("clips", []), 1):
            if sent_here >= per_job:
                break
            key = f"{job.name}/{clip.get('file', i)}"
            if key in posted:
                continue
            if skip_unapproved and clip.get("qa") and not clip["qa"].get("pass", False):
                log(f"skip {key}: QA did not pass")
                continue
            if not (job / clip.get("file", "")).exists():
                continue
            try:
                res = post_clip(job, i, mode=mode, dry_run=dry_run, credit=credit)
            except BufferError as e:
                log(f"failed {key}: {e}")
                continue
            sent_here += 1
            results.append(res)
            if not dry_run:
                posted[key] = {"post_id": res["post"]["id"], "at": res["post"].get("sentAt")
                               or res["post"].get("dueAt") or time.strftime("%Y-%m-%dT%H:%M:%S")}
                _save_state(st)
    if not results:
        log("nothing new to post")
    return results


# ------------------------------------------------------------------ setup ---
def setup_interactive(token: str | None = None) -> dict:
    cfg = load_config()
    if token:
        cfg["buffer_token"] = token
    if not cfg["buffer_token"]:
        cfg["buffer_token"] = input("Buffer API key: ").strip()
    buf = Buffer(cfg["buffer_token"])
    orgs = buf.organizations()
    if not orgs:
        raise BufferError("no organizations on this account")
    cfg["organization_id"] = orgs[0]["id"]
    chans = buf.channels(orgs[0]["id"])
    print("\nChannels:")
    for i, c in enumerate(chans, 1):
        print(f"  {i}. {c['service']:<10} {c['name']}  ({c['id']})")
    ig = [c for c in chans if c["service"] == "instagram"]
    if not ig:
        raise BufferError("no Instagram channel connected — connect one in Buffer first")
    cfg["channel_id"] = ig[0]["id"]
    if not cfg.get("credit"):
        cfg["credit"] = input(f"\nDefault credit for {ig[0]['name']} (e.g. @handle, blank for none): ").strip()
    save_config(cfg)
    print(f"\nReady. Posts go to Instagram channel '{ig[0]['name']}'.")
    return cfg
