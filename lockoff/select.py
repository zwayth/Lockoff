"""Moment selection: LLM over the timestamped transcript + signal anchors, then a
deterministic validation layer that makes every pick safe to cut.

Why the validation layer exists: a one-shot prompt returns plausible but sloppy
ranges (mid-word starts, "so…" openings, 90s clips, duplicates, made-up times).
Every LLM answer here goes through: JSON repair + retry → clamp to the video →
snap start/end to sentence boundaries on the word timeline → fix back-reference
openings → enforce min/max length → de-duplicate overlaps → blend with the
signal score. With no LLM available, signal peaks become the candidates, so the
factory never stops because of the model.

Prompt rules are merged from supoclip (grounding, boundary rules, virality
sub-scores), podcli (payoff-first titles, "needs" check, keep-segments) and the
studio's stream prompt (moment types, 1–2s pre-roll).
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.request

from . import signals as S
from .util import log, run

BAD_OPEN = {"so", "and", "but", "or", "yeah", "yes", "yep", "right", "exactly", "totally", "that",
            "it", "this", "they", "he", "she", "which", "because", "um", "uh", "like", "well", "also",
            "anyway", "okay", "ok", "then"}
SENT_END = re.compile(r"[.?!…]$")

LENGTHS = {"short": (15, 30, 18, 28), "medium": (20, 45, 25, 40), "long": (30, 75, 35, 60),
           "auto": (15, 60, 22, 45)}

LENS = {
    "podcast": ("a senior short-form editor for podcasts and interviews",
                "bold or contrarian claims, surprising numbers, war stories, mistakes and lessons, "
                "emotional peaks, complete answers to interesting questions, funny exchanges"),
    "stream": ("a senior clipper for gaming and IRL livestreams",
               "clutch plays, intense fights, aces, rage, hysterical laughter, jumpscares, funny fails, "
               "chat-worthy reactions, wholesome or absurd moments; the action must be visible and the "
               "reaction audible"),
}

PROMPT = """You are {editor}. Pick the {n} best moments in this video for TikTok / YouTube Shorts / Reels.

Return ONLY valid JSON. No markdown, no code fences, no commentary.

TIMESTAMPS: the transcript lines are [start-end] in SECONDS. Return seconds as numbers (e.g. 812.4).
Use only times that appear in the transcript. Every clip is ONE contiguous range.

LENGTH: {mn}-{mx} s each, ideally {imn}-{imx} s. Quality over quantity: return fewer if fewer are great.

WHAT TO LOOK FOR ({lens}).
SIGNAL ANCHORS below are seconds where loudness, laughter/cheering, reaction words or replay data spiked.
They are strong hints, not orders: a great moment usually sits on or just before one (the cause comes
before the reaction, so include the ~5-10 s that set it up).

BOUNDARIES (critical):
- Start on the first word of a sentence a stranger can follow with no context. The first 2-3 seconds are the hook.
- For an action/reaction moment start 1-2 s before the action, not on the reaction.
- Never open on a filler or back-reference ("so", "and", "but", "yeah", "that", "it", "they", "as I said").
  If the moment answers a question, start on the question.
- End right after the payoff lands, on a complete sentence. Do not trail into the next topic.
- Avoid intros, sponsor reads, "be right back", technical problems, and anything needing unseen context.
- Spread picks across the whole video; no two picks may overlap.

FOR EACH CLIP:
- payoff: one sentence, second person, what the viewer gets. If you can't write it, skip the moment.
- title: written FROM the payoff, under 60 characters, never a copied transcript fragment.
- hook_text: on-screen text for the first 3 seconds, max 8 words, same language as the video.
- scores 0-10: hook, standalone, emotion, shareability.
- needs: what the viewer must already know, or "nothing". If not "nothing", move start earlier to cover it.
- description: 1-2 sentences for the post caption. No credit line and no hashtags in it (added separately).
- hashtags: 4-6, specific to the moment.

{already}SIGNAL ANCHORS:
{anchors}

JSON SHAPE:
{{"clips": [{{"start": 0.0, "end": 0.0, "title": "", "hook_text": "", "payoff": "", "needs": "nothing",
  "content_type": "", "scores": {{"hook": 0, "standalone": 0, "emotion": 0, "shareability": 0}},
  "why": "", "description": "", "hashtags": ["#tag"]}}]}}

TRANSCRIPT ({dur:.0f} s, language {lang}):
{transcript}"""


# ---------------------------------------------------------------- LLM backends
def _call_claude(prompt: str, model: str | None, timeout: int) -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not on PATH")
    args = [exe, "-p", "--output-format", "text"]
    if model:
        args += ["--model", model]
    r = run(args, timeout=timeout, stdin_text=prompt, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def _call_ollama(prompt: str, model: str, timeout: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0.2, "num_ctx": 32768}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def call_llm(prompt: str, llm: str, timeout: int = 900) -> str:
    kind, _, model = llm.partition(":")
    if kind == "claude":
        return _call_claude(prompt, model or None, timeout)
    if kind == "ollama":
        return _call_ollama(prompt, model or "qwen2.5:7b", timeout)
    raise RuntimeError(f"unknown llm '{llm}'")


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips fences/prose, fixes trailing commas."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    t = re.sub(r",\s*([}\]])", r"\1", t[a: b + 1])
    data = json.loads(t)
    if isinstance(data, list):
        data = {"clips": data}
    if not isinstance(data.get("clips"), list):
        raise ValueError("reply has no 'clips' list")
    return data


# ---------------------------------------------------------------- transcript packing
def transcript_lines(transcript: dict, start: float = 0, end: float = 1e12) -> str:
    out = []
    for s in transcript["segments"]:
        if start <= s["start"] < end:
            out.append(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    return "\n".join(out)


def windows(duration: float, size: float = 20 * 60, overlap: float = 2 * 60) -> list[tuple[float, float]]:
    """Long videos are judged in overlapping windows (supoclip), then merged + ranked."""
    if duration <= size * 1.25:
        return [(0.0, duration + 1)]
    out, s = [], 0.0
    while s < duration:
        out.append((s, min(duration + 1, s + size)))
        s += size - overlap
    return out


# ---------------------------------------------------------------- validation
def _sentence_starts(words: list[dict]) -> list[int]:
    idx = [0] if words else []
    for i in range(1, len(words)):
        if SENT_END.search(words[i - 1]["w"]) or words[i]["s"] - words[i - 1]["e"] > 0.7:
            idx.append(i)
    return idx


def _sentence_ends(words: list[dict]) -> list[int]:
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if SENT_END.search(w["w"]) or nxt is None or nxt["s"] - w["e"] > 0.6:
            out.append(i)
    return out


def _clean(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def snap(clip: dict, words: list[dict], duration: float, mn: float, mx: float) -> dict | None:
    """Snap a raw [start,end] onto sentence boundaries of the word timeline."""
    if not words:
        s, e = max(0.0, clip["start"]), min(duration, clip["end"])
        return {**clip, "start": s, "end": e} if mn <= e - s <= mx else None
    starts, ends = _sentence_starts(words), _sentence_ends(words)
    s0, e0 = float(clip["start"]), float(clip["end"])

    # start: nearest sentence start within [-6, +3] s, preferring earlier (keeps setup)
    cand = [i for i in starts if s0 - 6 <= words[i]["s"] <= s0 + 3]
    si = min(cand, key=lambda i: abs(words[i]["s"] - s0) + (0.5 if words[i]["s"] > s0 else 0)) if cand else \
        min(range(len(words)), key=lambda i: abs(words[i]["s"] - s0))
    # back-reference opening → walk back one sentence (max 10 s) to include what it points at
    for _ in range(2):
        if _clean(words[si]["w"]) in BAD_OPEN:
            prev = [i for i in starts if i < si and words[si]["s"] - words[i]["s"] <= 10]
            if prev:
                si = prev[-1]
                continue
        break
    start = words[si]["s"]

    # end: nearest sentence end within [-4, +5] s that respects length
    cand = [i for i in ends if e0 - 4 <= words[i]["e"] <= e0 + 5 and mn <= words[i]["e"] - start <= mx]
    if not cand:
        cand = [i for i in ends if mn <= words[i]["e"] - start <= mx]
    if not cand:
        return None
    ei = min(cand, key=lambda i: abs(words[i]["e"] - e0))
    end = words[ei]["e"]

    start = max(0.0, start - 0.15)
    end = min(duration, end + 0.35)
    if not (mn <= end - start <= mx + 0.6):
        return None
    return {**clip, "start": round(start, 2), "end": round(end, 2),
            "opens_with": " ".join(w["w"] for w in words[si: si + 6])}


def _overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))


def finalize(raw: list[dict], transcript: dict, sig: dict, duration: float, length: str,
             n: int, llm_weight: float) -> list[dict]:
    mn, mx, _, _ = LENGTHS[length]
    words = transcript.get("words", [])
    out = []
    for c in raw:
        try:
            c["start"], c["end"] = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if c["end"] <= c["start"] or c["start"] >= duration:
            continue
        sc = c.get("scores") or {}
        vals = [float(sc.get(k, 0) or 0) for k in ("hook", "standalone", "emotion", "shareability")]
        c["llm_score"] = round(sum(vals) / 40.0, 3) if any(vals) else c.get("llm_score", 0.0)
        fixed = snap(c, words, duration, mn, mx)
        if not fixed:
            log(f"  dropped pick {c['start']:.0f}-{c['end']:.0f}s (no valid sentence-bounded range)")
            continue
        fixed["signal_score"] = S.window_score(sig, fixed["start"], fixed["end"])
        fixed["score"] = round(llm_weight * fixed["llm_score"] + (1 - llm_weight) * fixed["signal_score"], 4)
        out.append(fixed)
    out.sort(key=lambda c: -c["score"])
    kept: list[dict] = []
    for c in out:
        if all(_overlap(c, k) < 0.3 for k in kept):
            kept.append(c)
        if len(kept) >= n:
            break
    return kept


# ---------------------------------------------------------------- candidates
def heuristic(transcript: dict, sig: dict, length: str, n: int) -> list[dict]:
    """No-LLM fallback: signal peaks → window with ~60% before the peak (cause→reaction)."""
    _, _, imn, imx = LENGTHS[length]
    target = (imn + imx) / 2
    out = []
    for p in S.peaks(sig["fused"], k=n * 3):
        s = max(0.0, p["t"] - target * 0.6)
        e = s + target
        text = " ".join(w["w"] for w in transcript["words"] if s <= w["s"] < s + 8)
        out.append({"start": s, "end": e, "title": (text[:55] or f"Moment at {int(p['t'])}s").strip(),
                    "hook_text": " ".join(text.split()[:6]).upper(), "payoff": "", "needs": "nothing",
                    "content_type": "signal-peak", "llm_score": p["score"],
                    "why": f"signal peak ({S.describe_peak(sig, p['t'])})", "description": "",
                    "hashtags": []})
    return out


def select(transcript: dict, sig: dict, *, duration: float, profile: str, n: int, length: str,
           llm: str, exclude: list[dict] | None = None) -> list[dict]:
    mn, mx, imn, imx = LENGTHS[length]
    editor, lens = LENS[profile]
    pk = S.peaks(sig["fused"], k=40)
    raw: list[dict] = []
    if llm != "none" and transcript.get("segments"):
        wins = windows(duration)
        per_win = max(3, -(-n * 2 // len(wins)))
        for wi, (ws, we) in enumerate(wins, 1):
            anchors = "\n".join(f"- {p['t']}s ({S.describe_peak(sig, p['t'])})"
                                for p in pk if ws <= p["t"] < we) or "- none"
            already = ""
            if exclude:
                already = "ALREADY CLIPPED (do not overlap):\n" + "\n".join(
                    f"- {c['start']:.0f}-{c['end']:.0f}s {c.get('title', '')}" for c in exclude) + "\n\n"
            prompt = PROMPT.format(editor=editor, n=per_win, mn=mn, mx=mx, imn=imn, imx=imx, lens=lens,
                                   anchors=anchors, already=already, dur=duration,
                                   lang=transcript.get("language", "?"),
                                   transcript=transcript_lines(transcript, ws, we))
            for attempt in range(1, 4):
                try:
                    log(f"LLM ({llm}) window {wi}/{len(wins)} attempt {attempt}…")
                    data = parse_json(call_llm(prompt, llm))
                    raw += data["clips"]
                    log(f"  {len(data['clips'])} picks")
                    break
                except Exception as e:  # noqa: BLE001 — retry, then fall back
                    log(f"  LLM error: {str(e)[:200]}")
                    prompt += "\n\nREMINDER: reply with the JSON object only."
    weight = 0.75 if profile == "podcast" else 0.55
    if not raw:
        log("no LLM picks — using signal-peak candidates")
        raw, weight = heuristic(transcript, sig, length, n), 0.0
    picks = finalize(raw, transcript, sig, duration, length, n, weight)
    if len(picks) < n and weight > 0:  # top up from signal peaks if the LLM under-delivered
        extra = finalize(heuristic(transcript, sig, length, n), transcript, sig, duration, length, n, 0.0)
        for c in extra:
            if len(picks) >= n:
                break
            if all(_overlap(c, k) < 0.3 for k in picks):
                c["content_type"] = "signal-peak (top-up)"
                picks.append(c)
    if exclude:
        picks = [c for c in picks if all(_overlap(c, x) < 0.3 for x in exclude)]
    return picks
