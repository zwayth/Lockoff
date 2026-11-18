"""Orchestrator: source → transcript → signals → picks → slices → reframe → render → QA → report.

Each stage caches to the job's _work folder, so a crash or Ctrl+C resumes where it
stopped, and `rerender` rebuilds clips from plan.json after hand edits.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

from . import captions, ingest, layout as L
from . import select as SEL
from . import signals as S
from .reframe import detect as D
from .reframe import pipeline as RP
from .reframe.types import CropPath
from .transcribe import transcribe, words_in
from .util import JOBS, YTDLP, StageError, auto_whisper_model, ff, ffmpeg, cached, encoder, hms, log, probe, read_json, run, slug, write_json

DEFAULTS = {"clips": 5, "length": "auto", "profile": "auto", "llm": "claude", "layout": "auto",
            "whisper": "auto", "lang": None, "credit": None, "captions": "hormozi", "hook": True,
            "device": "auto", "director": "off", "director_model": "sonnet", "director_max": 12}


def job_dir(meta_title: str, source: str, name: str | None) -> Path:
    if name:
        return JOBS / name
    for d in sorted(JOBS.glob("*"), reverse=True):  # resume an unfinished job for the same source
        m = read_json(d / "_work" / "source.json")
        if m and m.get("source") == source:
            return d
    return JOBS / f"{dt.date.today():%Y-%m-%d}-{slug(meta_title, 36)}"


def pick_profile(meta: dict, transcript: dict, forced: str) -> str:
    if forced != "auto":
        return forced
    if any(k in (meta.get("extractor") or "").lower() for k in ("twitch", "kick")):
        return "stream"
    wpm = len(transcript["words"]) / max(1.0, transcript["duration"] / 60)
    return "podcast" if wpm >= 110 else "stream"


# ------------------------------------------------------------------ render one clip
def render_clip(clip: dict, idx: int, meta: dict, job: Path, opts: dict) -> dict:
    work = job / "_work"
    name = f"c{idx:02d}"
    s, e = clip["start"], clip["end"]
    sl = ingest.fetch_slice(meta, work, name, s, e)
    info = probe(sl)
    log(f"clip {idx}: {hms(s)}–{hms(e)} ({e - s:.1f}s) “{clip.get('title', '')}”")

    analysis = D.analyse(sl, info)
    lay, why = L.decide(info, analysis, opts["profile_used"], clip.get("layout") or opts["layout"])
    log(f"  layout: {lay} ({why})")

    words = clip.get("words")
    if words is None:  # first render: take from the master transcript; stored for hand-fixing
        words = [{"w": w["w"], "s": round(w["s"] - s, 3), "e": round(w["e"] - s, 3)}
                 for w in words_in(opts["transcript"]["words"], s, e) if w["s"] - s >= -0.05]
        clip["words"] = words
    ass = work / f"{name}.ass"
    hook = clip.get("hook_text") if opts["hook"] else None
    captions.write_ass(ass, words if opts["captions"] != "off" else [], caption_y=L.CAPTION_Y[lay],
                       style=opts["captions"] if opts["captions"] != "off" else "hormozi",
                       hook=hook, hook_y=L.HOOK_Y[lay], duration=info["duration"])

    crop, face_box, coverage, modes, motion = None, None, None, None, None
    if lay == "speaker":
        director = None
        if opts.get("director") == "on":  # AI director: Claude picks who to frame, multi-person shots only
            from .reframe.director import Director
            director = Director(sl, work / f"{name}.director.json", words,
                                model=opts.get("director_model") or "sonnet",
                                max_calls=int(opts.get("director_max") or 12))
        path, modes = RP.analyze_clip(sl, info, analysis, L.W, L.H, director=director)
        write_json(work / f"{name}.croppath.json", path.to_json())
        text, cw, ch, x0, y0 = RP.sendcmd(path, info["duration"], 30.0)
        (work / f"{name}.cmd").write_text(text, encoding="utf-8")
        crop = {"cmd": f"{name}.cmd", "w": cw, "h": ch, "x0": x0, "y0": y0}
        coverage = RP.face_coverage(path, analysis)
        motion = RP.motion_report(path)
        log(f"  camera: {modes}, {motion['cuts']} cuts, {motion['glitch_jumps']} glitch jumps, "
            f"{motion['reversals_per_min']} wobbles/min, face-in-frame {coverage:.0%}")
    elif lay == "facecam":
        face_box = L.facecam_box(info, analysis)

    fg = L.filtergraph(lay, info, ass=ass.name, crop=crop, face_box=face_box)
    fg += ";[0:a]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,loudnorm=I=-14:TP=-1.5:LRA=11[aud]"
    out = job / f"{idx:02d}-{slug(clip.get('title') or name, 40)}.mp4"
    for old in job.glob(f"{idx:02d}-*.mp4"):
        old.unlink()
    enc = encoder()
    run([*ff("-i", sl.name), "-filter_complex", fg, "-map", "[v]", "-map", "[aud]",
         "-r", "30", *enc, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
         "-t", f"{info['duration']:.3f}", out.resolve()], cwd=work, timeout=3600)

    q = probe(out)
    qa = {"width": q["w"], "height": q["h"], "duration": round(q["duration"], 2),
          "sync_ok": abs(q["v_start"] - q["a_start"]) < 0.001,
          "length_ok": abs(q["duration"] - info["duration"]) < 0.25,
          "face_in_frame": coverage, "faces_seen": analysis["with_face"], "encoder": enc[1],
          "camera": motion}
    qa["pass"] = (q["w"], q["h"]) == (1080, 1920) and qa["sync_ok"] and qa["length_ok"] and \
        (coverage is None or coverage >= 0.85) and (motion is None or motion["glitch_jumps"] == 0)
    sheet = out.with_suffix(".jpg")
    contact_sheet(out, q["duration"], sheet, work)
    log(f"  → {out.name} {q['w']}x{q['h']} {q['duration']:.1f}s via {enc[1]} | QA {'PASS' if qa['pass'] else 'CHECK'}")
    clip.update(file=out.name, preview=sheet.name, layout_used=lay, layout_reason=why, qa=qa, _cut=[s, e])
    return clip


def contact_sheet(video: Path, dur: float, out: Path, work: Path) -> None:
    shots = []
    for i, f in enumerate((0.12, 0.5, 0.88)):
        p = work / f"_sheet{i}.jpg"
        run([ffmpeg(), "-v", "error", "-y", "-ss", f"{dur * f:.2f}", "-i", video, "-frames:v", "1",
             "-vf", "scale=360:640", p])
        shots.append(p)
    run([ffmpeg(), "-v", "error", "-y", "-i", shots[0], "-i", shots[1], "-i", shots[2],
         "-filter_complex", "hstack=inputs=3", out])
    for p in shots:
        p.unlink(missing_ok=True)


# ------------------------------------------------------------------ report
def write_report(job: Path, plan: dict) -> None:
    meta = plan["source"]
    credit = plan["opts"].get("credit") or (f"@{meta['uploader']}" if meta.get("uploader") else None)
    lines = [f"# Clips — {meta.get('title')}", "",
             f"- Source: {meta.get('webpage_url') or meta.get('path')}",
             f"- Profile: {plan['profile']} · LLM: {plan['opts']['llm']} · whisper {plan['transcript_model']}",
             f"- Made: {plan['made']}", ""]
    for i, c in enumerate(plan["clips"], 1):
        qa = c.get("qa", {})
        tags = " ".join(c.get("hashtags") or [])
        lines += [f"## {i}. {c.get('title', '')}", "",
                  f"- File: `{c.get('file')}` · preview `{c.get('preview')}`",
                  f"- Source range: {hms(c['start'])} – {hms(c['end'])} ({c['end'] - c['start']:.1f}s)",
                  f"- Score: {c['score']:.2f} (LLM {c.get('llm_score', 0):.2f}, signal {c.get('signal_score', 0):.2f}) · {c.get('content_type', '')}",
                  f"- Layout: {c.get('layout_used')} — {c.get('layout_reason')}",
                  f"- QA: {'PASS' if qa.get('pass') else 'CHECK'} · face in frame "
                  f"{'n/a' if qa.get('face_in_frame') is None else format(qa['face_in_frame'], '.0%')} · "
                  f"{qa.get('width')}x{qa.get('height')} · sync {'ok' if qa.get('sync_ok') else 'BAD'}",
                  f"- Hook text (0–3s): **{c.get('hook_text', '')}**",
                  f"- Opens with: “{c.get('opens_with', '')}…”",
                  f"- Payoff: {c.get('payoff', '')}", f"- Why: {c.get('why', '')}", "",
                  "Description:", "",
                  f"> {c.get('description', '')}{(' Credit: ' + credit) if credit else ''}", "",
                  f"Hashtags: {tags}", ""]
    lines += ["---", "Uploading stays manual. Only post clips you have the rights to; credit the creator."]
    (job / "clips.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ entry points
def run_job(source: str, **kw) -> Path:
    opts = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    if ingest.is_url(source):
        title = run([YTDLP, "--print", "title", "--no-warnings", "--no-playlist", source],
                    timeout=120, check=False).stdout.strip().splitlines()
        title = title[0] if title else "stream"
    else:
        title = Path(source).stem
    job = job_dir(title, source, kw.get("job"))
    work = job / "_work"
    meta = ingest.resolve(source, work)
    log(f"job: {job}")

    fresh = bool(kw.get("fresh"))
    whisper_model = auto_whisper_model() if opts["whisper"] == "auto" else opts["whisper"]
    tr = cached(work / "transcript.json",
                lambda: transcribe(Path(meta["wav"]), whisper_model, opts["lang"], opts["device"]), fresh)
    log(f"transcript: {len(tr['words'])} words, language {tr['language']}")
    profile = pick_profile(meta, tr, opts["profile"])
    log(f"profile: {profile}")
    sig = cached(work / f"signals-{profile}.json",
                 lambda: S.build(Path(meta["wav"]), tr, meta.get("heatmap"), profile), fresh)
    duration = float(meta.get("duration") or tr["duration"])

    llm_tag = slug(opts["llm"], 20)
    picks_file = work / f"picks-{profile}-{opts['clips']}-{opts['length']}-{llm_tag}.json"
    picks = cached(picks_file, lambda: SEL.select(tr, sig, duration=duration, profile=profile,
                                                  n=opts["clips"], length=opts["length"], llm=opts["llm"]), fresh)
    if not picks:
        raise StageError("no clip candidates survived validation")
    log(f"{len(picks)} clips selected:")
    for i, c in enumerate(picks, 1):
        log(f"  {i}. {hms(c['start'])}–{hms(c['end'])} score {c['score']:.2f} “{c.get('title', '')}”")

    opts.update(profile_used=profile, transcript=tr)
    plan = {"source": meta, "profile": profile, "transcript_model": f"{tr['model']}/{tr['device']}",
            "made": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
            "opts": {k: v for k, v in opts.items() if k not in ("transcript",)}, "clips": []}
    for i, c in enumerate(picks, 1):
        try:
            plan["clips"].append(render_clip(dict(c), i, meta, job, opts))
        except StageError as e:
            log(f"clip {i} FAILED: {e}")
            plan["clips"].append({**c, "error": str(e)[:500], "qa": {"pass": False}})
        write_json(job / "plan.json", plan)
    write_report(job, plan)
    ok = sum(1 for c in plan["clips"] if c.get("qa", {}).get("pass"))
    log(f"done: {ok}/{len(plan['clips'])} clips passed QA → {job}")
    return job


def rerender(job: Path, only: list[int] | None = None, layout: str | None = None,
             director: str | None = None) -> None:
    plan = read_json(job / "plan.json")
    if not plan:
        raise StageError(f"no plan.json in {job}")
    meta = plan["source"]
    opts = {**DEFAULTS, **plan["opts"]}
    tr = read_json(job / "_work" / "transcript.json")
    opts.update(profile_used=plan["profile"], transcript=tr)
    if layout:
        opts["layout"] = layout
    if director:
        opts["director"] = director
        plan["opts"]["director"] = director
    for i, c in enumerate(plan["clips"], 1):
        if only and i not in only:
            continue
        old = job / "_work" / f"c{i:02d}_slice.mp4"
        if old.exists() and c.get("_cut") not in (None, [c["start"], c["end"]]):
            old.unlink()  # in/out points were edited → re-cut and re-take caption words
            c.pop("words", None)
        if layout:
            c.pop("layout", None)
        plan["clips"][i - 1] = render_clip(c, i, meta, job, opts)
    write_json(job / "plan.json", plan)
    write_report(job, plan)


def clean(job: Path) -> None:
    shutil.rmtree(job / "_work", ignore_errors=True)
    log(f"removed working files; kept {', '.join(sorted(p.name for p in job.iterdir()))}")
