"""Fast, model-free tests of the decision layers (run: .venv\\Scripts\\python -m pytest -q tests)."""
import numpy as np

from lockoff import select as SEL
from lockoff import signals as S
from lockoff.ingest import align_offset
from lockoff.reframe import pipeline as RP
from lockoff.reframe.types import BoxSample


def _words(text, t0=0.0, step=0.4):
    out, t = [], t0
    for w in text.split():
        out.append({"w": w, "s": round(t, 2), "e": round(t + step - 0.05, 2)})
        t += step
    return out


def test_parse_json_tolerates_fences_and_trailing_commas():
    d = SEL.parse_json('Sure!\n```json\n{"clips": [{"start": 1, "end": 30,},]}\n```')
    assert d["clips"][0]["end"] == 30


def test_snap_moves_off_backreference_and_onto_sentence_end():
    text = ("Here is the setup you need. So that is why it failed badly. " +
            "We rebuilt everything from scratch in a week and it worked. " * 4 +
            "Next topic starts now and goes on for a while. " * 6)
    words = _words(text)
    so_idx = next(i for i, w in enumerate(words) if w["w"] == "So")
    clip = {"start": words[so_idx]["s"], "end": words[so_idx]["s"] + 22}
    out = SEL.snap(clip, words, duration=words[-1]["e"], mn=15, mx=60)
    assert out is not None
    assert out["opens_with"].startswith("Here")  # walked back past "So"
    end_word = next(w for w in words if abs(w["e"] + 0.35 - out["end"]) < 0.02)
    assert end_word["w"].endswith(".")


def test_finalize_dedupes_overlaps_and_respects_count():
    words = _words(("This is a complete sentence about a thing. " * 40).strip())
    tr = {"words": words}
    sig = {"fused": [0.1] * 200}
    raw = [{"start": 10, "end": 35, "scores": {"hook": 9, "standalone": 9, "emotion": 9, "shareability": 9}},
           {"start": 12, "end": 37, "scores": {"hook": 5, "standalone": 5, "emotion": 5, "shareability": 5}},
           {"start": 60, "end": 85, "scores": {"hook": 7, "standalone": 7, "emotion": 7, "shareability": 7}}]
    picks = SEL.finalize(raw, tr, sig, duration=words[-1]["e"], length="auto", n=5, llm_weight=0.8)
    assert len(picks) == 2
    assert picks[0]["llm_score"] > picks[1]["llm_score"]


def test_peaks_nms():
    x = np.zeros(300)
    x[50], x[55], x[200] = 1.0, 0.9, 0.8
    p = S.peaks(x.tolist(), k=5, min_gap=30)
    ts = [q["t"] for q in p]
    assert 50 in ts and 200 in ts and 55 not in ts


def test_align_offset_recovers_shift():
    rng = np.random.default_rng(0)
    master = (rng.standard_normal(16000 * 120) * np.repeat(rng.random(120 * 10), 1600)).astype(np.float32)
    true_start = 47.3
    piece = master[int(true_start * 16000):int((true_start + 30) * 16000)]
    got = align_offset(piece, master, guess=44.0)
    assert abs(got - true_start) < 0.05


def _speaker_boxes(dur=10.0, fps=10):
    """Two faces left/right; the right one talks (mouth moves) in the 2nd half."""
    out = []
    for i in range(int(dur * fps)):
        t = i / fps
        for tid, cx in ((0, 500), (1, 1450)):
            talking = (tid == 0) == (t < 5)
            out.append(BoxSample(t=t, track_id=tid, x1=cx - 150, y1=200, x2=cx + 150, y2=1000,
                                 has_face=True, face_cx=cx, face_cy=350,
                                 mouth_act=0.09 if talking else 0.005))
    return out


def test_reframe_follows_active_speaker(monkeypatch):
    from lockoff.reframe import detect as D
    from lockoff.reframe.types import Shot
    monkeypatch.setattr(D, "detect_shots", lambda v, d: [Shot(0.0, d)])
    info = {"w": 1920, "h": 1080, "duration": 10.0, "fps": 30.0}
    analysis = {"boxes": _speaker_boxes(), "sampled": {"times": [], "faces": []}}
    path, modes = RP.analyze_clip(None, info, analysis, 1080, 1920)
    x1, _, w, _ = path.at(2.0)
    x2, _, _, _ = path.at(8.5)
    assert x1 <= 500 <= x1 + w, "first half should frame the left speaker"
    assert x2 <= 1450 <= x2 + w, "second half should frame the right speaker"
    assert len(path.cuts) == 1, "one clean cut at the speaker switch, nothing else"
    assert RP.motion_report(path)["glitch_jumps"] == 0
    text, cw, ch, *_ = RP.sendcmd(path, 10.0, 30.0)
    assert cw == 608 and ch == 1080 and "crop@rf x" in text


def test_director_pick_steers_camera_and_single_speaker_costs_nothing(monkeypatch):
    from lockoff.reframe import detect as D
    from lockoff.reframe.director import significant_tracks
    from lockoff.reframe.types import Shot
    monkeypatch.setattr(D, "detect_shots", lambda v, d: [Shot(0.0, d)])
    info = {"w": 1920, "h": 1080, "duration": 10.0, "fps": 30.0}
    boxes = _speaker_boxes()  # left person talks in the first half

    class FakeDirector:  # the "AI" says: follow the RIGHT person the whole time
        calls, cache = 0, {}

        def decide(self, shot, by):
            assert len(significant_tracks(by, shot)) == 2
            self.calls += 1
            return [(0.0, 10.0, 1, False)]

    d = FakeDirector()
    path, modes = RP.analyze_clip(None, info, {"boxes": boxes, "sampled": {"times": [], "faces": []}},
                                  1080, 1920, director=d)
    x, _, w, _ = path.at(2.0)
    assert x <= 1450 <= x + w, "director's pick wins over lip motion"
    assert d.calls == 1 and "director" in modes[0]
    single = [b for b in boxes if b.track_id == 0]
    by = {0: {round(b.t, 3): b for b in single}}
    assert significant_tracks(by, Shot(0.0, 10.0)) == [], "one person → no Claude call"


def test_camera_is_calm_with_noisy_detections(monkeypatch):
    """Speaker sways ±40 px with detector noise ±25 px, then walks 500 px.
    The camera must hold through the sway and do ONE eased pan for the walk."""
    from lockoff.reframe import detect as D
    from lockoff.reframe.types import Shot
    monkeypatch.setattr(D, "detect_shots", lambda v, d: [Shot(0.0, d)])
    rng = np.random.default_rng(1)
    boxes = []
    for i in range(200):  # 20 s at 10 fps
        t = i / 10
        walk = 0 if t < 10 else min(500, (t - 10) * 250)
        cx = 700 + walk + 40 * np.sin(t * 2) + rng.normal(0, 25)
        boxes.append(BoxSample(t=t, track_id=0, x1=cx - 150, y1=200, x2=cx + 150, y2=1000, has_face=True,
                               face_cx=cx, face_cy=350, mouth_act=0.08))
    info = {"w": 1920, "h": 1080, "duration": 20.0, "fps": 30.0}
    path, _ = RP.analyze_clip(None, info, {"boxes": boxes, "sampled": {"times": [], "faces": []}}, 1080, 1920)
    m = RP.motion_report(path)
    assert m["glitch_jumps"] == 0 and m["cuts"] == 0
    assert m["reversals_per_min"] <= 6, m           # no wobble
    assert np.ptp(path.xs[:270]) < 15, "should hold still through the sway (first 9 s)"
    assert path.at(18)[0] <= 1200 <= path.at(18)[0] + path.crop_w, "followed the walk"
