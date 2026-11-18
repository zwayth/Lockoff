"""Crop-path contract — the one artifact every stage shares.

The analysis emits a CropPath; the editor previews it; the renderer consumes
it. All coordinates are SOURCE pixels; `t` is seconds relative to the
trimmed clip start (analysis runs on the in→out segment, and the render
applies -ss before the filter, so filter time matches).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CropKeyframe:
    t: float   # seconds from clip start
    x: float   # crop window left edge, source px
    y: float   # crop window top edge, source px
    w: float   # crop window width, source px
    h: float   # crop window height, source px


@dataclass
class CropPath:
    source_w: int
    source_h: int
    target_w: int          # output canvas (e.g. 1080x1920)
    target_h: int
    keyframes: list[CropKeyframe] = field(default_factory=list)
    mode: str = "subject"  # subject | saliency | center — what produced it
    version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "source": {"w": self.source_w, "h": self.source_h},
            "target": {"w": self.target_w, "h": self.target_h},
            "keyframes": [
                {"t": round(k.t, 3), "x": round(k.x, 1), "y": round(k.y, 1),
                 "w": round(k.w, 1), "h": round(k.h, 1)}
                for k in self.keyframes
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CropPath":
        return cls(
            source_w=int(data["source"]["w"]),
            source_h=int(data["source"]["h"]),
            target_w=int(data["target"]["w"]),
            target_h=int(data["target"]["h"]),
            keyframes=[CropKeyframe(float(k["t"]), float(k["x"]), float(k["y"]),
                                    float(k["w"]), float(k["h"]))
                       for k in data["keyframes"]],
            mode=data.get("mode", "subject"),
            version=int(data.get("version", SCHEMA_VERSION)),
        )


@dataclass
class ReframeResult:
    """What the pipeline hands back — path + how it was obtained, per shot."""
    path: CropPath
    shot_modes: list[str] = field(default_factory=list)  # per-shot: subject|saliency|center
    # All tracked person samples — powers the debug overlay (see the spike's
    # --debug flag) so a bad framing decision can be SEEN, not guessed at.
    debug_samples: list["BoxSample"] = field(default_factory=list)

    @property
    def mode_summary(self) -> str:
        if not self.shot_modes:
            return self.path.mode
        # Most informative mode wins the label: any subject > any saliency > centre.
        if "subject" in self.shot_modes:
            return "subject"
        if "saliency" in self.shot_modes:
            return "saliency"
        return "center"


# ── Analysis-side sample types (detector output → ranking/path input) ──

@dataclass(frozen=True)
class BoxSample:
    """One tracked box at one sampled timestamp."""
    t: float
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    has_face: bool = False
    # Actual face centre when a face was matched to this person box — the
    # framing target should be the FACE, not the torso (box centre), or a
    # close-up subject's face drifts out of the crop.
    face_cx: float | None = None
    face_cy: float | None = None
    # Lip-region motion vs this track's previous sample (0..~0.3). Talking
    # mouths move — the no-GPU speaker signal (ASD-lite).
    mouth_act: float | None = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Shot:
    start: float  # seconds, inclusive
    end: float    # seconds, exclusive
