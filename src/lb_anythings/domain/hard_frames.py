"""Hard Frames: which frames of a video the Detector is least sure about.

A Detector that is confident, or confidently silent, teaches an Annotator little. The frames
worth labelling next are the ones where it hesitates, and most of all where it hesitates over
something small: distant objects are the known weak spot of a detector trained on near ones.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from lb_anythings.domain.detection import Detection

SMALL_AREA = 0.02  # a box under 2% of the frame
MIDDLING_AREA = 0.05  # a box under 5% of the frame


@dataclass(frozen=True)
class UncertaintyBand:
    """The confidence range in which a Detector is neither sure nor dismissive."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError(f"an uncertainty band must be ordered within [0, 1], got {self}")

    def holds(self, score: float) -> bool:
        return self.low <= score < self.high


@dataclass(frozen=True)
class HardFrame:
    """One frame of a video and how much labelling it is worth."""

    index: int
    score: float


def score_frame(detections: Sequence[Detection], band: UncertaintyBand) -> float:
    """How much this frame is worth labelling: one per uncertain box, more when it is small."""
    return sum(1.0 + _size_weight(d) for d in detections if band.holds(d.score))


def _size_weight(detection: Detection) -> float:
    area = detection.box.area
    if area < SMALL_AREA:
        return 2.0
    if area < MIDDLING_AREA:
        return 1.0
    return 0.0


def select_hard_frames(candidates: Sequence[HardFrame], *, top_n: int, gap: int) -> list[HardFrame]:
    """The hardest frames, kept at least `gap` frames apart so near-duplicates are not labelled.

    Ties go to the earlier frame, and the picks come back in time order, which is how an
    Annotator reads a video.
    """
    picks: list[HardFrame] = []
    for frame in sorted(candidates, key=lambda f: (-f.score, f.index)):
        if len(picks) >= top_n:
            break
        if frame.score <= 0:
            continue
        if all(abs(frame.index - picked.index) >= gap for picked in picks):
            picks.append(frame)
    return sorted(picks, key=lambda f: f.index)
