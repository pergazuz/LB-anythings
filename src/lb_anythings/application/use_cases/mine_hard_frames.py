"""Mine a video for Hard Frames: the frames worth labelling next."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from lb_anythings.application.ports import Detector, FrameSource, Image
from lb_anythings.domain.hard_frames import (
    HardFrame,
    UncertaintyBand,
    score_frame,
    select_hard_frames,
)

logger = logging.getLogger(__name__)

REPORT_EVERY = 200  # sampled frames


@dataclass(frozen=True)
class MiningParameters:
    stride: int  # sample every Nth frame
    top_n: int  # how many Hard Frames to keep
    gap: int  # fewest frames between two picks
    band: UncertaintyBand

    def __post_init__(self) -> None:
        # The CLI's flags reach here without passing through the settings model's bounds.
        if self.stride < 1:
            raise ValueError(f"the sampling stride must be at least 1, got {self.stride}")
        if self.top_n < 1:
            raise ValueError(f"there is no point mining for {self.top_n} frames")
        if self.gap < 0:
            raise ValueError(f"the gap between picks cannot be negative, got {self.gap}")

    def frames_to_sample(self, frame_count: int) -> int:
        return math.ceil(max(frame_count, 0) / self.stride)


@dataclass(frozen=True)
class MiningProgress:
    """How far a scan has got, for an Operator watching a long one."""

    sampled: int
    to_sample: int
    candidates: int


@dataclass(frozen=True)
class HardFramePick:
    index: int
    score: float
    image: Image


class MineHardFrames:
    def __init__(
        self,
        detector: Detector,
        on_progress: Callable[[MiningProgress], None] = lambda _: None,
    ) -> None:
        self._detector = detector
        self._on_progress = on_progress

    def __call__(self, source: FrameSource, parameters: MiningParameters) -> list[HardFramePick]:
        candidates = self._scan(source, parameters)
        picks = []
        for frame in select_hard_frames(candidates, top_n=parameters.top_n, gap=parameters.gap):
            image = source.frame_at(frame.index)
            if image is None:
                continue  # the frame source says why
            picks.append(HardFramePick(frame.index, frame.score, image))
        return picks

    def _scan(self, source: FrameSource, parameters: MiningParameters) -> list[HardFrame]:
        """Score every sampled frame, keeping only those worth something: a video is long."""
        to_sample = parameters.frames_to_sample(source.frame_count)
        candidates: list[HardFrame] = []
        sampled = 0
        for index, image in source.frames(parameters.stride):
            sampled += 1
            score = score_frame(self._detector.detect(image), parameters.band)
            if score > 0:
                candidates.append(HardFrame(index, score))
            if sampled % REPORT_EVERY == 0:
                self._on_progress(MiningProgress(sampled, to_sample, len(candidates)))
        if sampled % REPORT_EVERY != 0:
            self._on_progress(MiningProgress(sampled, to_sample, len(candidates)))
        return candidates
