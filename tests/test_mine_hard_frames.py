"""Mining a video for the frames the Detector is least sure about."""

from collections.abc import Iterator, Sequence
from dataclasses import replace

import numpy as np
import pytest

from lb_anythings.application.ports import Image
from lb_anythings.application.use_cases.mine_hard_frames import (
    MineHardFrames,
    MiningParameters,
    MiningProgress,
)
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.geometry import Box
from lb_anythings.domain.hard_frames import UncertaintyBand
from tests.fakes import ScriptedDetector

PARAMETERS = MiningParameters(stride=10, top_n=2, gap=50, band=UncertaintyBand(0.25, 0.55))
UNCERTAIN = Detection(Box(0.0, 0.0, 0.1, 0.1), 0.4, "pipe")  # 1% of the frame: worth 3
CONFIDENT = Detection(Box(0.0, 0.0, 0.1, 0.1), 0.95, "pipe")
WIDTH, HEIGHT = 200, 100


def _frame(index: int) -> Image:
    """A frame that carries its own index in its first pixel, so any order of reads is fine."""
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[0, 0, 0], pixels[0, 0, 1] = divmod(index, 256)
    return Image(pixels)


class FakeFrameSource:
    """A video whose frames show `hard[index]` uncertain boxes and one confident box."""

    def __init__(self, frame_count: int, hard: dict[int, int]) -> None:
        self.frame_count = frame_count
        self.size = (WIDTH, HEIGHT)
        self.hard = hard
        self.reads: list[int] = []

    def frames(self, stride: int) -> Iterator[tuple[int, Image]]:
        for index in range(0, self.frame_count, stride):
            yield index, _frame(index)

    def frame_at(self, index: int) -> Image | None:
        self.reads.append(index)
        return _frame(index)


class PerFrameDetector:
    """Detects what the source scripts for the frame it is handed."""

    version = "test"

    def __init__(self, source: FakeFrameSource) -> None:
        self._source = source

    def detect(self, image: Image) -> Sequence[Detection]:
        index = int(image.pixels[0, 0, 0]) * 256 + int(image.pixels[0, 0, 1])
        return [UNCERTAIN] * self._source.hard.get(index, 0) + [CONFIDENT]


@pytest.fixture
def source() -> FakeFrameSource:
    return FakeFrameSource(frame_count=300, hard={0: 1, 10: 3, 120: 2, 200: 1})


def test_the_hardest_frames_come_back_with_their_images(source: FakeFrameSource) -> None:
    picks = MineHardFrames(PerFrameDetector(source))(source, PARAMETERS)

    assert [(p.index, p.score) for p in picks] == [(10, 9.0), (120, 6.0)]
    assert all(p.image.width == WIDTH and p.image.height == HEIGHT for p in picks)


def test_the_picked_frames_are_re_read_from_the_video(source: FakeFrameSource) -> None:
    picks = MineHardFrames(PerFrameDetector(source))(source, PARAMETERS)

    assert sorted(source.reads) == sorted(pick.index for pick in picks)


def test_a_video_the_detector_is_sure_about_yields_nothing() -> None:
    easy = FakeFrameSource(frame_count=100, hard={})

    assert MineHardFrames(PerFrameDetector(easy))(easy, PARAMETERS) == []


def test_progress_counts_the_frames_that_will_be_scored_not_the_whole_video(
    source: FakeFrameSource,
) -> None:
    reported: list[MiningProgress] = []

    miner = MineHardFrames(PerFrameDetector(source), on_progress=reported.append)
    miner(source, PARAMETERS)

    assert reported, "a long scan must show it is alive"
    # 300 frames at a stride of 10 means 30 to score, not 300
    assert reported[-1] == MiningProgress(sampled=30, to_sample=30, candidates=4)


def test_progress_is_reported_once_per_batch_and_once_at_the_end() -> None:
    long_video = FakeFrameSource(frame_count=4000, hard={})
    reported: list[MiningProgress] = []

    miner = MineHardFrames(PerFrameDetector(long_video), on_progress=reported.append)
    miner(long_video, replace(PARAMETERS, stride=1))

    # every 200 of the 4000, and no repeat of the last batch as a final line
    assert [p.sampled for p in reported] == list(range(200, 4001, 200))


def test_a_detector_serving_no_checkpoint_finds_nothing(source: FakeFrameSource) -> None:
    assert MineHardFrames(ScriptedDetector("none", []))(source, PARAMETERS) == []


def test_a_stride_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="stride must be at least 1"):
        replace(PARAMETERS, stride=0)


def test_mining_for_no_frames_is_refused() -> None:
    with pytest.raises(ValueError, match="no point mining"):
        replace(PARAMETERS, top_n=0)


def test_a_negative_gap_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(PARAMETERS, gap=-1)
