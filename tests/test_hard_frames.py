"""Hard Frame scoring and selection: the pure rules behind what to label next."""

import pytest

from lb_anythings.domain.detection import Detection
from lb_anythings.domain.geometry import Box
from lb_anythings.domain.hard_frames import (
    HardFrame,
    UncertaintyBand,
    score_frame,
    select_hard_frames,
)

BAND = UncertaintyBand(0.25, 0.55)


def _detection(score: float, area: float) -> Detection:
    """A box of the given fraction of the frame, as a square-ish strip from the corner."""
    return Detection(Box(0.0, 0.0, area, 1.0), score, "pipe")


def test_a_frame_the_detector_is_sure_about_scores_nothing() -> None:
    confident = [_detection(0.9, area=0.01), _detection(0.05, area=0.01)]

    assert score_frame(confident, BAND) == 0.0


def test_a_large_uncertain_box_counts_once() -> None:
    assert score_frame([_detection(0.4, area=0.30)], BAND) == 1.0


def test_a_middling_uncertain_box_counts_twice() -> None:
    assert score_frame([_detection(0.4, area=0.04)], BAND) == 2.0


def test_a_small_uncertain_box_counts_three_times() -> None:
    # small and distant is the known weak spot, so it is worth the most labelling attention
    assert score_frame([_detection(0.4, area=0.01)], BAND) == 3.0


def test_the_band_includes_its_floor_and_excludes_its_ceiling() -> None:
    assert score_frame([_detection(0.25, area=0.30)], BAND) == 1.0
    assert score_frame([_detection(0.55, area=0.30)], BAND) == 0.0


def test_a_frames_score_is_the_sum_over_its_uncertain_boxes() -> None:
    detections = [
        _detection(0.3, area=0.01),
        _detection(0.5, area=0.30),
        _detection(0.9, area=0.01),
    ]

    assert score_frame(detections, BAND) == 4.0


def test_a_frame_with_no_detections_scores_nothing() -> None:
    assert score_frame([], BAND) == 0.0


def test_a_band_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="uncertainty band"):
        UncertaintyBand(0.6, 0.4)


def test_the_hardest_frames_win_but_stay_apart_and_come_back_in_time_order() -> None:
    scored = [
        HardFrame(0, 5.0),
        HardFrame(10, 4.0),  # within the gap of the hardest frame
        HardFrame(100, 3.0),
        HardFrame(200, 1.0),
    ]

    picks = select_hard_frames(scored, top_n=2, gap=60)

    assert [(p.index, p.score) for p in picks] == [(0, 5.0), (100, 3.0)]


def test_frames_scoring_nothing_are_never_picked() -> None:
    picks = select_hard_frames([HardFrame(0, 0.0), HardFrame(500, 2.0)], top_n=5, gap=60)

    assert [p.index for p in picks] == [500]


def test_asking_for_more_than_the_gap_allows_returns_what_fits() -> None:
    scored = [HardFrame(index, 1.0) for index in (0, 10, 20, 30)]

    picks = select_hard_frames(scored, top_n=10, gap=60)

    assert [p.index for p in picks] == [0]


def test_nothing_scored_means_nothing_picked() -> None:
    assert select_hard_frames([], top_n=10, gap=60) == []
