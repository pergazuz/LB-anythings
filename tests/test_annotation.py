"""An Annotation's regions become an Example's ground-truth boxes (pure domain seam)."""

import pytest

from lb_anythings.domain.annotation import Annotation, GroundTruthBox, parse_annotation
from lb_anythings.domain.example import Example
from lb_anythings.domain.geometry import Box
from tests.annotations import CHOICE, RECTANGLE


def test_rectangle_regions_become_normalized_boxes_with_their_label() -> None:
    annotation = parse_annotation({"result": [RECTANGLE]})

    assert annotation.regions[0].box.coordinates() == pytest.approx((0.1, 0.1, 0.6, 0.6))
    assert annotation.regions[0].label == "pipe"


def test_regions_that_are_not_rectangles_are_ignored() -> None:
    annotation = parse_annotation({"result": [CHOICE, RECTANGLE]})

    assert len(annotation.regions) == 1


def test_a_rectangle_without_a_label_keeps_an_empty_label_for_the_target_to_fill() -> None:
    unlabeled = {"type": "rectanglelabels", "value": {"x": 0, "y": 0, "width": 10, "height": 10}}

    assert parse_annotation({"result": [unlabeled]}).regions[0].label == ""


def test_cancelled_and_skipped_annotations_are_marked() -> None:
    assert parse_annotation({"result": [], "was_cancelled": True}).cancelled
    assert parse_annotation({"result": [], "skipped": True}).cancelled
    assert not parse_annotation({"result": []}).cancelled


def test_an_annotation_with_no_result_has_no_regions() -> None:
    assert parse_annotation({}) == Annotation(regions=(), cancelled=False)


def test_an_example_is_positive_when_it_has_at_least_one_box() -> None:
    box = GroundTruthBox(Box(0.1, 0.1, 0.2, 0.2), "pipe")

    assert Example("task7", (box,)).is_positive
    assert not Example("task8", ()).is_positive
