"""Predictions and Detections: the pure rules behind what Label Studio receives."""

import pytest

from lb_anythings.domain.annotation_target import AnnotationTarget
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.geometry import Box
from lb_anythings.domain.prediction import Prediction

TARGET = AnnotationTarget("label", "image", "image", labels=("car", "truck"))
BOX = Box(0.1, 0.1, 0.6, 0.6)


def test_a_predictions_score_is_the_mean_of_its_regions() -> None:
    prediction = Prediction(
        regions=(Detection(BOX, 0.9, "car"), Detection(BOX, 0.5, "truck")), version="v"
    )

    assert prediction.score == pytest.approx(0.7)


def test_a_prediction_with_no_regions_scores_zero() -> None:
    assert Prediction(regions=(), version="v").score == 0.0


def test_a_detected_label_that_the_project_knows_is_kept() -> None:
    assert TARGET.label_for("truck") == "truck"


def test_an_unknown_detected_label_falls_back_to_the_projects_first_label() -> None:
    assert TARGET.label_for("pipe") == "car"
    assert TARGET.label_for("0") == "car"


def test_a_detection_score_is_a_confidence_between_zero_and_one() -> None:
    with pytest.raises(ValueError):
        Detection(BOX, 1.5, "car")
