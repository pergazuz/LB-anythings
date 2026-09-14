"""The real YOLO Detector against a real Checkpoint. Runs only when both are provided."""

import os
from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.yolo.detector import YoloDetectorFactory
from lb_anythings.application.ports import Image
from lb_anythings.domain.checkpoint import Checkpoint

CHECKPOINT = os.environ.get("LB_TEST_CHECKPOINT")
IMAGE = os.environ.get("LB_TEST_IMAGE")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (CHECKPOINT and IMAGE),
        reason="set LB_TEST_CHECKPOINT and LB_TEST_IMAGE to run the real Detector",
    ),
]


def test_the_checkpoint_detects_its_class_on_a_real_frame() -> None:
    cv2 = pytest.importorskip("cv2")
    assert CHECKPOINT and IMAGE
    path = Path(CHECKPOINT)
    detector = YoloDetectorFactory(conf=0.25, imgsz=1024).load(
        Checkpoint(path, path.stat().st_mtime)
    )

    detections = detector.detect(Image(cv2.imread(IMAGE)))

    assert detector.version.startswith(path.name + "@")
    assert detections, "expected at least one Detection on the reference frame"
    assert all(d.label for d in detections), "labels come from the Checkpoint's class names"
    assert all(0.25 <= d.score <= 1.0 for d in detections)
    assert all(d.box.x2 > d.box.x1 and d.box.y2 > d.box.y1 for d in detections)
