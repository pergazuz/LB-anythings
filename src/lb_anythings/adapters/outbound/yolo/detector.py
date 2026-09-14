"""The ultralytics YOLO Detector. Heavy imports happen only when a Checkpoint is loaded."""

from collections.abc import Sequence
from typing import Any

from lb_anythings.application.ports import Image
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.geometry import Box


class YoloDetector:
    def __init__(self, model: Any, version: str, *, conf: float, imgsz: int) -> None:
        self._model = model
        self.version = version
        self._conf = conf
        self._imgsz = imgsz

    def detect(self, image: Image) -> Sequence[Detection]:
        result = self._model.predict(
            image.pixels, conf=self._conf, imgsz=self._imgsz, verbose=False
        )[0]
        height, width = result.orig_shape
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes, strict=True):
            box = Box.from_pixels(
                float(x1), float(y1), float(x2), float(y2), width=int(width), height=int(height)
            )
            detections.append(Detection(box, float(score), str(names[int(cls)])))
        return detections


class YoloDetectorFactory:
    def __init__(self, *, conf: float, imgsz: int) -> None:
        self._conf = conf
        self._imgsz = imgsz

    def load(self, checkpoint: Checkpoint) -> YoloDetector:
        from ultralytics import YOLO  # deferred: importing torch takes seconds and a GPU context

        return YoloDetector(
            YOLO(str(checkpoint.path)), checkpoint.version, conf=self._conf, imgsz=self._imgsz
        )
