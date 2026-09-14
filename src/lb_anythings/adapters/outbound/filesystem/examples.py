"""The Example Store on disk, in the YOLO layout (ADR 0003).

    <root>/images/task<id>.jpg      the image
    <root>/labels/task<id>.txt      one `class cx cy w h` line per box, normalized
    <root>/classes.txt              label names, one per line, in first-seen order

The `task<id>` stem is the previous backend's naming, so its Examples are readable as-is.
Writes are atomic per file and serialized, since webhooks arrive on parallel threads.
"""

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lb_anythings.application.ports import Image
from lb_anythings.domain.annotation import GroundTruthBox
from lb_anythings.domain.example import Example
from lb_anythings.domain.geometry import Box

STEM_PREFIX = "task"
ENCODING = "utf-8"


class FilesystemExampleStore:
    def __init__(self, root: Path) -> None:
        self._images = root / "images"
        self._labels = root / "labels"
        self._classes = root / "classes.txt"
        self._lock = threading.Lock()

    def save(self, example: Example, image: Image) -> None:
        jpeg = _encode_jpeg(image)  # outside the lock: the slow part
        with self._lock:
            self._images.mkdir(parents=True, exist_ok=True)
            self._labels.mkdir(parents=True, exist_ok=True)
            classes = self._read_classes()
            for region in example.boxes:
                if region.label not in classes:
                    classes.append(region.label)
            _write_atomically(
                self._classes, "".join(f"{name}\n" for name in classes).encode(ENCODING)
            )

            stem = f"{STEM_PREFIX}{example.task_id}"
            _write_atomically(self._images / f"{stem}.jpg", jpeg)
            lines = [
                "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    classes.index(region.label), *region.box.to_yolo()
                )
                for region in example.boxes
            ]
            label_text = "".join(f"{line}\n" for line in lines)
            _write_atomically(self._labels / f"{stem}.txt", label_text.encode(ENCODING))

    def positive_count(self) -> int:
        return len(self.training_files())

    def training_files(self) -> list[tuple[str, Path, Path]]:
        """(task id, image file, label file) for every positive Example with its image present.

        The trainer copies these into its own layout; only this adapter reads the store's.
        """
        if not self._labels.is_dir():
            return []
        files = []
        for label in sorted(self._labels.glob("*.txt")):
            image = self._images / f"{label.stem}.jpg"
            if label.read_text(encoding=ENCODING).strip() and image.is_file():
                files.append((label.stem.removeprefix(STEM_PREFIX), image, label))
        return files

    def class_names(self) -> list[str]:
        """Label names by class index, as the classes file records them."""
        return self._read_classes()

    def all(self) -> Sequence[Example]:
        if not self._labels.is_dir():
            return []
        classes = self._read_classes()
        return [self._read_example(path, classes) for path in sorted(self._labels.glob("*.txt"))]

    def _read_classes(self) -> list[str]:
        if not self._classes.is_file():
            return []
        lines = self._classes.read_text(encoding=ENCODING).splitlines()
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def _read_example(label_file: Path, classes: list[str]) -> Example:
        boxes: list[GroundTruthBox] = []
        for line in label_file.read_text(encoding=ENCODING).splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            index = int(parts[0])
            label = classes[index] if index < len(classes) else str(index)
            boxes.append(GroundTruthBox(Box.from_yolo(*map(float, parts[1:])), label))
        return Example(label_file.stem.removeprefix(STEM_PREFIX), tuple(boxes))


def _write_atomically(path: Path, data: bytes) -> None:
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    partial.write_bytes(data)
    os.replace(partial, path)


def _encode_jpeg(image: Image) -> bytes:
    import cv2  # deferred: part of the ML dependency group

    ok, encoded = cv2.imencode(".jpg", image.pixels)
    if not ok:
        raise ValueError("could not encode the image as JPEG")
    return bytes(np.asarray(encoded).tobytes())
