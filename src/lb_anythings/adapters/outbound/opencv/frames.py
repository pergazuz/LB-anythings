"""Reads a video's frames with OpenCV, and writes the picked ones out."""

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

import numpy as np

from lb_anythings.application.ports import Image

logger = logging.getLogger(__name__)


class OpenCvFrameSource:
    """A video opened for mining. Close it, or use it as a context manager."""

    def __init__(self, path: Path) -> None:
        import cv2  # deferred: part of the ML dependency group

        self._cv2 = cv2
        self._capture: Any = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"cannot open video {str(path)!r}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._size = (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def frames(self, stride: int) -> Iterator[tuple[int, Image]]:
        """Read forward, keeping every `stride`-th frame: far faster than seeking to each."""
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
        index = 0
        while True:
            read, pixels = self._capture.read()
            if not read:
                return
            if index % stride == 0:
                yield index, Image(np.asarray(pixels, dtype=np.uint8))
            index += 1

    def frame_at(self, index: int) -> Image | None:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        read, pixels = self._capture.read()
        if not read:
            logger.warning("frame %d could not be read from the video", index)
            return None
        return Image(np.asarray(pixels, dtype=np.uint8))

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "OpenCvFrameSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Pick(Protocol):
    """A frame chosen for labelling: whatever the miner returns, described by what is written."""

    @property
    def index(self) -> int: ...

    @property
    def score(self) -> float: ...

    @property
    def image(self) -> Image: ...


def write_frames(picks: Iterable[Pick], out_dir: Path) -> list[Path]:
    """Write each pick as `hard_<index>_s<score>.jpg`, so a labelled frame is traceable."""
    import cv2  # deferred: part of the ML dependency group

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pick in picks:
        path = out_dir / f"hard_{pick.index:06d}_s{int(pick.score)}.jpg"
        if cv2.imwrite(str(path), pick.image.pixels):
            written.append(path)
        else:
            logger.warning("could not write %s; skipping that frame", path)
    return written
