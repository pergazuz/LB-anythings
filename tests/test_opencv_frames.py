"""Contract of the OpenCV frame source, against a video generated in the test."""

from pathlib import Path

import numpy as np
import pytest

from lb_anythings.adapters.outbound.opencv.frames import OpenCvFrameSource, write_frames
from lb_anythings.application.ports import Image

WIDTH, HEIGHT, FRAMES = 32, 24, 30


@pytest.fixture
def video(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2", reason="reading video needs the ML dependency group")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), "could not open a writer for the test clip"
    for index in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), index * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_a_video_reports_its_length_and_size(video: Path) -> None:
    with OpenCvFrameSource(video) as source:
        assert source.frame_count == FRAMES
        assert source.size == (WIDTH, HEIGHT)


def test_frames_come_back_at_the_stride_with_their_indices(video: Path) -> None:
    with OpenCvFrameSource(video) as source:
        sampled = list(source.frames(stride=10))

    assert [index for index, _ in sampled] == [0, 10, 20]
    assert all(image.width == WIDTH and image.height == HEIGHT for _, image in sampled)


def test_every_frame_is_read_at_a_stride_of_one(video: Path) -> None:
    with OpenCvFrameSource(video) as source:
        assert len(list(source.frames(stride=1))) == FRAMES


def test_one_frame_can_be_read_by_index(video: Path) -> None:
    with OpenCvFrameSource(video) as source:
        image = source.frame_at(20)

    assert image is not None
    assert (image.width, image.height) == (WIDTH, HEIGHT)


def test_a_frame_past_the_end_is_not_read(video: Path) -> None:
    with OpenCvFrameSource(video) as source:
        assert source.frame_at(FRAMES + 100) is None


def test_a_video_that_cannot_be_opened_says_so(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason="reading video needs the ML dependency group")

    with pytest.raises(FileNotFoundError, match="cannot open"):
        OpenCvFrameSource(tmp_path / "missing.avi")


class Pick:
    def __init__(self, index: int, score: float, image: Image) -> None:
        self.index = index
        self.score = score
        self.image = image


def test_picked_frames_are_written_named_by_their_index_and_score(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2", reason="writing JPEG needs the ML dependency group")
    picks = [
        Pick(10, 3.0, Image(np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8))),
        Pick(4875, 5.6, Image(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))),
    ]

    written = write_frames(picks, tmp_path / "hard_frames")

    assert [path.name for path in written] == ["hard_000010_s3.jpg", "hard_004875_s5.jpg"]
    assert all(cv2.imread(str(path)) is not None for path in written)


def test_writing_creates_the_output_folder(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason="writing JPEG needs the ML dependency group")
    out = tmp_path / "does" / "not" / "exist"

    write_frames([Pick(1, 1.0, Image(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)))], out)

    assert out.is_dir()


def test_writing_nothing_writes_nothing(tmp_path: Path) -> None:
    assert write_frames([], tmp_path / "hard_frames") == []
