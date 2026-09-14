"""Contract of the filesystem Example Store: the YOLO layout the previous backend also wrote."""

from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.filesystem.examples import FilesystemExampleStore
from lb_anythings.domain.annotation import GroundTruthBox
from lb_anythings.domain.example import Example
from lb_anythings.domain.geometry import Box
from tests.fakes import image

PIPE_BOX = GroundTruthBox(Box(0.1, 0.1, 0.6, 0.6), "pipe")
CAR_BOX = GroundTruthBox(Box(0.0, 0.0, 0.5, 0.25), "car")
ONE_BOX = "0 0.500000 0.500000 0.100000 0.100000"


@pytest.fixture
def store(tmp_path: Path) -> FilesystemExampleStore:
    pytest.importorskip("cv2", reason="saving encodes JPEG with the ML dependency group")
    return FilesystemExampleStore(tmp_path / "examples")


def _legacy_layout(root: Path, labels: dict[str, str], classes: str | None = "pipe\n") -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for stem, text in labels.items():
        (root / "images" / f"{stem}.jpg").write_bytes(b"jpeg bytes")
        (root / "labels" / f"{stem}.txt").write_text(text)
    if classes is not None:
        (root / "classes.txt").write_text(classes)


def test_saving_writes_the_image_the_yolo_lines_and_the_classes_file(
    store: FilesystemExampleStore, tmp_path: Path
) -> None:
    store.save(Example("7", (PIPE_BOX,)), image(200, 100))

    root = tmp_path / "examples"
    assert (root / "images/task7.jpg").stat().st_size > 0
    assert (root / "labels/task7.txt").read_text() == "0 0.350000 0.350000 0.500000 0.500000\n"
    assert (root / "classes.txt").read_text() == "pipe\n"


def test_new_labels_are_appended_to_the_classes_file_in_first_seen_order(
    store: FilesystemExampleStore, tmp_path: Path
) -> None:
    store.save(Example("7", (PIPE_BOX,)), image(200, 100))
    store.save(Example("8", (CAR_BOX, PIPE_BOX)), image(200, 100))

    assert (tmp_path / "examples/classes.txt").read_text() == "pipe\ncar\n"
    lines = (tmp_path / "examples/labels/task8.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["1", "0"]


def test_saving_the_same_task_again_replaces_both_files(
    store: FilesystemExampleStore, tmp_path: Path
) -> None:
    image_file = tmp_path / "examples/images/task7.jpg"
    store.save(Example("7", (PIPE_BOX, PIPE_BOX)), image(200, 100))
    first_image = image_file.read_bytes()

    store.save(Example("7", (PIPE_BOX,)), image(100, 50))

    [example] = store.all()
    assert (example.task_id, len(example.boxes)) == ("7", 1)
    assert image_file.read_bytes() != first_image


def test_a_negative_example_is_stored_but_not_counted(store: FilesystemExampleStore) -> None:
    store.save(Example("9", ()), image(200, 100))
    store.save(Example("7", (PIPE_BOX,)), image(200, 100))

    assert store.positive_count() == 1
    assert {e.task_id: e.is_positive for e in store.all()} == {"7": True, "9": False}


def test_an_empty_store_has_nothing(tmp_path: Path) -> None:
    store = FilesystemExampleStore(tmp_path / "examples")

    assert store.positive_count() == 0
    assert list(store.all()) == []
    assert store.training_files() == []
    assert store.class_names() == []


def test_the_previous_backends_layout_is_read_without_conversion(tmp_path: Path) -> None:
    root = tmp_path / "examples"
    _legacy_layout(root, {"task10": "0 0.350000 0.350000 0.500000 0.500000", "task11": ""})
    store = FilesystemExampleStore(root)

    assert store.positive_count() == 1
    examples = {e.task_id: e for e in store.all()}
    assert examples["10"].boxes[0].label == "pipe"
    assert examples["10"].boxes[0].box.coordinates() == pytest.approx((0.1, 0.1, 0.6, 0.6))
    assert examples["11"].boxes == ()


def test_the_store_hands_the_trainer_its_positive_files_and_class_names(tmp_path: Path) -> None:
    root = tmp_path / "examples"
    _legacy_layout(root, {"task1": ONE_BOX + "\n", "task2": "", "task3": ONE_BOX}, "pipe\ncar\n")
    (root / "labels/task4.txt").write_text(ONE_BOX)  # positive, but its image is gone
    store = FilesystemExampleStore(root)

    files = store.training_files()

    assert [(task_id, image.name, label.name) for task_id, image, label in files] == [
        ("1", "task1.jpg", "task1.txt"),
        ("3", "task3.jpg", "task3.txt"),
    ]
    assert store.class_names() == ["pipe", "car"]
    assert store.positive_count() == 2
