"""Contract of the trainer's layout builder: the Training Set becomes a fresh train/val layout."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lb_anythings.adapters.outbound.yolo.training import (
    ExampleFiles,
    TrainingConfig,
    build_training_layout,
    split_for_training,
    training_arguments,
)
from lb_anythings.domain.errors import NotEnoughExamples


def _files(root: Path, count: int) -> list[ExampleFiles]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    files: list[ExampleFiles] = []
    for i in range(count):
        image, label = root / "images" / f"task{i}.jpg", root / "labels" / f"task{i}.txt"
        image.write_bytes(b"jpeg")
        label.write_text("0 0.500000 0.500000 0.100000 0.100000\n")
        files.append((str(i), image, label))
    return files


def test_the_split_holds_out_fifteen_percent_with_at_least_one_for_validation(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path, 20)

    train, val = split_for_training(files, minimum=4)

    assert (len(train), len(val)) == (17, 3)
    assert sorted(train + val) == sorted(files)


def test_the_split_keeps_one_for_validation_even_below_fifteen_percent(tmp_path: Path) -> None:
    train, val = split_for_training(_files(tmp_path, 4), minimum=4)

    assert (len(train), len(val)) == (3, 1)


def test_the_split_is_shuffled_but_repeatable(tmp_path: Path) -> None:
    files = _files(tmp_path, 20)

    first = split_for_training(files, minimum=4)

    assert first == split_for_training(files, minimum=4)
    assert first[0] != files[:17]


def test_too_few_examples_refuse_to_split_with_the_counts_in_the_reason(tmp_path: Path) -> None:
    with pytest.raises(NotEnoughExamples, match="3 .* need at least 4"):
        split_for_training(_files(tmp_path, 3), minimum=4)


def test_the_layout_copies_the_split_and_describes_it(tmp_path: Path) -> None:
    files = _files(tmp_path / "examples", 5)

    layout = build_training_layout(files, ["pipe"], tmp_path / "dataset", minimum=4)

    assert (layout.train_count, layout.val_count) == (4, 1)
    assert sorted(p.name for p in layout.root.rglob("*.jpg")) == [f"task{i}.jpg" for i in range(5)]
    assert sorted(p.name for p in layout.root.rglob("*.txt")) == [f"task{i}.txt" for i in range(5)]
    assert yaml.safe_load(layout.description.read_text()) == {
        "path": str(layout.root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pipe"},
    }


def test_a_rebuilt_layout_starts_fresh(tmp_path: Path) -> None:
    files = _files(tmp_path / "examples", 6)
    build_training_layout(files, ["pipe"], tmp_path / "dataset", minimum=4)

    layout = build_training_layout(files[:5], ["pipe"], tmp_path / "dataset", minimum=4)

    assert not list(layout.root.rglob("task5.*"))


def test_multiple_classes_keep_their_indices(tmp_path: Path) -> None:
    layout = build_training_layout(_files(tmp_path, 4), ["pipe", "car"], tmp_path / "d", minimum=4)

    assert yaml.safe_load(layout.description.read_text())["names"] == {0: "pipe", 1: "car"}


def test_a_store_without_class_names_is_refused_with_advice(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="classes file"):
        build_training_layout(_files(tmp_path, 4), [], tmp_path / "dataset", minimum=4)


CONFIG = TrainingConfig(
    base_model="yolo11s.pt",
    epochs=1,
    patience=3,
    imgsz=640,
    batch=2,
    device="0",
    lr0=None,
    run_name="active",
    min_examples=4,
)


def test_the_run_output_path_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ultralytics reads a relative `project` against its own runs directory, which would put
    # the Checkpoint where the Detector never looks.
    monkeypatch.chdir(tmp_path)
    layout = build_training_layout(
        _files(tmp_path / "examples", 4), ["pipe"], Path("dataset"), minimum=4
    )

    arguments = training_arguments(layout, Path("data/runs"), CONFIG)

    assert Path(str(arguments["project"])).is_absolute()
    assert Path(str(arguments["project"])) == (tmp_path / "data/runs").resolve()
    assert Path(str(arguments["data"])).is_absolute()


def test_an_optional_learning_rate_is_passed_only_when_set(tmp_path: Path) -> None:
    layout = build_training_layout(_files(tmp_path, 4), ["pipe"], tmp_path / "d", minimum=4)

    assert "lr0" not in training_arguments(layout, tmp_path, CONFIG)
    assert training_arguments(layout, tmp_path, replace(CONFIG, lr0=0.005))["lr0"] == 0.005
