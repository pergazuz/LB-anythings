"""The in-process Training Run: lay the Training Set out for ultralytics and train.

The Example Store hands over the positive Examples' files and the class names; this adapter
splits them, copies them into a fresh train/val layout with a data description, and trains.
The Checkpoint lands at `<runs>/<run name>/weights/best.pt`, where the Detector looks for it.
"""

import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from lb_anythings.domain.errors import NotEnoughExamples

ENCODING = "utf-8"
VALIDATION_FRACTION = 0.15

ExampleFiles = tuple[str, Path, Path]  # (task id, image file, label file)


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    epochs: int
    patience: int
    imgsz: int
    batch: int
    device: str
    lr0: float | None
    run_name: str
    min_examples: int


@dataclass(frozen=True)
class TrainingLayout:
    """The train/val folders ultralytics reads, plus the data description that points at them."""

    root: Path
    description: Path
    train_count: int
    val_count: int


@dataclass(frozen=True)
class TrainingReport:
    layout: TrainingLayout
    checkpoint: Path


def checkpoint_path(runs_root: Path, run_name: str) -> Path:
    """Where ultralytics writes a run's best Checkpoint."""
    return runs_root / run_name / "weights" / "best.pt"


def split_for_training(
    examples: Sequence[ExampleFiles], *, minimum: int, seed: int = 0
) -> tuple[list[ExampleFiles], list[ExampleFiles]]:
    """A repeatable shuffle, then 15% held out for validation, never fewer than one."""
    if len(examples) < minimum:
        raise NotEnoughExamples(
            f"{len(examples)} positive Examples; need at least {minimum} to train"
        )
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    held_out = max(1, int(len(shuffled) * VALIDATION_FRACTION))
    return shuffled[held_out:], shuffled[:held_out]


def build_training_layout(
    examples: Sequence[ExampleFiles],
    class_names: Sequence[str],
    layout_root: Path,
    *,
    minimum: int,
    seed: int = 0,
) -> TrainingLayout:
    """Copy the positive Examples into a fresh train/val layout with a data description."""
    if not class_names:
        raise ValueError(
            "the Example Store names no classes; write its classes file (one label per line)"
        )
    train, val = split_for_training(examples, minimum=minimum, seed=seed)

    if layout_root.exists():
        shutil.rmtree(layout_root)
    for split, files in (("train", train), ("val", val)):
        (layout_root / "images" / split).mkdir(parents=True)
        (layout_root / "labels" / split).mkdir(parents=True)
        for _, image, label in files:
            shutil.copy2(image, layout_root / "images" / split / image.name)
            shutil.copy2(label, layout_root / "labels" / split / label.name)

    description = layout_root / "data.yaml"
    description.write_text(
        yaml.safe_dump(
            {
                "path": str(layout_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(class_names)),
            }
        ),
        encoding=ENCODING,
    )
    return TrainingLayout(layout_root, description, len(train), len(val))


def training_arguments(
    layout: TrainingLayout, runs_root: Path, config: TrainingConfig
) -> dict[str, object]:
    """What ultralytics is asked to do. The only place these arguments are named.

    Both paths are resolved here, for the same reason and against different directories of
    ultralytics' own: a relative `project` is read against its runs directory, which would put
    the Checkpoint somewhere the Detector never looks, and a relative `data` against its
    datasets directory. `exist_ok` keeps the run name given: without it ultralytics invents
    `<name>2` and the Detector, which looks under `<name>`, would never see the Checkpoint.
    """
    arguments: dict[str, object] = {
        "data": str(layout.description.resolve()),
        "epochs": config.epochs,
        "patience": config.patience,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "project": str(runs_root.resolve()),
        "name": config.run_name,
        "exist_ok": True,
        "workers": 0,  # Windows-safe: no forked data-loader workers
        "verbose": False,
    }
    if config.lr0 is not None:
        arguments["lr0"] = config.lr0
    return arguments


def run_training(
    *,
    examples: Sequence[ExampleFiles],
    class_names: Sequence[str],
    layout_root: Path,
    runs_root: Path,
    config: TrainingConfig,
) -> TrainingReport:
    layout = build_training_layout(examples, class_names, layout_root, minimum=config.min_examples)
    arguments = training_arguments(layout, runs_root, config)

    from ultralytics import YOLO  # deferred: importing torch takes seconds and a GPU context

    YOLO(config.base_model).train(**arguments)
    # Read back the resolved `project` rather than resolving again: one rule, one place.
    return TrainingReport(layout, checkpoint_path(Path(str(arguments["project"])), config.run_name))
