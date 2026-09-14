"""As an Annotator labels, Training Runs start on their own at the Retrain Threshold."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.annotations import annotation_event
from tests.conftest import Fakes
from tests.label_configs import PIPE


@pytest.fixture
def settings_overrides() -> dict[str, Any]:
    return {"retrain_every": 2, "min_examples": 2}


@pytest.fixture
def labelling(client: TestClient, fakes: Fakes) -> TestClient:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    return client


def _annotate(client: TestClient, *task_ids: int) -> None:
    for task_id in task_ids:
        client.post("/webhook", json=annotation_event(task_id=task_id))


def test_a_run_starts_exactly_when_the_training_set_reaches_the_threshold(
    labelling: TestClient, fakes: Fakes
) -> None:
    _annotate(labelling, 1)
    assert fakes.trainer.runs == []

    _annotate(labelling, 2)

    assert len(fakes.trainer.runs) == 1


def test_nothing_starts_between_thresholds(labelling: TestClient, fakes: Fakes) -> None:
    _annotate(labelling, 1, 2)
    fakes.trainer.finish(fakes.trainer.runs[0], succeeded=True)

    _annotate(labelling, 3)

    assert len(fakes.trainer.runs) == 1


def test_no_second_run_starts_while_one_is_active(labelling: TestClient, fakes: Fakes) -> None:
    _annotate(labelling, 1, 2, 3, 4)

    assert len(fakes.trainer.runs) == 1
    assert fakes.trainer.active() is not None


def test_the_next_multiple_starts_a_run_once_the_previous_finished(
    labelling: TestClient, fakes: Fakes
) -> None:
    _annotate(labelling, 1, 2, 3, 4)  # 4 was refused: run 1 active
    fakes.trainer.finish(fakes.trainer.runs[0], succeeded=True)

    _annotate(labelling, 5)
    assert len(fakes.trainer.runs) == 1
    _annotate(labelling, 6)

    assert len(fakes.trainer.runs) == 2


def test_negatives_and_resubmissions_do_not_move_the_training_set_toward_the_threshold(
    labelling: TestClient, fakes: Fakes
) -> None:
    _annotate(labelling, 1)
    labelling.post("/webhook", json=annotation_event(task_id=2, result=[]))
    _annotate(labelling, 1)  # the same Task again

    assert fakes.trainer.runs == []
