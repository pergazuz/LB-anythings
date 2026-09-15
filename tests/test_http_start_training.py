"""The Start Training button: pull every annotated Task from the project and train on all."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from lb_anythings.application.project_context import Credentials
from lb_anythings.bootstrap.container import build_app
from lb_anythings.bootstrap.settings import Settings
from tests.annotations import RECTANGLE
from tests.conftest import Fakes
from tests.label_configs import PIPE

CREDENTIALS = {"hostname": "http://ls:8080", "access_token": "t"}


@pytest.fixture
def settings_overrides() -> dict[str, Any]:
    return {"min_examples": 2}


def exported(task_id: int, image: str = "a.jpg", **annotation: Any) -> dict[str, Any]:
    result = annotation.pop("result", [RECTANGLE])
    return {
        "id": task_id,
        "data": {"image": image},
        "annotations": [{"result": result, **annotation}],
    }


def start_training(client: TestClient, project_id: int = 1) -> Any:
    return client.post("/webhook", json={"action": "START_TRAINING", "project": {"id": project_id}})


def test_start_training_collects_every_annotated_task_and_launches_a_run(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [exported(11), exported(12), exported(13)]

    response = start_training(client)

    assert response.status_code == 201 and response.json().keys() == {"job_id"}
    assert sorted(e.task_id for e in fakes.examples.all()) == ["11", "12", "13"]
    assert len(fakes.trainer.runs) == 1


def test_the_project_is_exported_with_the_setup_credentials(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[42] = []

    start_training(client, project_id=42)

    assert fakes.project.requests == [(42, Credentials("http://ls:8080", "t"))]


def test_cancelled_and_unannotated_tasks_are_not_collected(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [
        exported(11),
        exported(12, was_cancelled=True),
        {"id": 13, "data": {"image": "a.jpg"}, "annotations": []},
    ]

    start_training(client)

    assert [e.task_id for e in fakes.examples.all()] == ["11"]


def test_collected_examples_replace_earlier_ones_for_the_same_task(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [exported(11, result=[RECTANGLE, RECTANGLE])]
    start_training(client)
    fakes.project.tasks[1] = [exported(11, result=[RECTANGLE])]

    start_training(client)

    [example] = fakes.examples.all()
    assert len(example.boxes) == 1


def test_no_run_starts_below_the_minimum(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [exported(11)]

    start_training(client)

    assert fakes.trainer.runs == []


def test_no_second_run_starts_while_one_is_active(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [exported(11), exported(12)]
    start_training(client)

    start_training(client)

    assert len(fakes.trainer.runs) == 1


def test_missing_credentials_are_refused_with_what_is_missing(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})  # Label Studio sent no hostname or token

    response = start_training(client)

    assert response.status_code == 201
    assert response.json()["status"] == "skipped"
    assert "LABEL_STUDIO_URL" in response.json()["reason"]
    assert fakes.project.requests == []


def test_a_missing_project_id_is_refused(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})

    response = client.post("/webhook", json={"action": "START_TRAINING"})

    assert response.json()["status"] == "skipped"
    assert "project id" in response.json()["reason"]


def test_the_legacy_train_endpoint_behaves_like_start_training(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.tasks[1] = [exported(11), exported(12)]

    response = client.post("/train", json={"project": {"id": 1}})

    assert response.status_code == 201 and response.json().keys() == {"job_id"}
    assert len(fakes.trainer.runs) == 1


def test_an_export_failure_stores_nothing_and_still_acknowledges(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, **CREDENTIALS})
    fakes.project.fail_with = "Label Studio answered 500"

    response = start_training(client)

    assert response.status_code == 201
    assert list(fakes.examples.all()) == []
    assert fakes.trainer.runs == []


def test_the_configured_token_never_travels_to_a_host_setup_introduced(
    settings: Settings, fakes: Fakes
) -> None:
    configured = Settings(
        data_dir=settings.data_dir,
        min_examples=2,
        label_studio_url="http://ls-configured:8080",
        label_studio_api_key=SecretStr("cfg"),
    )
    with TestClient(build_app(configured, ports=fakes.ports)) as client:
        fakes.serve()
        client.post("/setup", json={"schema": PIPE, "hostname": "http://ls-elsewhere:8080"})
        fakes.project.tasks[1] = []

        response = start_training(client)

    assert response.json()["status"] == "skipped"
    assert "LABEL_STUDIO_API_KEY" in response.json()["reason"]
    assert fakes.project.requests == []
