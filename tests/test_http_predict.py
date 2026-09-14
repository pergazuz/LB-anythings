"""Predicting for Tasks: what an Annotator sees when opening one."""

import logging

import pytest
from fastapi.testclient import TestClient

from lb_anythings.bootstrap.container import build_app
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.geometry import Box
from tests.conftest import BEST, Fakes
from tests.label_configs import PIPE, VEHICLES

ONE_TASK = [{"id": 1, "data": {"image": "a.jpg"}}]
TWO_TASKS = [{"id": 1, "data": {"image": "a.jpg"}}, {"id": 2, "data": {"image": "b.jpg"}}]
SMALL = Box(0.1, 0.1, 0.2, 0.2)


def test_without_a_checkpoint_every_task_gets_an_empty_prediction(client: TestClient) -> None:
    client.post("/setup", json={"schema": PIPE})

    response = client.post("/predict", json={"tasks": TWO_TASKS})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"result": [], "score": 0.0, "model_version": "none"},
            {"result": [], "score": 0.0, "model_version": "none"},
        ],
        "model_version": "none",
    }


def test_detections_become_percent_regions_carrying_the_projects_label(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve(
        [
            Detection(Box.from_pixels(20, 10, 120, 60, width=200, height=100), 0.9, "0"),
            Detection(Box.from_pixels(0, 0, 200, 100, width=200, height=100), 0.5, "0"),
        ]
    )
    client.post("/setup", json={"schema": PIPE})

    body = client.post("/predict", json={"tasks": ONE_TASK}).json()

    assert body["model_version"] == "best.pt#1"
    [prediction] = body["results"]
    assert prediction["score"] == pytest.approx(0.7)
    first, second = prediction["result"]
    assert (first["from_name"], first["to_name"], first["type"]) == (
        "label",
        "image",
        "rectanglelabels",
    )
    assert first["score"] == pytest.approx(0.9)
    assert first["value"]["rectanglelabels"] == ["pipe"]
    geometry = {k: v for k, v in first["value"].items() if k != "rectanglelabels"}
    assert geometry == pytest.approx({"x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0})
    assert second["value"]["width"] == pytest.approx(100.0)


def test_a_detected_label_the_project_knows_is_kept_and_unknown_ones_take_the_first(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve([Detection(SMALL, 0.9, "truck"), Detection(SMALL, 0.8, "bus")])
    client.post("/setup", json={"schema": VEHICLES})

    body = client.post("/predict", json={"tasks": [{"id": 1, "data": {"photo": "a.jpg"}}]}).json()

    labels = [r["value"]["rectanglelabels"] for r in body["results"][0]["result"]]
    assert labels == [["truck"], ["car"]]


def test_a_newer_checkpoint_is_picked_up_before_the_next_prediction(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    assert client.post("/predict", json={"tasks": ONE_TASK}).json()["model_version"] == "best.pt#1"

    fakes.checkpoints.checkpoint = Checkpoint(BEST, modified_at=2.0)

    assert client.post("/predict", json={"tasks": ONE_TASK}).json()["model_version"] == "best.pt#2"


def test_an_unchanged_checkpoint_is_not_reloaded(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    client.post("/predict", json={"tasks": ONE_TASK})
    body = client.post("/predict", json={"tasks": ONE_TASK}).json()

    assert body["model_version"] == "best.pt#1"


def test_force_reload_reloads_the_same_checkpoint(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    client.post("/predict", json={"tasks": ONE_TASK})

    body = client.post("/predict", json={"tasks": ONE_TASK, "force_reload": True}).json()

    assert body["model_version"] == "best.pt#2"


def test_a_label_config_on_predict_sets_the_project_up_when_nothing_is(
    client: TestClient,
) -> None:
    body = client.post("/predict", json={"tasks": ONE_TASK, "label_config": PIPE}).json()

    assert body == {
        "results": [{"result": [], "score": 0.0, "model_version": "none"}],
        "model_version": "none",
    }
    # ...and the context sticks for the next call without one.
    assert len(client.post("/predict", json={"tasks": ONE_TASK}).json()["results"]) == 1


def test_one_unreadable_image_does_not_fail_the_batch(client: TestClient, fakes: Fakes) -> None:
    fakes.serve([Detection(SMALL, 0.9, "pipe")])
    client.post("/setup", json={"schema": PIPE})
    tasks = [
        {"id": 1, "data": {"image": "a.jpg"}},
        {"id": 2, "data": {"image": "missing.jpg"}},
        {"id": 3, "data": {}},
    ]

    response = client.post("/predict", json={"tasks": tasks})

    assert response.status_code == 200
    counts = [len(p["result"]) for p in response.json()["results"]]
    assert counts == [1, 0, 0]


def test_a_later_setup_replaces_the_project_context(client: TestClient, fakes: Fakes) -> None:
    fakes.serve([Detection(SMALL, 0.9, "0")])
    client.post("/setup", json={"schema": PIPE})
    before = client.post("/predict", json={"tasks": ONE_TASK}).json()["results"][0]["result"][0]

    client.post("/setup", json={"schema": VEHICLES})
    after = client.post("/predict", json={"tasks": [{"id": 1, "data": {"photo": "a.jpg"}}]}).json()
    region = after["results"][0]["result"][0]

    assert (before["from_name"], before["value"]["rectanglelabels"]) == ("label", ["pipe"])
    assert (region["from_name"], region["value"]["rectanglelabels"]) == ("boxes", ["car"])


def test_health_and_versions_name_the_checkpoint_serving_predictions(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    client.post("/predict", json={"tasks": ONE_TASK})

    assert client.get("/health").json()["model_version"] == "best.pt#1"
    assert client.post("/versions", json={}).json() == {"versions": ["best.pt#1"]}


def test_a_checkpoint_present_at_startup_is_loaded_before_the_first_request(
    settings: Settings, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    fakes.serve()

    with caplog.at_level(logging.INFO), TestClient(build_app(settings, ports=fakes.ports)) as c:
        assert c.get("/health").json()["model_version"] == "best.pt#1"

    assert fakes.detectors.loads == [BEST]


def test_startup_without_a_checkpoint_says_so(
    settings: Settings, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO), TestClient(build_app(settings, ports=fakes.ports)):
        pass

    assert "no Checkpoint found" in caplog.text
