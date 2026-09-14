"""Annotation events from Label Studio become Examples in the Training Set."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.annotations import RECTANGLE
from tests.annotations import annotation_event as event
from tests.conftest import Fakes
from tests.label_configs import PIPE, VEHICLES

NO_CONTROL = "<View><Image name='image'/></View>"


def _with_labels(labels: list[str] | None) -> dict[str, Any]:
    value = {k: v for k, v in RECTANGLE["value"].items() if k != "rectanglelabels"}
    if labels is not None:
        value["rectanglelabels"] = labels
    return {**RECTANGLE, "value": value}


def test_a_submitted_annotation_is_stored_as_an_example(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    response = client.post("/webhook", json=event())

    assert response.status_code == 201
    assert response.json().keys() == {"job_id"}
    [example] = fakes.examples.all()
    assert example.task_id == "7"
    assert [r.label for r in example.boxes] == ["pipe"]
    assert example.boxes[0].box.coordinates() == pytest.approx((0.1, 0.1, 0.6, 0.6))


def test_a_second_annotation_of_the_same_task_replaces_the_example(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})
    client.post("/webhook", json=event())

    client.post("/webhook", json=event("ANNOTATION_UPDATED", result=[RECTANGLE, RECTANGLE]))

    [example] = fakes.examples.all()
    assert len(example.boxes) == 2


def test_a_cancelled_annotation_is_ignored(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    client.post("/webhook", json=event(was_cancelled=True))

    assert list(fakes.examples.all()) == []


def test_an_annotation_without_boxes_is_a_negative_example_that_does_not_count(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    client.post("/webhook", json=event(result=[]))

    [example] = fakes.examples.all()
    assert example.boxes == ()
    assert fakes.examples.positive_count() == 0


def test_an_annotators_label_is_kept_as_given_and_a_missing_one_takes_the_first(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": VEHICLES})
    annotation = [_with_labels(["lorry"]), _with_labels(None)]

    client.post("/webhook", json=event(result=annotation, image_field="photo"))

    [example] = fakes.examples.all()
    assert [r.label for r in example.boxes] == ["lorry", "car"]


def test_a_webhook_before_setup_is_acknowledged_and_skipped_with_a_reason(
    client: TestClient, fakes: Fakes
) -> None:
    response = client.post("/webhook", json=event())

    assert response.status_code == 201
    assert response.json()["status"] == "skipped"
    assert "set up" in response.json()["reason"]
    assert list(fakes.examples.all()) == []


def test_a_webhook_carrying_the_label_config_sets_the_project_up(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    payload = event()
    payload["project"]["label_config"] = PIPE

    response = client.post("/webhook", json=payload)

    assert response.json().keys() == {"job_id"}
    assert len(fakes.examples.all()) == 1


def test_a_webhook_carrying_an_unusable_label_config_is_skipped_with_the_parsers_reason(
    client: TestClient, fakes: Fakes
) -> None:
    payload = event()
    payload["project"]["label_config"] = NO_CONTROL

    response = client.post("/webhook", json=payload)

    assert response.json()["status"] == "skipped"
    assert "RectangleLabels" in response.json()["reason"]


def test_other_actions_are_acknowledged_and_skipped(client: TestClient, fakes: Fakes) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    response = client.post("/webhook", json=event("PROJECT_UPDATED"))

    assert response.status_code == 201
    assert response.json() == {
        "status": "skipped",
        "reason": "action 'PROJECT_UPDATED' is not handled",
    }
    assert list(fakes.examples.all()) == []


def test_an_unreadable_image_stores_nothing_but_still_acknowledges(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE})

    response = client.post("/webhook", json=event(image="missing.jpg"))

    assert response.status_code == 201
    assert list(fakes.examples.all()) == []


def test_the_example_image_comes_through_the_media_resolver_with_setup_credentials(
    client: TestClient, fakes: Fakes
) -> None:
    fakes.serve()
    client.post("/setup", json={"schema": PIPE, "hostname": "http://ls:8080", "access_token": "t"})

    client.post("/webhook", json=event(image="a.jpg"))

    assert fakes.media.requests[-1][0] == "a.jpg"
    assert fakes.media.requests[-1][1].access_token == "t"
