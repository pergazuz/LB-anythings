"""Endpoints older Label Studio versions probe; they must answer even with nothing set up."""

from fastapi.testclient import TestClient


def test_is_training_is_false_when_nothing_can_train(client: TestClient) -> None:
    assert client.get("/is_training").json() == {"is_training": False}


def test_metrics_is_an_empty_object(client: TestClient) -> None:
    assert client.get("/metrics").json() == {}


def test_versions_is_empty_until_a_detector_exists(client: TestClient) -> None:
    assert client.post("/versions", json={}).json() == {"versions": []}


def test_predict_without_a_project_context_returns_nothing(client: TestClient) -> None:
    response = client.post("/predict", json={"tasks": [{"id": 1, "data": {"image": "x.jpg"}}]})

    assert response.status_code == 200
    assert response.json() == {"results": [], "model_version": None}
