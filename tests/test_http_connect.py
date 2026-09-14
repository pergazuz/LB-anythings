"""Connecting from Label Studio: what the Connect Model dialog exercises."""

import pytest
from fastapi.testclient import TestClient

from tests.label_configs import PIPE

NO_CONTROL = "<View><Image name='image'/></View>"


@pytest.mark.parametrize("path", ["/health", "/"])
def test_health_reports_up_and_no_checkpoint(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.json() == {"status": "UP", "model_version": "none"}


def test_setup_accepts_a_valid_label_config(client: TestClient) -> None:
    response = client.post(
        "/setup", json={"schema": PIPE, "hostname": "http://ls:8080", "access_token": "t"}
    )

    assert response.status_code == 200
    assert response.json() == {"model_version": "none"}


def test_setup_rejects_an_invalid_label_config_with_the_reason(client: TestClient) -> None:
    response = client.post("/setup", json={"schema": NO_CONTROL})

    assert response.status_code == 400
    assert response.json() == {"detail": "expected exactly one RectangleLabels control, found 0"}


def test_setup_also_accepts_the_label_config_key(client: TestClient) -> None:
    response = client.post("/setup", json={"label_config": PIPE})

    assert response.status_code == 200
