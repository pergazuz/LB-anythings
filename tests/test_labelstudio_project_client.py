"""Contract of the Label Studio project client: export a project's annotated Tasks."""

import json

import httpx
import pytest

from lb_anythings.adapters.outbound.labelstudio.project import LabelStudioExportClient
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import ProjectExportFailed
from tests.annotations import RECTANGLE

CREDENTIALS = Credentials(hostname="http://ls:8080/", access_token="secret")

EXPORT = [
    {
        "id": 11,
        "data": {"image": "/data/upload/1/a.jpg"},
        "annotations": [
            {"result": [RECTANGLE], "was_cancelled": True},
            {"result": [RECTANGLE, RECTANGLE], "was_cancelled": False},
        ],
    },
    {"id": 12, "data": {"image": "/data/upload/1/b.jpg"}, "annotations": []},
    {"id": 13, "data": {"image": "/data/upload/1/c.jpg"}},
]


def _client(handler: object) -> LabelStudioExportClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LabelStudioExportClient(httpx.Client(transport=transport))


def test_the_export_is_requested_as_json_with_the_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EXPORT)

    _client(handler).exported_tasks(7, CREDENTIALS)

    [request] = seen
    assert str(request.url) == "http://ls:8080/api/projects/7/export?exportType=JSON"
    assert request.headers["Authorization"] == "Token secret"


def test_each_task_comes_with_its_first_non_cancelled_annotation() -> None:
    tasks = _client(lambda r: httpx.Response(200, json=EXPORT)).exported_tasks(7, CREDENTIALS)

    by_id = {t.task.id: t for t in tasks}
    assert set(by_id) == {11, 12, 13}
    assert by_id[11].task.data == {"image": "/data/upload/1/a.jpg"}
    assert by_id[11].annotation is not None and len(by_id[11].annotation.regions) == 2
    assert by_id[12].annotation is None
    assert by_id[13].annotation is None


def test_a_failed_export_is_reported_with_the_status() -> None:
    with pytest.raises(ProjectExportFailed, match="500"):
        _client(lambda r: httpx.Response(500)).exported_tasks(7, CREDENTIALS)


def test_an_export_that_is_not_a_task_list_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"detail": "nope"}).encode())

    with pytest.raises(ProjectExportFailed, match="not a list of tasks"):
        _client(handler).exported_tasks(7, CREDENTIALS)


def test_no_hostname_anywhere_is_reported_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("no request should be made without a hostname")

    with pytest.raises(ProjectExportFailed, match="no Label Studio hostname"):
        _client(handler).exported_tasks(7, Credentials(access_token="secret"))
