"""Contract of the Label Studio admin client: find, create and wire a project.

This is the half of Label Studio's API an Operator would otherwise drive by hand through the
UI, so the tests are about the requests it sends, not about what it does with the answers.
"""

import json

import httpx
import pytest

from lb_anythings.adapters.outbound.labelstudio.admin import LabelStudioAdminClient
from lb_anythings.application.ports import ConnectedModel, LabelStudioProject
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import ProjectWiringFailed

CREDENTIALS = Credentials(hostname="http://ls:8080/", access_token="secret")

Handler = object


def _client(handler: Handler, credentials: Credentials = CREDENTIALS) -> LabelStudioAdminClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LabelStudioAdminClient(credentials, httpx.Client(transport=transport))


def _recording(response: httpx.Response) -> tuple[list[httpx.Request], object]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return seen, handler


# --- reading projects ---


def test_a_project_is_read_by_id_with_the_token() -> None:
    seen, handler = _recording(httpx.Response(200, json={"id": 7, "title": "Pipes"}))

    project = _client(handler).project(7)

    assert project == LabelStudioProject(7, "Pipes")
    assert str(seen[0].url) == "http://ls:8080/api/projects/7"
    assert seen[0].headers["Authorization"] == "Token secret"


def test_a_project_that_is_not_there_is_not_an_error() -> None:
    assert _client(lambda r: httpx.Response(404)).project(7) is None


def test_a_project_is_found_by_its_exact_title() -> None:
    """Label Studio's title filter matches substrings, so the exact one has to be picked out."""
    listing = {
        "results": [
            {"id": 3, "title": "LB-anythings (old)"},
            {"id": 4, "title": "LB-anythings"},
        ],
        "next": None,
    }

    project = _client(lambda r: httpx.Response(200, json=listing)).project_titled("LB-anythings")

    assert project == LabelStudioProject(4, "LB-anythings")


def test_the_title_is_sent_as_a_filter_so_a_big_instance_is_not_walked() -> None:
    seen, handler = _recording(httpx.Response(200, json={"results": [], "next": None}))

    _client(handler).project_titled("LB-anythings")

    assert seen[0].url.params["title"] == "LB-anythings"


def test_a_title_spread_over_pages_is_still_found() -> None:
    pages = {
        "http://ls:8080/api/projects": {
            "results": [{"id": 1, "title": "Other"}],
            "next": "http://ls:8080/api/projects?page=2",
        },
        "http://ls:8080/api/projects?page=2": {
            "results": [{"id": 2, "title": "Pipes"}],
            "next": None,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        key = "http://ls:8080/api/projects" + (f"?page={page}" if page else "")
        return httpx.Response(200, json=pages[key])

    assert _client(handler).project_titled("Pipes") == LabelStudioProject(2, "Pipes")


def test_a_listing_that_is_a_bare_list_is_read_too() -> None:
    """Not every Label Studio paginates this endpoint."""
    listing = [{"id": 9, "title": "Pipes"}]

    assert _client(lambda r: httpx.Response(200, json=listing)).project_titled("Pipes") is not None


def test_no_project_with_that_title_is_not_an_error() -> None:
    listing = {"results": [{"id": 3, "title": "Something"}], "next": None}

    assert _client(lambda r: httpx.Response(200, json=listing)).project_titled("Pipes") is None


# --- creating and wiring ---


def test_a_created_project_carries_its_title_and_config() -> None:
    seen, handler = _recording(httpx.Response(201, json={"id": 11, "title": "Pipes"}))

    project = _client(handler).create_project("Pipes", "<View/>")

    assert project == LabelStudioProject(11, "Pipes")
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://ls:8080/api/projects"
    assert json.loads(seen[0].content) == {"title": "Pipes", "label_config": "<View/>"}


def test_training_on_submit_is_switched_on_with_a_patch() -> None:
    seen, handler = _recording(httpx.Response(200, json={"id": 7}))

    _client(handler).enable_training_on_submit(7)

    assert seen[0].method == "PATCH"
    assert str(seen[0].url) == "http://ls:8080/api/projects/7"
    assert json.loads(seen[0].content) == {"start_training_on_annotation_update": True}


def test_the_models_connected_to_a_project_are_listed_for_that_project() -> None:
    listing = {
        "results": [{"id": 1, "url": "http://localhost:9090", "is_interactive": True}],
        "next": None,
    }
    seen, handler = _recording(httpx.Response(200, json=listing))

    models = _client(handler).connected_models(7)

    assert list(models) == [ConnectedModel("http://localhost:9090", interactive=True)]
    assert seen[0].url.params["project"] == "7"


def test_a_model_label_studio_never_asks_for_predictions_is_read_as_such() -> None:
    """The field is absent on older answers, and absent means off."""
    listing = {"results": [{"id": 1, "url": "http://localhost:9090"}], "next": None}

    models = _client(lambda r: httpx.Response(200, json=listing)).connected_models(7)

    assert list(models) == [ConnectedModel("http://localhost:9090", interactive=False)]


def test_connecting_the_model_asks_for_interactive_predictions() -> None:
    """Without it Label Studio never asks for a Prediction while a Task is open."""
    seen, handler = _recording(httpx.Response(201, json={"id": 1}))

    _client(handler).connect_model(7, "http://localhost:9090", "lb-anythings")

    assert str(seen[0].url) == "http://ls:8080/api/ml"
    assert json.loads(seen[0].content) == {
        "project": 7,
        "url": "http://localhost:9090",
        "title": "lb-anythings",
        "is_interactive": True,
    }


# --- when Label Studio says no ---


def test_label_studios_own_words_are_kept_when_it_refuses_a_model() -> None:
    """Its validation message names the real problem: an unreachable or unhealthy backend."""
    refusal = ["Can't connect to ML backend http://localhost:9090, health check failed."]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=refusal)

    with pytest.raises(ProjectWiringFailed, match="health check failed"):
        _client(handler).connect_model(7, "http://localhost:9090", "lb-anythings")


def test_a_rejected_token_says_so() -> None:
    with pytest.raises(ProjectWiringFailed, match="token"):
        _client(lambda r: httpx.Response(401)).project_titled("Pipes")


def test_a_label_studio_that_cannot_be_reached_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProjectWiringFailed, match="connection refused"):
        _client(handler).project(7)


def test_without_a_hostname_it_says_what_to_set() -> None:
    client = _client(lambda r: httpx.Response(200), Credentials(access_token="secret"))

    with pytest.raises(ProjectWiringFailed, match="LABEL_STUDIO_URL"):
        client.project(7)


def test_without_a_token_it_says_what_to_set() -> None:
    client = _client(lambda r: httpx.Response(200), Credentials(hostname="http://ls:8080"))

    with pytest.raises(ProjectWiringFailed, match="LABEL_STUDIO_API_KEY"):
        client.project(7)


def test_a_personal_access_token_is_exchanged_and_sent_as_a_bearer_token() -> None:
    """The same scheme the rest of the backend uses; a JWT is not a legacy token."""
    jwt = "header.payload.signature"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/token/refresh":
            return httpx.Response(200, json={"access": "short-lived"})
        return httpx.Response(200, json={"id": 7, "title": "Pipes"})

    _client(handler, Credentials("http://ls:8080", jwt)).project(7)

    assert seen[-1].headers["Authorization"] == "Bearer short-lived"
