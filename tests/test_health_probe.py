"""Whether a Service is up. The one rule: this never raises, it only answers yes or no."""

import httpx

from lb_anythings.adapters.outbound.http.probe import HttpHealthProbe

URL = "http://localhost:8080/health"


def _probe(handler: object) -> HttpHealthProbe:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return HttpHealthProbe(httpx.Client(transport=transport))


def test_a_healthy_service_answers() -> None:
    assert _probe(lambda r: httpx.Response(200)).answers(URL) is True


def test_something_listening_at_the_wrong_path_still_counts_as_up() -> None:
    """The question is whether the Service is there, not whether we guessed its path."""
    assert _probe(lambda r: httpx.Response(404)).answers(URL) is True


def test_a_service_that_answers_with_its_own_failure_is_not_up_yet() -> None:
    assert _probe(lambda r: httpx.Response(503)).answers(URL) is False


def test_nothing_listening_is_not_up() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert _probe(refused).answers(URL) is False


def test_a_service_too_busy_to_answer_yet_is_not_up() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    assert _probe(slow).answers(URL) is False


def test_an_answer_that_is_not_http_at_all_is_not_up() -> None:
    def rubbish(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("not http")

    assert _probe(rubbish).answers(URL) is False
