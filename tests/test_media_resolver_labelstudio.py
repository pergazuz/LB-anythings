"""Contract of the media resolver for images that live inside Label Studio."""

from pathlib import Path

import httpx
import pytest

from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.application.project_context import NO_CREDENTIALS, Credentials
from lb_anythings.domain.errors import MediaUnavailable

SETUP = Credentials(hostname="http://ls-from-setup:8080", access_token="setup-token")
CONFIGURED = Credentials(hostname="http://ls-from-settings:8080/", access_token="cfg-token")
UPLOAD = "/data/upload/3/abc.jpg"


class FakeLabelStudio:
    """Records every request and answers each with the same PNG."""

    def __init__(self, png: bytes) -> None:
        self.png = png
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, content=self.png)

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]

    @property
    def auth_headers(self) -> list[str | None]:
        return [r.headers.get("Authorization") for r in self.requests]


@pytest.fixture
def label_studio(png: bytes) -> FakeLabelStudio:
    return FakeLabelStudio(png)


def _resolver(
    label_studio: FakeLabelStudio, tmp_path: Path, defaults: Credentials = NO_CREDENTIALS
) -> LabelStudioMediaResolver:
    client = httpx.Client(transport=httpx.MockTransport(label_studio))
    return LabelStudioMediaResolver(client, cache_dir=tmp_path / "cache", defaults=defaults)


def test_an_upload_path_is_fetched_from_the_setup_hostname_with_the_token(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    image = _resolver(label_studio, tmp_path).load(UPLOAD, SETUP)

    assert (image.width, image.height) == (40, 30)
    assert label_studio.urls == ["http://ls-from-setup:8080/data/upload/3/abc.jpg"]
    assert label_studio.auth_headers == ["Token setup-token"]


def test_a_local_files_path_keeps_its_query_string(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    _resolver(label_studio, tmp_path).load("/data/local-files/?d=frames/f_000001.jpg", SETUP)

    assert label_studio.urls == [
        "http://ls-from-setup:8080/data/local-files/?d=frames/f_000001.jpg"
    ]


def test_settings_supply_hostname_and_token_when_setup_sent_none(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    _resolver(label_studio, tmp_path, CONFIGURED).load(UPLOAD, Credentials())

    assert label_studio.urls == ["http://ls-from-settings:8080/data/upload/3/abc.jpg"]
    assert label_studio.auth_headers == ["Token cfg-token"]


def test_setup_credentials_take_precedence_over_settings(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    _resolver(label_studio, tmp_path, CONFIGURED).load(UPLOAD, SETUP)

    assert label_studio.urls == ["http://ls-from-setup:8080/data/upload/3/abc.jpg"]
    assert label_studio.auth_headers == ["Token setup-token"]


def test_the_configured_token_never_travels_to_a_host_setup_introduced(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    setup_without_token = Credentials(hostname="http://ls-from-setup:8080")

    _resolver(label_studio, tmp_path, CONFIGURED).load(UPLOAD, setup_without_token)

    assert label_studio.urls == ["http://ls-from-setup:8080/data/upload/3/abc.jpg"]
    assert label_studio.auth_headers == [None]


def test_the_configured_token_serves_the_configured_host_when_setup_sent_none(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    same_host_spelled_differently = Credentials(hostname="http://LS-FROM-SETTINGS:8080")

    _resolver(label_studio, tmp_path, CONFIGURED).load(UPLOAD, same_host_spelled_differently)

    assert label_studio.auth_headers == ["Token cfg-token"]


def test_a_label_studio_path_with_no_hostname_anywhere_fails_with_a_reason(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    with pytest.raises(MediaUnavailable, match="no Label Studio hostname"):
        _resolver(label_studio, tmp_path).load(UPLOAD, Credentials())


def test_an_absolute_url_on_the_label_studio_host_carries_the_token(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    _resolver(label_studio, tmp_path).load("http://LS-From-Setup:8080/data/upload/3/abc.jpg", SETUP)

    assert label_studio.auth_headers == ["Token setup-token"]


def test_an_absolute_url_elsewhere_is_fetched_without_the_token(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    _resolver(label_studio, tmp_path).load("http://images.example/abc.jpg", SETUP)

    assert label_studio.auth_headers == [None]


def test_a_fetched_image_is_served_from_the_cache_next_time(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    resolver = _resolver(label_studio, tmp_path)

    resolver.load(UPLOAD, SETUP)
    again = resolver.load(UPLOAD, SETUP)

    assert len(label_studio.requests) == 1
    assert (again.width, again.height) == (40, 30)


def test_different_references_do_not_share_a_cache_entry(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    resolver = _resolver(label_studio, tmp_path)

    resolver.load(UPLOAD, SETUP)
    resolver.load("/data/upload/3/def.jpg", SETUP)

    assert len(label_studio.requests) == 2


def test_the_same_path_on_a_new_hostname_is_fetched_again(
    label_studio: FakeLabelStudio, tmp_path: Path
) -> None:
    resolver = _resolver(label_studio, tmp_path)

    resolver.load(UPLOAD, SETUP)
    resolver.load(UPLOAD, Credentials(hostname="http://another-ls:8080", access_token="t2"))

    assert label_studio.urls == [
        "http://ls-from-setup:8080/data/upload/3/abc.jpg",
        "http://another-ls:8080/data/upload/3/abc.jpg",
    ]
