"""Contract of the media resolver for local files and plain URLs."""

from pathlib import Path

import httpx
import pytest

from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import MediaUnavailable

NO_CREDENTIALS = Credentials()


def _resolver(handler: object, tmp_path: Path) -> LabelStudioMediaResolver:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LabelStudioMediaResolver(httpx.Client(transport=transport), cache_dir=tmp_path / "cache")


def test_a_local_file_is_decoded(tmp_path: Path, png: bytes) -> None:
    path = tmp_path / "frame.png"
    path.write_bytes(png)

    image = LabelStudioMediaResolver(cache_dir=tmp_path / "cache").load(str(path), NO_CREDENTIALS)

    assert (image.width, image.height) == (40, 30)


def test_a_plain_url_is_fetched_and_decoded(tmp_path: Path, png: bytes) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=png)

    image = _resolver(handler, tmp_path).load("http://images.example/frame.png", NO_CREDENTIALS)

    assert (image.width, image.height) == (40, 30)
    assert seen == ["http://images.example/frame.png"]


def test_an_http_error_is_media_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(MediaUnavailable, match="fetching .* failed"):
        _resolver(handler, tmp_path).load("http://images.example/gone.png", NO_CREDENTIALS)


def test_undecodable_bytes_are_media_unavailable(tmp_path: Path) -> None:
    pytest.importorskip("cv2", reason="decoding needs the ML dependency group")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not an image")

    with pytest.raises(MediaUnavailable, match="cannot decode"):
        _resolver(handler, tmp_path).load("http://images.example/garbage.png", NO_CREDENTIALS)


def test_an_unknown_reference_is_media_unavailable(tmp_path: Path) -> None:
    with pytest.raises(MediaUnavailable, match="cannot resolve"):
        LabelStudioMediaResolver(cache_dir=tmp_path / "cache").load(
            str(tmp_path / "missing.png"), NO_CREDENTIALS
        )
