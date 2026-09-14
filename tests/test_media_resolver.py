"""Contract of the media resolver: local files and plain URLs (Label Studio forms come later)."""

from pathlib import Path

import httpx
import numpy as np
import pytest

from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import MediaUnavailable

NO_CREDENTIALS = Credentials()


@pytest.fixture
def png() -> bytes:
    cv2 = pytest.importorskip("cv2", reason="decoding needs the ML dependency group")
    ok, encoded = cv2.imencode(".png", np.zeros((30, 40, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded)


def _resolver(handler: object) -> LabelStudioMediaResolver:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LabelStudioMediaResolver(httpx.Client(transport=transport))


def test_a_local_file_is_decoded(tmp_path: Path, png: bytes) -> None:
    path = tmp_path / "frame.png"
    path.write_bytes(png)

    image = LabelStudioMediaResolver().load(str(path), NO_CREDENTIALS)

    assert (image.width, image.height) == (40, 30)


def test_a_plain_url_is_fetched_and_decoded(png: bytes) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=png)

    image = _resolver(handler).load("http://images.example/frame.png", NO_CREDENTIALS)

    assert (image.width, image.height) == (40, 30)
    assert seen == ["http://images.example/frame.png"]


def test_an_http_error_is_media_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(MediaUnavailable, match="fetching .* failed"):
        _resolver(handler).load("http://images.example/gone.png", NO_CREDENTIALS)


def test_undecodable_bytes_are_media_unavailable() -> None:
    pytest.importorskip("cv2", reason="decoding needs the ML dependency group")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not an image")

    with pytest.raises(MediaUnavailable, match="cannot decode"):
        _resolver(handler).load("http://images.example/garbage.png", NO_CREDENTIALS)


def test_an_unknown_reference_is_media_unavailable(tmp_path: Path) -> None:
    with pytest.raises(MediaUnavailable, match="cannot resolve"):
        LabelStudioMediaResolver().load(str(tmp_path / "missing.png"), NO_CREDENTIALS)
