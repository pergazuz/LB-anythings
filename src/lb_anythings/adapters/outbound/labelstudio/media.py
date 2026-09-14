"""Resolves a Task's image reference to a decoded image."""

from pathlib import Path

import httpx
import numpy as np

from lb_anythings.application.ports import Image
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import MediaUnavailable


class LabelStudioMediaResolver:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=30.0, follow_redirects=True)

    def load(self, reference: str, credentials: Credentials) -> Image:
        del credentials  # Label Studio-hosted forms, which need them, arrive in the next slice
        return _decode(self._fetch(reference), reference)

    def _fetch(self, reference: str) -> bytes:
        if reference.startswith(("http://", "https://")):
            try:
                response = self._http.get(reference)
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise MediaUnavailable(f"fetching {reference!r} failed: {e}") from e
            return response.content
        path = Path(reference)
        if path.is_file():
            return path.read_bytes()
        raise MediaUnavailable(f"cannot resolve image reference {reference!r}")


def _decode(data: bytes, reference: str) -> Image:
    import cv2  # deferred: part of the ML dependency group

    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise MediaUnavailable(f"cannot decode image {reference!r}")
    return Image(pixels.astype(np.uint8, copy=False))
