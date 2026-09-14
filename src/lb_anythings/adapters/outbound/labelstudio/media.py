"""Resolves a Task's image reference to a decoded image.

A reference is one of three kinds: a path inside Label Studio (`/data/upload/...` or
`/data/local-files/?d=...`), an absolute http(s) URL, or a local file. Label Studio-hosted
images need the hostname and access token; setup's credentials win over the configured ones
as a pair, so the configured token only ever travels to the configured host. Fetched bytes
are cached under the data directory, keyed by a hash of the resolved URL.
"""

import hashlib
import logging
import os
from enum import Enum, auto
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from lb_anythings.application.ports import Image
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import MediaUnavailable

logger = logging.getLogger(__name__)

NO_CREDENTIALS = Credentials()


class _Kind(Enum):
    LABEL_STUDIO_PATH = auto()
    ABSOLUTE_URL = auto()
    LOCAL_FILE = auto()


def _classify(reference: str) -> _Kind:
    if reference.startswith("/data/"):
        return _Kind.LABEL_STUDIO_PATH
    if reference.startswith(("http://", "https://")):
        return _Kind.ABSOLUTE_URL
    return _Kind.LOCAL_FILE


def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


class LabelStudioMediaResolver:
    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        cache_dir: Path,
        defaults: Credentials = NO_CREDENTIALS,
    ) -> None:
        self._http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        self._cache_dir = cache_dir
        self._defaults = defaults

    def load(self, reference: str, credentials: Credentials) -> Image:
        kind = _classify(reference)
        if kind is _Kind.LOCAL_FILE:
            path = Path(reference)
            if not path.is_file():
                raise MediaUnavailable(f"cannot resolve image reference {reference!r}")
            return _decode(path.read_bytes(), reference)

        hostname, token = self._effective(credentials)
        if kind is _Kind.LABEL_STUDIO_PATH:
            if not hostname:
                raise MediaUnavailable(
                    f"no Label Studio hostname to resolve {reference!r}: "
                    "Label Studio sends one at setup, or set LABEL_STUDIO_URL"
                )
            url, authorized = hostname.rstrip("/") + reference, True
        else:
            url, authorized = reference, hostname is not None and _same_host(reference, hostname)
        headers = {"Authorization": f"Token {token}"} if authorized and token else {}
        return _decode(self._fetch_cached(url, headers, reference), reference)

    def _effective(self, credentials: Credentials) -> tuple[str | None, str | None]:
        """Setup's credentials win as a pair over the configured defaults.

        When setup names a hostname but sends no token, the configured token is used only if
        that hostname is the configured host; it never travels to a host setup introduced.
        """
        if credentials.hostname:
            token = credentials.access_token
            configured = self._defaults.hostname
            if token is None and configured and _same_host(credentials.hostname, configured):
                token = self._defaults.access_token
            return credentials.hostname, token
        return self._defaults.hostname, credentials.access_token or self._defaults.access_token

    def _fetch_cached(self, url: str, headers: dict[str, str], reference: str) -> bytes:
        entry = self._cache_dir / _cache_name(url)
        if entry.is_file():
            logger.debug("cache hit for %s", reference)
            return entry.read_bytes()
        data = self._fetch(url, headers, reference)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        partial = entry.with_name(f"{entry.name}.{os.getpid()}.part")
        partial.write_bytes(data)
        os.replace(partial, entry)  # never leave a half-written entry for another request
        logger.debug("fetched %s (%d bytes) into the cache", reference, len(data))
        return data

    def _fetch(self, url: str, headers: dict[str, str], reference: str) -> bytes:
        try:
            response = self._http.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise MediaUnavailable(f"fetching {reference!r} failed: {e}") from e
        return response.content


def _cache_name(url: str) -> str:
    digest = hashlib.sha256(url.encode()).hexdigest()[:32]
    suffix = Path(urlparse(url).path).suffix.lower()
    return digest + (suffix if suffix.isascii() and suffix[1:].isalnum() else "")


def _decode(data: bytes, reference: str) -> Image:
    import cv2  # deferred: part of the ML dependency group

    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise MediaUnavailable(f"cannot decode image {reference!r}")
    return Image(pixels.astype(np.uint8, copy=False))
