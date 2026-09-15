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
from collections.abc import Sequence
from enum import Enum, auto
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from lb_anythings.adapters.outbound.labelstudio.auth import LabelStudioAuth
from lb_anythings.application.ports import Image
from lb_anythings.application.project_context import NO_CREDENTIALS, Credentials, same_host
from lb_anythings.domain.errors import MediaUnavailable

logger = logging.getLogger(__name__)

_UNAUTHORIZED = frozenset({401, 403})


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
        self._auth = LabelStudioAuth(self._http)

    def load(self, reference: str, credentials: Credentials) -> Image:
        kind = _classify(reference)
        if kind is _Kind.LOCAL_FILE:
            path = Path(reference)
            if not path.is_file():
                raise MediaUnavailable(f"cannot resolve image reference {reference!r}")
            return _decode(path.read_bytes(), reference)

        effective = credentials.resolved_against(self._defaults)
        hostname = effective.hostname
        if kind is _Kind.LABEL_STUDIO_PATH:
            if not hostname:
                raise MediaUnavailable(
                    f"no Label Studio hostname to resolve {reference!r}: "
                    "Label Studio sends one at setup, or set LABEL_STUDIO_URL"
                )
            url, authorized = hostname.rstrip("/") + reference, True
        else:
            url, authorized = reference, hostname is not None and same_host(reference, hostname)
        candidates = credentials.candidates_against(self._defaults) if authorized else ()
        return _decode(self._fetch_cached(url, candidates, reference), reference)

    def _fetch_cached(self, url: str, candidates: Sequence[Credentials], reference: str) -> bytes:
        entry = self._cache_dir / _cache_name(url)
        if entry.is_file():
            logger.debug("cache hit for %s", reference)
            return entry.read_bytes()
        data = self._fetch(url, candidates, reference)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        partial = entry.with_name(f"{entry.name}.{os.getpid()}.part")
        partial.write_bytes(data)
        os.replace(partial, entry)  # never leave a half-written entry for another request
        logger.debug("fetched %s (%d bytes) into the cache", reference, len(data))
        return data

    def _fetch(self, url: str, candidates: Sequence[Credentials], reference: str) -> bytes:
        attempts = [self._auth.headers(c.hostname, c.access_token) for c in candidates] or [{}]
        failure: Exception = MediaUnavailable(f"fetching {reference!r} was never attempted")
        for remaining, headers in enumerate(attempts, start=1 - len(attempts)):
            try:
                response = self._http.get(url, headers=headers)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as e:
                failure = e
                if e.response.status_code not in _UNAUTHORIZED or not remaining:
                    break
                logger.info(
                    "%s: Label Studio refused its own credentials; trying the configured ones",
                    reference,
                )
            except httpx.HTTPError as e:
                failure = e
                break
        raise MediaUnavailable(f"fetching {reference!r} failed: {failure}") from failure


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
