"""The one project this server is set up for: its Annotation Target and credentials."""

import threading
from dataclasses import dataclass
from urllib.parse import urlparse

from lb_anythings.domain.annotation_target import AnnotationTarget


def same_host(one: str, other: str) -> bool:
    return urlparse(one).netloc.lower() == urlparse(other).netloc.lower()


@dataclass(frozen=True)
class Credentials:
    """A Label Studio hostname and access token, as setup sends them or settings configure them."""

    hostname: str | None = None
    access_token: str | None = None

    def resolved_against(self, configured: "Credentials") -> "Credentials":
        """These credentials, filled in from the configured ones. They win as a pair.

        When setup names a hostname but sends no token, the configured token is used only if
        that hostname is the configured host: it never travels to a host setup introduced.
        """
        if not self.hostname:
            return Credentials(configured.hostname, self.access_token or configured.access_token)
        if self.access_token:
            return self
        borrowable = configured.hostname and same_host(self.hostname, configured.hostname)
        return Credentials(self.hostname, configured.access_token if borrowable else None)


NO_CREDENTIALS = Credentials()


@dataclass(frozen=True)
class ProjectContext:
    target: AnnotationTarget
    credentials: Credentials


class ProjectContextHolder:
    """Thread-safe holder for the current ProjectContext; a later setup replaces it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: ProjectContext | None = None

    def get(self) -> ProjectContext | None:
        with self._lock:
            return self._current

    def set(self, context: ProjectContext) -> None:
        with self._lock:
            self._current = context
