"""The one project this server is set up for: its Annotation Target and credentials."""

import threading
from dataclasses import dataclass

from lb_anythings.domain.annotation_target import AnnotationTarget


@dataclass(frozen=True)
class Credentials:
    hostname: str | None = None
    access_token: str | None = None


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
