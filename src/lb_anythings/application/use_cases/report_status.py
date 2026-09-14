"""What the Operator and Label Studio can ask about the backend's state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Status:
    version: str
    is_training: bool = False
    versions: tuple[str, ...] = ()


class ReportStatus:
    def __call__(self) -> Status:
        return Status(version="none")
