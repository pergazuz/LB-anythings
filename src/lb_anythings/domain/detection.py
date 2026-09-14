"""A Detection: one box with what the Detector says about it."""

from dataclasses import dataclass

from lb_anythings.domain.geometry import Box


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    label: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"a Detection's score is a confidence in [0, 1], got {self.score}")
