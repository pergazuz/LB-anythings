"""Boxes. The canonical form is normalized xyxy in the unit square, relative to the image.

Label Studio speaks percent (x, y, width, height); YOLO speaks normalized centre + size.
Both are conversions of this one type, and exist nowhere else.
"""

from dataclasses import dataclass


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x1 <= self.x2 <= 1.0 and 0.0 <= self.y1 <= self.y2 <= 1.0):
            raise ValueError(f"box must be ordered and inside the unit square: {self}")

    @classmethod
    def from_pixels(
        cls, x1: float, y1: float, x2: float, y2: float, *, width: int, height: int
    ) -> "Box":
        return cls(_clamp(x1 / width), _clamp(y1 / height), _clamp(x2 / width), _clamp(y2 / height))

    @classmethod
    def from_label_studio(cls, *, x: float, y: float, width: float, height: float) -> "Box":
        return cls(
            _clamp(x / 100), _clamp(y / 100), _clamp((x + width) / 100), _clamp((y + height) / 100)
        )

    def to_label_studio(self) -> dict[str, float]:
        return {
            "x": self.x1 * 100,
            "y": self.y1 * 100,
            "width": (self.x2 - self.x1) * 100,
            "height": (self.y2 - self.y1) * 100,
        }

    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float) -> "Box":
        return cls(_clamp(cx - w / 2), _clamp(cy - h / 2), _clamp(cx + w / 2), _clamp(cy + h / 2))

    def to_yolo(self) -> tuple[float, float, float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
            self.x2 - self.x1,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> float:
        """The fraction of its image this box covers."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def coordinates(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
