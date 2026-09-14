"""A Prediction: the regions proposed for one Task, and the Checkpoint version behind them."""

from dataclasses import dataclass

from lb_anythings.domain.detection import Detection


@dataclass(frozen=True)
class Prediction:
    regions: tuple[Detection, ...]
    version: str

    @property
    def score(self) -> float:
        """Mean region score, zero when empty: what Label Studio orders predictions by."""
        if not self.regions:
            return 0.0
        return sum(region.score for region in self.regions) / len(self.regions)
