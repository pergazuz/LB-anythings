"""All configuration, read once at startup. Shell variables win over `.env`."""

import logging
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lb_anythings.adapters.outbound.yolo.training import checkpoint_path

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LB_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9090
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # Detector
    checkpoint: Path | None = (
        None  # a configured Checkpoint; the newer of it and the trained one serves
    )
    conf: float = 0.25
    imgsz: int = 1024

    # Training Runs
    train_run_name: str = "active"
    retrain_every: int = Field(default=25, ge=1)  # the Retrain Threshold
    min_examples: int = Field(default=4, ge=1)
    train_base_model: str = "yolo11s.pt"
    train_epochs: int = 100
    train_patience: int = 30
    train_batch: int = 8
    train_lr0: float | None = None
    device: str = "0"

    # Experiment tracking: what a Training Run recorded, and where
    tracking: bool = True  # honoured only when the tracking dependency group is installed
    tracking_uri: str | None = None  # an MLflow URI; default: SQLite under <data dir>/mlflow
    tracking_experiment: str = "lb-anythings"
    tracking_system_metrics: bool = True  # CPU, memory and GPU while a Training Run works

    # Mining Hard Frames
    mine_video: Path | None = None
    mine_stride: int = Field(default=15, ge=1)
    mine_topn: int = Field(default=40, ge=1)
    mine_gap: int = Field(default=60, ge=0)
    mine_uncertain_lo: float = 0.25
    mine_uncertain_hi: float = 0.55
    mine_conf: float = 0.15
    mine_out: Path | None = None  # default: <data dir>/hard_frames

    # Label Studio's own names, unprefixed. HOSTNAME is the previous backend's name for URL.
    label_studio_url: str | None = Field(
        default=None, validation_alias=AliasChoices("LABEL_STUDIO_URL", "LABEL_STUDIO_HOSTNAME")
    )
    label_studio_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LABEL_STUDIO_API_KEY")
    )

    @field_validator("data_dir", "checkpoint", "mine_out", mode="after")
    @classmethod
    def _resolved_once(cls, path: Path | None) -> Path | None:
        """Anchor every configured path to the directory the backend was started in.

        A relative path would otherwise be read against whatever directory finally uses it,
        and that is not always this one: a Training Run is spawned as its own process, and
        ultralytics resolves a relative path against directories of its own. Doing it here,
        once, means nothing downstream can be misled -- and the effective-settings line says
        where the backend will really write rather than what was typed.

        `mine_video` is left alone: it is an input the Operator names on the command line,
        read in that same process, where relative means what they expect.
        """
        return path.resolve() if path is not None else None

    # --- the data directory layout ---
    @property
    def examples_dir(self) -> Path:
        return self.data_dir / "examples"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def hard_frames_dir(self) -> Path:
        return self.mine_out or self.data_dir / "hard_frames"

    @property
    def tracking_dir(self) -> Path:
        """Holds the SQLite store and the archived Checkpoints, unless a URI overrides it."""
        return self.data_dir / "mlflow"

    @property
    def trained_checkpoint(self) -> Path:
        """Where a Training Run writes its Checkpoint, and where the Detector looks first."""
        return checkpoint_path(self.runs_dir, self.train_run_name)


def render_effective_settings(settings: Settings) -> str:
    """One line of `name=value` pairs. Secrets render masked, so this is safe to log."""
    return " ".join(f"{name}={value}" for name, value in settings)


def log_effective_settings(settings: Settings) -> None:
    """Logged once per startup so it is obvious which values actually loaded."""
    logger.info("effective settings: %s", render_effective_settings(settings))
