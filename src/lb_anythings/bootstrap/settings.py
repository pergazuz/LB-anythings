"""All configuration, read once at startup. Shell variables win over `.env`."""

import logging
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    train_run_name: str = "active"

    # Label Studio's own names, unprefixed. HOSTNAME is the previous backend's name for URL.
    label_studio_url: str | None = Field(
        default=None, validation_alias=AliasChoices("LABEL_STUDIO_URL", "LABEL_STUDIO_HOSTNAME")
    )
    label_studio_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LABEL_STUDIO_API_KEY")
    )

    @property
    def trained_checkpoint(self) -> Path:
        """Where a Training Run writes its Checkpoint (ultralytics' own layout under the run)."""
        return self.data_dir / "runs" / self.train_run_name / "weights" / "best.pt"


def render_effective_settings(settings: Settings) -> str:
    """One line of `name=value` pairs. Secrets render masked, so this is safe to log."""
    return " ".join(f"{name}={value}" for name, value in settings)


def log_effective_settings(settings: Settings) -> None:
    """Logged once per startup so it is obvious which values actually loaded."""
    logger.info("effective settings: %s", render_effective_settings(settings))
