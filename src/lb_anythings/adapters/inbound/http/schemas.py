"""Request shapes as Label Studio sends them. Tolerant of extra fields: payloads vary by version."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SetupRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_: str | None = Field(default=None, alias="schema")
    label_config: str | None = None
    hostname: str | None = None
    access_token: str | None = None
    force_reload: bool = False

    @property
    def effective_label_config(self) -> str | None:
        return self.schema_ or self.label_config


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tasks: list[dict[str, Any]] = Field(default_factory=list)
    label_config: str | None = None
    force_reload: bool = False


class WebhookRequest(BaseModel):
    """A Label Studio webhook event. `action` names it; the rest varies by event and version."""

    model_config = ConfigDict(extra="ignore")

    action: str | None = None
    task: dict[str, Any] | None = None
    annotation: dict[str, Any] | None = None
    project: dict[str, Any] | None = None
    label_config: str | None = None

    @property
    def effective_label_config(self) -> str | None:
        return self.label_config or (self.project or {}).get("label_config")

    @property
    def project_id(self) -> int | None:
        raw = (self.project or {}).get("id")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
