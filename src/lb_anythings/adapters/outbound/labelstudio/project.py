"""Pulls a project's Tasks and Annotations through Label Studio's export API."""

from collections.abc import Sequence
from typing import Any

import httpx

from lb_anythings.application.ports import ExportedTask
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.annotation import first_usable_annotation
from lb_anythings.domain.errors import ProjectExportFailed
from lb_anythings.domain.task import Task


class LabelStudioExportClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=120.0, follow_redirects=True)

    def exported_tasks(self, project_id: int, credentials: Credentials) -> Sequence[ExportedTask]:
        if not credentials.hostname:
            raise ProjectExportFailed(
                f"no Label Studio hostname to export project {project_id}: "
                "Label Studio sends one at setup, or set LABEL_STUDIO_URL"
            )
        url = f"{credentials.hostname.rstrip('/')}/api/projects/{project_id}/export"
        headers = {"Authorization": f"Token {credentials.access_token}"}
        try:
            response = self._http.get(url, params={"exportType": "JSON"}, headers=headers)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as e:
            raise ProjectExportFailed(
                f"exporting project {project_id} failed: Label Studio answered "
                f"{e.response.status_code}"
            ) from e
        except (httpx.HTTPError, ValueError) as e:
            raise ProjectExportFailed(f"exporting project {project_id} failed: {e}") from e
        if not isinstance(payload, list):
            raise ProjectExportFailed(
                f"exporting project {project_id} failed: the response is not a list of tasks"
            )
        return [_exported(item) for item in payload]


def _exported(item: dict[str, Any]) -> ExportedTask:
    task = Task(id=item.get("id"), data=item.get("data") or {})
    return ExportedTask(task, first_usable_annotation(item.get("annotations") or []))
