"""Pulls a project's Tasks and Annotations through Label Studio's export API."""

from collections.abc import Sequence
from typing import Any

import httpx

from lb_anythings.adapters.outbound.labelstudio.auth import LabelStudioAuth
from lb_anythings.application.ports import ExportedTask
from lb_anythings.application.project_context import NO_CREDENTIALS, Credentials
from lb_anythings.domain.annotation import first_usable_annotation
from lb_anythings.domain.errors import ProjectExportFailed
from lb_anythings.domain.task import Task

_UNAUTHORIZED = frozenset({401, 403})


class LabelStudioExportClient:
    def __init__(
        self, http: httpx.Client | None = None, *, defaults: Credentials = NO_CREDENTIALS
    ) -> None:
        self._http = http or httpx.Client(timeout=120.0, follow_redirects=True)
        self._auth = LabelStudioAuth(self._http)
        self._defaults = defaults

    def _export(self, url: str, candidates: Sequence[Credentials], project_id: int) -> object:
        """Try each credential in turn: Label Studio can hand over one its own API rejects."""
        attempts = [self._auth.headers(c.hostname, c.access_token) for c in candidates] or [{}]
        for remaining, headers in enumerate(attempts, start=1 - len(attempts)):
            try:
                response = self._http.get(url, params={"exportType": "JSON"}, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in _UNAUTHORIZED and remaining:
                    continue
                raise ProjectExportFailed(
                    f"exporting project {project_id} failed: Label Studio answered "
                    f"{e.response.status_code}"
                ) from e
            except (httpx.HTTPError, ValueError) as e:
                raise ProjectExportFailed(f"exporting project {project_id} failed: {e}") from e
        raise ProjectExportFailed(f"exporting project {project_id} failed: no credentials worked")

    def exported_tasks(self, project_id: int, credentials: Credentials) -> Sequence[ExportedTask]:
        if not credentials.hostname:
            raise ProjectExportFailed(
                f"no Label Studio hostname to export project {project_id}: "
                "Label Studio sends one at setup, or set LABEL_STUDIO_URL"
            )
        url = f"{credentials.hostname.rstrip('/')}/api/projects/{project_id}/export"
        payload = self._export(url, credentials.candidates_against(self._defaults), project_id)
        if not isinstance(payload, list):
            raise ProjectExportFailed(
                f"exporting project {project_id} failed: the response is not a list of tasks"
            )
        return [_exported(item) for item in payload]


def _exported(item: dict[str, Any]) -> ExportedTask:
    task = Task(id=item.get("id"), data=item.get("data") or {})
    return ExportedTask(task, first_usable_annotation(item.get("annotations") or []))
