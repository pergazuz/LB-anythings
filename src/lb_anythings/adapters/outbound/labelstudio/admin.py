"""Label Studio's project API: what wiring a project by hand through the UI would do.

Only what `up` needs -- find or create a project, switch on the toggle that makes Label Studio
send Annotations, and connect this backend as the project's model. Everything Label Studio
refuses comes back as ProjectWiringFailed carrying its own words, because its validation
message ("health check failed", "authentication parameters ... incorrect") names the real
problem far better than a status code does.
"""

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from lb_anythings.adapters.outbound.labelstudio.auth import LabelStudioAuth
from lb_anythings.application.ports import ConnectedModel, LabelStudioProject
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.errors import ProjectWiringFailed

logger = logging.getLogger(__name__)

PAGE_SIZE = 100  # Label Studio's own maximum for this endpoint
REFUSAL_LIMIT = 400  # enough of Label Studio's answer to act on, not a wall of HTML


class LabelStudioAdminClient:
    def __init__(self, credentials: Credentials, http: httpx.Client | None = None) -> None:
        self._credentials = credentials
        self._http = http or httpx.Client(timeout=60.0, follow_redirects=True)
        self._auth = LabelStudioAuth(self._http)

    def project(self, project_id: int) -> LabelStudioProject | None:
        found = self._request("GET", f"/api/projects/{project_id}", absent_is_none=True)
        return _project(found) if isinstance(found, dict) else None

    def project_titled(self, title: str) -> LabelStudioProject | None:
        url: str | None = self._url("/api/projects")
        params: dict[str, str | int] | None = {"title": title, "page_size": PAGE_SIZE}
        while url:
            page = self._request("GET", url, params=params)
            for item in _items(page):
                if item.get("title") == title:
                    return _project(item)
            url = page.get("next") if isinstance(page, dict) else None
            params = None  # the next link carries its own
        return None

    def create_project(self, title: str, label_config: str) -> LabelStudioProject:
        created = self._request(
            "POST", "/api/projects", json={"title": title, "label_config": label_config}
        )
        if not isinstance(created, dict) or "id" not in created:
            raise ProjectWiringFailed(f"Label Studio created no project for '{title}'")
        logger.info("created Label Studio project %s (%s)", created["id"], title)
        return _project(created)

    def enable_training_on_submit(self, project_id: int) -> None:
        self._request(
            "PATCH",
            f"/api/projects/{project_id}",
            json={"start_training_on_annotation_update": True},
        )

    def connected_models(self, project_id: int) -> Sequence[ConnectedModel]:
        listed = self._request("GET", "/api/ml", params={"project": project_id})
        return [
            ConnectedModel(str(item["url"]), bool(item.get("is_interactive")))
            for item in _items(listed)
            if item.get("url")
        ]

    def connect_model(self, project_id: int, url: str, title: str) -> None:
        # Label Studio health-checks the URL and calls its /setup before accepting it, so a
        # refusal here is usually the backend, not the request.
        self._request(
            "POST",
            "/api/ml",
            json={"project": project_id, "url": url, "title": title, "is_interactive": True},
        )
        logger.info("connected %s to project %s as '%s'", url, project_id, title)

    # --- one way in and out ---

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        json: dict[str, Any] | None = None,
        absent_is_none: bool = False,
    ) -> Any:
        request = f"{method} {url}"
        try:
            response = self._http.request(
                method, self._url(url), params=params, json=json, headers=self._headers()
            )
            if absent_is_none and response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            raise ProjectWiringFailed(f"{request}: {_why(e.response)}") from e
        except httpx.HTTPError as e:
            raise ProjectWiringFailed(f"{request}: {e}") from e
        except ValueError as e:
            raise ProjectWiringFailed(f"{request}: Label Studio answered with no JSON: {e}") from e

    def _url(self, path: str) -> str:
        hostname = self._credentials.hostname
        if not hostname:
            raise ProjectWiringFailed(
                "no Label Studio to wire a project in: set LABEL_STUDIO_URL"
            )
        return path if path.startswith("http") else f"{hostname.rstrip('/')}{path}"

    def _headers(self) -> dict[str, str]:
        if not self._credentials.access_token:
            raise ProjectWiringFailed(
                "no Label Studio token to wire a project with: set LABEL_STUDIO_API_KEY to a "
                "personal access token (Account & Settings -> Access Token)"
            )
        return self._auth.headers(self._credentials.hostname, self._credentials.access_token)


def _why(response: httpx.Response) -> str:
    detail = response.text.strip().replace("\n", " ")[:REFUSAL_LIMIT]
    if response.status_code in (401, 403):
        return (
            f"Label Studio rejected the token ({response.status_code}); LABEL_STUDIO_API_KEY "
            f"needs a current personal access token. {detail}"
        )
    return f"Label Studio answered {response.status_code}. {detail}"


def _items(payload: Any) -> list[dict[str, Any]]:
    """Label Studio pages some listings and not others, and has changed its mind before."""
    listed = payload.get("results", []) if isinstance(payload, dict) else payload
    return [item for item in listed if isinstance(item, dict)] if isinstance(listed, list) else []


def _project(item: dict[str, Any]) -> LabelStudioProject:
    return LabelStudioProject(int(item["id"]), str(item.get("title", "")))
