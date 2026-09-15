"""How a Label Studio access token becomes an Authorization header.

Label Studio has two token schemes and only one of them is the old `Authorization: Token
<key>`. Since 1.ix the default for a fresh install is a personal access token, which is a JWT
*refresh* token: it authenticates nothing by itself, and is exchanged at `/api/token/refresh`
for a short-lived access token sent as `Authorization: Bearer <access>`. A 1.23 install
reports `legacy_api_tokens_enabled: false`, and rejects the old header with 401 even when the
user still has a legacy token on record.

So the scheme is chosen from the token itself: three dot-separated segments means a JWT, and
anything else is a legacy token. Access tokens are cached until shortly before they expire,
because a Prediction fetches an image per Task and each exchange is a round trip.
"""

import base64
import binascii
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Refresh a little before expiry, so a token cannot lapse between the check and the request.
EXPIRY_MARGIN_SECONDS = 30.0
# What to assume when a token does not say when it expires. Label Studio's are far longer.
UNKNOWN_LIFETIME_SECONDS = 60.0


def looks_like_a_jwt(token: str) -> bool:
    return token.count(".") == 2 and all(part for part in token.split("."))


def expires_at(access_token: str) -> float:
    """The `exp` claim, or a cautious guess. The token is ours; there is nothing to verify."""
    try:
        payload = access_token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return float(claims["exp"])
    except (IndexError, ValueError, KeyError, TypeError, binascii.Error):
        return time.time() + UNKNOWN_LIFETIME_SECONDS


class LabelStudioAuth:
    """Turns a hostname and token into the header that Label Studio actually accepts."""

    def __init__(self, http: httpx.Client) -> None:
        self._http = http
        self._access: dict[tuple[str, str], tuple[str, float]] = {}

    def headers(self, hostname: str | None, token: str | None) -> dict[str, str]:
        if not token:
            return {}
        if not looks_like_a_jwt(token) or not hostname:
            return {"Authorization": f"Token {token}"}
        access = self._access_token(hostname, token)
        if access is None:
            # Send the legacy header anyway: an instance that still allows it will accept it,
            # and the caller reports the 401 with its own context if it does not.
            return {"Authorization": f"Token {token}"}
        return {"Authorization": f"Bearer {access}"}

    def _access_token(self, hostname: str, refresh: str) -> str | None:
        key = (hostname.rstrip("/"), refresh)
        cached = self._access.get(key)
        if cached is not None and time.time() < cached[1] - EXPIRY_MARGIN_SECONDS:
            return cached[0]
        access = self._exchange(key[0], refresh)
        if access is not None:
            self._access[key] = (access, expires_at(access))
        return access

    def _exchange(self, hostname: str, refresh: str) -> str | None:
        try:
            response = self._http.post(
                f"{hostname}/api/token/refresh", json={"refresh": refresh}, timeout=30.0
            )
            response.raise_for_status()
            access = response.json().get("access")
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("could not exchange the Label Studio access token: %s", e)
            return None
        if not isinstance(access, str) or not access:
            logger.warning("Label Studio returned no access token to use")
            return None
        return access
