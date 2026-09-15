"""Contract of Label Studio authentication, against the two token schemes it has.

Verified against a real Label Studio 1.23: it reports `legacy_api_tokens_enabled: false`,
rejects `Authorization: Token <key>` with 401, and hands an ML backend a legacy token at
`/setup` regardless -- one its own API then refuses.
"""

import base64
import json
import time

import httpx
import pytest

from lb_anythings.adapters.outbound.labelstudio.auth import (
    LabelStudioAuth,
    expires_at,
    looks_like_a_jwt,
)
from lb_anythings.application.project_context import Credentials

HOST = "http://ls:8080"
LEGACY = "d88e7439e0180ecc7057dd138c4532312a00d119"


def jwt(**claims: object) -> str:
    def segment(payload: dict[str, object]) -> str:
        raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        return raw.rstrip("=")

    return f"{segment({'alg': 'HS256'})}.{segment(claims)}.signature"


REFRESH = jwt(token_type="refresh", user_id="1")
ACCESS = jwt(token_type="access", exp=time.time() + 300)


class FakeLabelStudio:
    def __init__(self, access: str = ACCESS, status: int = 200) -> None:
        self.access, self.status = access, status
        self.exchanges = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/token/refresh"
        self.exchanges += 1
        if self.status != 200:
            return httpx.Response(self.status)
        return httpx.Response(200, json={"access": self.access})


def client(handler: FakeLabelStudio) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_legacy_token_is_sent_the_legacy_way() -> None:
    talker = FakeLabelStudio()

    headers = LabelStudioAuth(client(talker)).headers(HOST, LEGACY)

    assert headers == {"Authorization": f"Token {LEGACY}"}
    assert talker.exchanges == 0  # nothing to exchange


def test_a_personal_access_token_is_exchanged_and_sent_as_a_bearer() -> None:
    talker = FakeLabelStudio()

    headers = LabelStudioAuth(client(talker)).headers(HOST, REFRESH)

    assert headers == {"Authorization": f"Bearer {ACCESS}"}
    assert talker.exchanges == 1


def test_the_access_token_is_reused_rather_than_exchanged_every_time() -> None:
    """A Prediction fetches an image per Task; an exchange each time is a round trip each time."""
    talker = FakeLabelStudio()
    auth = LabelStudioAuth(client(talker))

    for _ in range(5):
        auth.headers(HOST, REFRESH)

    assert talker.exchanges == 1


def test_an_expired_access_token_is_exchanged_again() -> None:
    talker = FakeLabelStudio(access=jwt(exp=time.time() - 1))
    auth = LabelStudioAuth(client(talker))

    auth.headers(HOST, REFRESH)
    auth.headers(HOST, REFRESH)

    assert talker.exchanges == 2


def test_no_token_means_no_header() -> None:
    assert LabelStudioAuth(client(FakeLabelStudio())).headers(HOST, None) == {}


def test_a_refused_exchange_falls_back_to_the_legacy_header() -> None:
    """An instance that still allows legacy tokens accepts it; the caller reports any 401."""
    talker = FakeLabelStudio(status=401)

    headers = LabelStudioAuth(client(talker)).headers(HOST, REFRESH)

    assert headers == {"Authorization": f"Token {REFRESH}"}


def test_a_token_that_does_not_say_when_it_expires_is_still_usable() -> None:
    assert expires_at("not.a.jwt") > time.time()


@pytest.mark.parametrize(
    ("token", "expected"),
    [(REFRESH, True), (LEGACY, False), ("a.b", False), ("a..c", False), ("", False)],
)
def test_which_tokens_read_as_a_jwt(token: str, expected: bool) -> None:
    assert looks_like_a_jwt(token) is expected


# --- the fallback, when Label Studio hands over a credential it will not accept ---

SETUP = Credentials(hostname=HOST, access_token=LEGACY)
CONFIGURED = Credentials(hostname=HOST, access_token=REFRESH)


def test_setups_credentials_are_tried_first_and_the_configured_ones_second() -> None:
    first, second = SETUP.candidates_against(CONFIGURED)

    assert first.access_token == LEGACY
    assert second.access_token == REFRESH


def test_there_is_nothing_to_fall_back_to_when_the_tokens_are_the_same() -> None:
    same = Credentials(hostname=HOST, access_token=LEGACY)

    assert len(same.candidates_against(same)) == 1


def test_the_configured_token_is_never_offered_to_a_host_setup_introduced() -> None:
    """The pair rule still holds: a fallback is for the same host, not any host."""
    elsewhere = Credentials(hostname="http://somewhere-else:8080", access_token=None)

    (only,) = elsewhere.candidates_against(CONFIGURED)

    assert only.access_token is None
