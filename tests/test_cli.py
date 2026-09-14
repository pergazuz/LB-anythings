"""The console entry point: `lb-anythings serve` starts the server on the configured address."""

import logging
from pathlib import Path

import pytest

from lb_anythings.bootstrap.main import main


class FakeServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, app: object, *, host: str, port: int) -> None:
        self.calls.append((host, port))


def test_serve_uses_settings_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    server = FakeServer()

    main(["serve"], run_server=server)

    assert server.calls == [("0.0.0.0", 9090)]


def test_serve_flags_override_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_PORT", "9000")
    server = FakeServer()

    main(["serve", "--host", "127.0.0.1", "--port", "9001"], run_server=server)

    assert server.calls == [("127.0.0.1", 9001)]


def test_startup_never_logs_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The one property only observable at the log boundary: the secret stays out of it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-token")

    with caplog.at_level(logging.INFO):
        main(["serve"], run_server=FakeServer())

    assert caplog.records, "startup should log its effective settings"
    assert "secret-token" not in caplog.text
