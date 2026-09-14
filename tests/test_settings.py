"""Settings are read once, the shell beats `.env`, and secrets never reach the log."""

from pathlib import Path

import pytest

from lb_anythings.bootstrap.settings import Settings, render_effective_settings


@pytest.fixture(autouse=True)
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Settings read `.env` from the working directory; every test starts with none there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_dotenv(directory: Path, text: str) -> None:
    (directory / ".env").write_text(text)


def test_dotenv_is_read_when_the_shell_says_nothing(working_directory: Path) -> None:
    _write_dotenv(working_directory, "LB_PORT=9000\n")

    assert Settings().port == 9000


def test_the_shell_wins_over_dotenv(
    working_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dotenv(working_directory, "LB_PORT=9000\n")
    monkeypatch.setenv("LB_PORT", "9091")

    assert Settings().port == 9091


def test_defaults_match_the_spec() -> None:
    settings = Settings()

    assert (settings.host, settings.port, settings.log_level) == ("0.0.0.0", 9090, "INFO")
    assert settings.data_dir == Path("data")
    assert settings.label_studio_url is None
    assert settings.label_studio_api_key is None


def test_label_studio_names_are_not_prefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-token")

    settings = Settings()

    assert settings.label_studio_url == "http://ls:8080"
    assert settings.label_studio_api_key is not None
    assert settings.label_studio_api_key.get_secret_value() == "secret-token"


def test_label_studio_hostname_is_an_alias_of_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABEL_STUDIO_HOSTNAME", "http://old-name:8080")

    assert Settings().label_studio_url == "http://old-name:8080"


def test_effective_settings_render_every_value_with_the_api_key_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-token")
    monkeypatch.setenv("LB_PORT", "9091")

    line = render_effective_settings(Settings())

    assert "secret-token" not in line
    assert "port=9091" in line
    assert "label_studio_api_key=**********" in line
