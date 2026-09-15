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
    assert settings.data_dir == Path("data").resolve()  # `data`, anchored where we started
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


def test_configured_paths_stop_following_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Training Run is spawned as its own process; a relative path would move under it."""
    started_in = tmp_path / "here"
    started_in.mkdir()
    monkeypatch.chdir(started_in)
    monkeypatch.setenv("LB_DATA_DIR", "data")
    monkeypatch.setenv("LB_CHECKPOINT", "weights/best.pt")
    monkeypatch.setenv("LB_MINE_OUT", "picks")

    settings = Settings()
    monkeypatch.chdir(tmp_path)  # something later runs somewhere else entirely

    assert settings.data_dir == started_in / "data"
    assert settings.checkpoint == started_in / "weights/best.pt"
    assert settings.hard_frames_dir == started_in / "picks"
    for derived in (
        settings.examples_dir,
        settings.runs_dir,
        settings.dataset_dir,
        settings.cache_dir,
        settings.tracking_dir,
        settings.trained_checkpoint,
    ):
        assert derived.is_absolute()
        assert started_in in derived.parents


def test_the_effective_settings_line_says_where_the_backend_will_really_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_DATA_DIR", "data")

    line = render_effective_settings(Settings())

    assert f"data_dir={tmp_path.resolve() / 'data'}" in line
