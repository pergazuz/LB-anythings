"""The console entry point: `lb-anythings serve` starts the server on the configured address."""

import logging
from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.yolo.training import TrainingLayout, TrainingReport
from lb_anythings.bootstrap.main import main
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.errors import NotEnoughExamples


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


def test_train_reports_the_split_and_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    def training(settings: Settings) -> TrainingReport:
        layout = TrainingLayout(tmp_path / "dataset", tmp_path / "dataset/data.yaml", 17, 3)
        return TrainingReport(layout, tmp_path / "runs/active/weights/best.pt")

    code = main(["train"], train=training)

    out = capsys.readouterr().out
    assert code == 0
    assert "17 train / 3 val" in out and "best.pt" in out


def test_train_exits_non_zero_when_the_training_set_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    def refusing(settings: Settings) -> TrainingReport:
        raise NotEnoughExamples("3 positive Examples; need at least 4 to train")

    code = main(["train"], train=refusing)

    assert code == 1
    assert "need at least 4" in capsys.readouterr().err
