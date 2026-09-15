"""The console entry point: `lb-anythings serve` starts the server on the configured address."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.yolo.training import TrainingLayout, TrainingReport
from lb_anythings.application.use_cases.mine_hard_frames import (
    MiningParameters,
    MiningProgress,
)
from lb_anythings.bootstrap.main import main
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.errors import NoCheckpointAvailable, NotEnoughExamples


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


def _mine_nothing(
    settings: Settings,
    video: Path,
    parameters: MiningParameters,
    on_progress: Callable[[MiningProgress], None],
) -> list[Path]:
    return []


def test_mine_needs_a_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    code = main(["mine"], mine=_mine_nothing)

    assert code == 1
    assert "no video" in capsys.readouterr().err


def test_mine_takes_the_video_and_parameters_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_MINE_VIDEO", str(tmp_path / "clip.mov"))
    monkeypatch.setenv("LB_MINE_TOPN", "7")
    seen: list[tuple[Path, MiningParameters]] = []

    def mine(
        settings: Settings,
        video: Path,
        parameters: MiningParameters,
        on_progress: Callable[[MiningProgress], None],
    ) -> list[Path]:
        seen.append((video, parameters))
        return [tmp_path / "hard_000010_s3.jpg"]

    assert main(["mine"], mine=mine) == 0

    [(video, parameters)] = seen
    assert video == tmp_path / "clip.mov"
    assert (parameters.top_n, parameters.stride, parameters.gap) == (7, 15, 60)
    assert (parameters.band.low, parameters.band.high) == (0.25, 0.55)


def test_mine_flags_override_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_MINE_TOPN", "7")
    monkeypatch.setenv("LB_MINE_CONF", "0.15")
    seen: list[tuple[Settings, Path, MiningParameters]] = []

    def mine(
        settings: Settings,
        video: Path,
        parameters: MiningParameters,
        on_progress: Callable[[MiningProgress], None],
    ) -> list[Path]:
        seen.append((settings, video, parameters))
        return []

    code = main(
        [
            "mine",
            "--video",
            str(tmp_path / "other.mov"),
            "--top-n",
            "3",
            "--stride",
            "5",
            "--gap",
            "20",
            "--uncertain-lo",
            "0.3",
            "--uncertain-hi",
            "0.6",
            "--conf",
            "0.1",
            "--out",
            str(tmp_path / "picked"),
        ],
        mine=mine,
    )

    assert code == 0
    [(settings, video, parameters)] = seen
    assert video == tmp_path / "other.mov"
    assert (parameters.top_n, parameters.stride, parameters.gap) == (3, 5, 20)
    assert (parameters.band.low, parameters.band.high) == (0.3, 0.6)
    assert settings.mine_conf == 0.1
    assert settings.hard_frames_dir == tmp_path / "picked"


def test_mine_reports_what_it_wrote_and_where(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_MINE_VIDEO", str(tmp_path / "clip.mov"))
    monkeypatch.setenv("LB_MINE_OUT", str(tmp_path / "picked"))

    def mine(
        settings: Settings,
        video: Path,
        parameters: MiningParameters,
        on_progress: Callable[[MiningProgress], None],
    ) -> list[Path]:
        on_progress(MiningProgress(200, 202, 12))
        return [tmp_path / "picked" / "hard_000010_s3.jpg"]

    main(["mine"], mine=mine)

    out = capsys.readouterr().out
    assert "200/202 frames scored, 12 candidates" in out
    assert "wrote 1 Hard Frame " in out and str(tmp_path / "picked") in out


def test_mine_without_a_checkpoint_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LB_MINE_VIDEO", str(tmp_path / "clip.mov"))

    def refusing(
        settings: Settings,
        video: Path,
        parameters: MiningParameters,
        on_progress: Callable[[MiningProgress], None],
    ) -> list[Path]:
        raise NoCheckpointAvailable("nothing to mine with: train a Checkpoint first")

    assert main(["mine"], mine=refusing) == 1
    assert "train a Checkpoint first" in capsys.readouterr().err
