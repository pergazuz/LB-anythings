"""Console entry point: `lb-anythings <command>`."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import uvicorn

from lb_anythings.adapters.inbound.cli.parser import parse_args
from lb_anythings.adapters.outbound.yolo.training import TrainingReport
from lb_anythings.application.use_cases.mine_hard_frames import (
    MiningParameters,
    MiningProgress,
)
from lb_anythings.bootstrap.container import build_app, mine_with_yolo, train_with_yolo
from lb_anythings.bootstrap.settings import Settings, log_effective_settings
from lb_anythings.bootstrap.stack import StackOptions, run_stack
from lb_anythings.domain.errors import NoCheckpointAvailable, NotEnoughExamples
from lb_anythings.domain.hard_frames import UncertaintyBand

# CLI flags that override a setting of the same meaning.
MINE_OVERRIDES = {
    "video": "mine_video",
    "stride": "mine_stride",
    "top_n": "mine_topn",
    "gap": "mine_gap",
    "uncertain_lo": "mine_uncertain_lo",
    "uncertain_hi": "mine_uncertain_hi",
    "conf": "mine_conf",
    "out": "mine_out",
}


class RunServer(Protocol):
    def __call__(self, app: Any, *, host: str, port: int) -> None: ...


class Train(Protocol):
    def __call__(self, settings: Settings, tracked_as: str | None = None) -> TrainingReport: ...


class Up(Protocol):
    def __call__(self, settings: Settings, options: StackOptions) -> int: ...


class Mine(Protocol):
    def __call__(
        self,
        settings: Settings,
        video: Path,
        parameters: MiningParameters,
        on_progress: Callable[[MiningProgress], None],
    ) -> list[Path]: ...


def main(
    argv: Sequence[str] | None = None,
    run_server: RunServer = uvicorn.run,
    train: Train = train_with_yolo,
    mine: Mine = mine_with_yolo,
    up: Up = run_stack,
) -> int:
    args = parse_args(argv)
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    log_effective_settings(settings)

    if args.command == "serve":
        run_server(
            build_app(settings),
            host=args.host or settings.host,
            port=args.port or settings.port,
        )
        return 0

    if args.command == "up":
        return up(settings, _stack_options(args))

    if args.command == "train":
        return _train(settings, train, args.tracked_as)

    if args.command == "mine":
        return _mine(settings, args, mine)

    raise AssertionError(f"unhandled command {args.command!r}")  # the parser rejects others


def _stack_options(args: argparse.Namespace) -> StackOptions:
    return StackOptions(
        project_id=args.project,
        title=args.title,
        always_create=args.new,
        labels=tuple(args.labels or ()),
        label_studio=args.label_studio,
        mlflow=args.mlflow,
        wire=args.wire,
        host=args.host,
        port=args.port,
        quiet=args.quiet,
        open_browser=args.open_browser,
    )


def _train(settings: Settings, train: Train, tracked_as: str | None) -> int:
    try:
        report = train(settings, tracked_as)
    except NotEnoughExamples as e:
        print(f"cannot train: {e}", file=sys.stderr)
        return 1
    layout = report.layout
    print(f"trained on {layout.train_count} train / {layout.val_count} val Examples")
    print(f"Checkpoint: {report.checkpoint}")
    return 0


def _mine(settings: Settings, args: argparse.Namespace, mine: Mine) -> int:
    given = {field: getattr(args, flag) for flag, field in MINE_OVERRIDES.items()}
    settings = settings.model_copy(update={k: v for k, v in given.items() if v is not None})
    if settings.mine_video is None:
        print("cannot mine: no video (pass --video or set LB_MINE_VIDEO)", file=sys.stderr)
        return 1
    try:
        parameters = _mining_parameters(settings)
    except ValueError as e:
        print(f"cannot mine: {e}", file=sys.stderr)
        return 1

    def report(progress: MiningProgress) -> None:
        print(
            f"  ...{progress.sampled}/{progress.to_sample} frames scored,"
            f" {progress.candidates} candidates so far"
        )

    try:
        written = mine(settings, settings.mine_video, parameters, report)
    except (NoCheckpointAvailable, FileNotFoundError) as e:
        print(f"cannot mine: {e}", file=sys.stderr)
        return 1
    frames = "Hard Frame" if len(written) == 1 else "Hard Frames"
    print(f"wrote {len(written)} {frames} -> {settings.hard_frames_dir}")
    print("import that folder into Label Studio and label them next")
    return 0


def _mining_parameters(settings: Settings) -> MiningParameters:
    return MiningParameters(
        stride=settings.mine_stride,
        top_n=settings.mine_topn,
        gap=settings.mine_gap,
        band=UncertaintyBand(settings.mine_uncertain_lo, settings.mine_uncertain_hi),
    )
