"""Console entry point: `lb-anythings <command>`."""

import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import uvicorn

from lb_anythings.adapters.inbound.cli.parser import parse_args
from lb_anythings.adapters.outbound.yolo.training import TrainingReport
from lb_anythings.bootstrap.container import build_app, train_with_yolo
from lb_anythings.bootstrap.settings import Settings, log_effective_settings
from lb_anythings.domain.errors import NotEnoughExamples


class RunServer(Protocol):
    def __call__(self, app: Any, *, host: str, port: int) -> None: ...


def main(
    argv: Sequence[str] | None = None,
    run_server: RunServer = uvicorn.run,
    train: Callable[[Settings], TrainingReport] = train_with_yolo,
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

    if args.command == "train":
        try:
            report = train(settings)
        except NotEnoughExamples as e:
            print(f"cannot train: {e}", file=sys.stderr)
            return 1
        layout = report.layout
        print(f"trained on {layout.train_count} train / {layout.val_count} val Examples")
        print(f"Checkpoint: {report.checkpoint}")
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")  # the parser rejects others
