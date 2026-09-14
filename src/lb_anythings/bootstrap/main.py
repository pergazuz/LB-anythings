"""Console entry point: `lb-anythings <command>`."""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

import uvicorn

from lb_anythings.adapters.inbound.cli.parser import parse_args
from lb_anythings.bootstrap.container import build_app
from lb_anythings.bootstrap.settings import Settings, log_effective_settings


class RunServer(Protocol):
    def __call__(self, app: Any, *, host: str, port: int) -> None: ...


def main(argv: Sequence[str] | None = None, run_server: RunServer = uvicorn.run) -> None:
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
