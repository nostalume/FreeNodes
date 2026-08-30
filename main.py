#!/usr/bin/env python
"""FreeNodeSpider CLI — AI-powered proxy node crawler.

Usage:
    uv run python main.py
    uv run python main.py clashmeta
    uv run python main.py --validate-profiles .private/profile-validation
    uv run python main.py --help
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from dotenv import load_dotenv
from pydantic import Field

from src.config import FrozenModel, load_config
from src.mihomo import MihomoProbeSession, MihomoValidator, acquire_pinned_mihomo
from src.profiles import PublicEntryRegistry
from src.public_verification import PublicVerificationError, verify_remote_entries
from src.publication import PublicationError, validate_bundle_output_parent
from src.scheduler import Scheduler


# Windows console (GBK) can't encode flag emojis in summary output.
# Force UTF-8 with replacement chars so prints don't crash.
class _ReconfigurableStream(Protocol):
    def reconfigure(self, *, encoding: str, errors: str) -> None: ...


try:
    cast(_ReconfigurableStream, sys.stdout).reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass

# Surface LLM warnings/errors (provider failures, model not found).
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

load_dotenv()
sys.path.insert(0, ".")


class PublishCommand(FrozenModel):
    kind: Literal["publish"] = "publish"


class DiscoverCommand(FrozenModel):
    kind: Literal["discover"] = "discover"
    target: str


class ValidateCommand(FrozenModel):
    kind: Literal["validate"] = "validate"
    output_parent: Path = Field(strict=False)
    target: str | None = None


class VerifyPublicCommand(FrozenModel):
    kind: Literal["verify_public"] = "verify_public"


Command = Annotated[
    PublishCommand | DiscoverCommand | ValidateCommand | VerifyPublicCommand,
    Field(discriminator="kind"),
]


def parse_args(argv: list[str] | None = None) -> Command:
    parser = argparse.ArgumentParser(
        description="FreeNodeSpider — AI-powered proxy node crawler",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="discover one site without publishing (default: publish all configured sites)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--verify-public",
        action="store_true",
        help="fetch and consume the published direct/CDN import entries without writing files",
    )
    modes.add_argument(
        "--validate-profiles",
        dest="validation_output",
        metavar="OUTPUT_DIR",
        help=(
            "discover, render, and validate profiles in a new private bundle "
            "directory; public nodes, README, and config are not changed"
        ),
    )
    args = parser.parse_args(argv)
    if args.verify_public:
        if args.target:
            parser.error("--verify-public does not accept a target")
        return VerifyPublicCommand()
    if args.validation_output:
        return ValidateCommand(
            target=args.target,
            output_parent=args.validation_output,
        )
    if args.target:
        return DiscoverCommand(target=args.target)
    return PublishCommand()


async def run(command: Command) -> int:
    config = load_config()

    scheduler = Scheduler(config)
    if command.kind == "verify_public":
        acquired = acquire_pinned_mihomo(Path(".cache") / "mihomo")
        receipt = await verify_remote_entries(
            PublicEntryRegistry.from_identity(config.repository),
            acquired.executable,
        )
        print(
            f"Public entries: direct={receipt.direct}, cdn={receipt.cdn}, "
            f"generation={receipt.direct_generation}"
        )
        return 0
    if command.kind == "validate":
        output_parent = validate_bundle_output_parent(
            command.output_parent,
            config.output.dir,
        )
        acquired = acquire_pinned_mihomo(output_parent / ".consumer" / "mihomo")
        receipt = await scheduler.validate_profiles(
            output_parent=output_parent,
            validator=MihomoValidator(acquired.executable),
            target=command.target,
        )
        print(
            f"Profiles validated: {receipt.output_dir} "
            f"({receipt.accepted_count} admitted, {receipt.rejected_count} rejected)"
        )
        return 0

    if command.kind == "discover":
        results = await scheduler.run(target=command.target)
        if not results or all(not result.artifacts for result in results):
            return 2
        return 0

    cache = Path(".cache") / "mihomo"
    acquired = acquire_pinned_mihomo(cache)
    validator = MihomoValidator(acquired.executable)
    receipt = await scheduler.publish_profiles(
        repository_root=Path.cwd(),
        validator=validator,
        probe_session=MihomoProbeSession(acquired.executable),
        runner_vantage="github-actions"
        if os.environ.get("GITHUB_ACTIONS")
        else "local",
    )
    print(f"Publication {receipt.status}: {len(receipt.managed_files)} managed files")
    return 0


async def main() -> None:
    try:
        exit_code = await run(parse_args())
    except (PublicationError, PublicVerificationError) as error:
        logging.error("Publication rejected: %s", error)
        exit_code = 2
    except Exception:
        logging.exception("Publication fault")
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
