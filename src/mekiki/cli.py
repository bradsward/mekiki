"""Command-line entry point for mekiki.

This is a placeholder: the real ``audit``/``report``/``coverage`` subcommands
land in M5. For now it only proves the packaging and entry-point wiring work
end to end, and answers ``--version``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mekiki import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        An ``argparse.ArgumentParser`` with mekiki's version flag wired up.
        Subcommands (``audit``, ``report``, ``coverage``) are added in M5.
    """
    parser = argparse.ArgumentParser(
        prog="mekiki",
        description="Find the bad demonstrations before you train on them.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mekiki {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mekiki CLI.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code. ``0`` on success.

    Example:
        >>> main(["--version"])  # doctest: +SKIP
        mekiki 0.1.0
        0
    """
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
