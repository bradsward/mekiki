"""Tests for the mekiki CLI entry point."""

import pytest

from mekiki import __version__
from mekiki.cli import build_parser, main


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``--version`` prints the package version and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_main_with_no_args_returns_zero() -> None:
    """Running with no arguments succeeds (no subcommands exist yet)."""
    assert main([]) == 0


def test_build_parser_prog_name() -> None:
    """The parser identifies itself as ``mekiki``."""
    assert build_parser().prog == "mekiki"
