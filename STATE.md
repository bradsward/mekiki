# State

Read this first. You have no memory between sessions — this file, `ROADMAP.md`,
and `DECISIONS.md` are it.

## Where things stand

M0 (Foundation) is done. Repo skeleton exists: `pyproject.toml` (hatchling
build, ruff + mypy strict + pytest config), `src/mekiki/` package with a
stubbed CLI entry point (`mekiki.cli:main`, `--version` only — real
subcommands are M5), `tests/` with passing coverage, GitHub Actions CI
(`.github/workflows/ci.yml`, matrix over Python 3.10–3.13), pre-commit config,
Apache-2.0 `LICENSE` (fetched verbatim from apache.org), and a README stub.

A local dev venv exists at `.venv/` (gitignored) with `ruff`, `mypy`,
`pytest`, `pytest-cov`, `numpy`, `pyarrow` installed via `pip install -e
".[dev]"`. No `uv` on this machine — plain `venv` + `pip` was used instead.
Future sessions: reuse `.venv/Scripts/python.exe -m <tool>` rather than
recreating the venv, unless it's missing.

Verified this session: `ruff check .`, `ruff format --check .`,
`mypy --strict src/`, `pytest -q` all pass. Coverage 94% (bar is 90%).

## Next

M1 — Episode abstraction. Per the roadmap, **write `docs/episode.md` before
any code**: nail down action-space conventions (delta vs. absolute), frame/
coordinate handling, and what a `Frame`/`Episode` actually holds. Only after
that, build the core dataclasses and a LeRobotDataset reader, round-tripped
against one small public dataset (e.g. a small LeRobot-hosted example on
Hugging Face — pick the smallest one that has proprioception + at least one
camera stream). RLDS/Open X-Embodiment reader comes after LeRobot round-trips
cleanly.

## Open questions / risks

- No dataset has been pulled down yet in this environment — M1's first task
  needs network access to Hugging Face (or wherever LeRobotDataset examples
  are hosted) and some non-trivial disk space. Check that's available before
  committing to a specific dataset in `docs/episode.md`.
- GitHub remote does not exist yet (local repo only, no `origin`). CI won't
  actually run until this is pushed somewhere. Not urgent — no reason to
  create a remote until there's a reason to push.

## Log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, README · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
