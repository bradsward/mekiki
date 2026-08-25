# State

no memory between sessions — this file plus ROADMAP.md and DECISIONS.md is it.

## where things are

M0's done. skeleton's in: pyproject (hatchling, ruff, mypy strict, pytest config), `src/mekiki` with a stub CLI (`--version` only, real subcommands land in M5), tests passing, CI matrix on 3.10-3.13, pre-commit config, apache-2.0 license, readme.

local venv at `.venv/` (gitignored) — ruff, mypy, pytest, pytest-cov, numpy, pyarrow, installed via `pip install -e ".[dev]"`. no uv on this machine, just venv + pip. reuse `.venv/Scripts/python.exe -m <tool>` rather than recreating it unless it's actually missing.

repo lives at github.com/bradsward/mekiki, private. checked this session: ruff check, ruff format --check, mypy --strict, pytest -q all pass, coverage 94% (bar's 90%).

## next

M1 — episode abstraction. write `docs/episode.md` before touching code: action space conventions (delta vs absolute), frame/coordinate handling, what a `Frame`/`Episode` actually holds. then the core dataclasses, then a LeRobotDataset reader round-tripped against one small public dataset (smallest LeRobot-hosted example with proprio + at least one camera stream). RLDS/Open X-Embodiment reader after that's solid.

## open questions / risks

- haven't pulled any dataset down in this environment yet — M1 needs network access + some disk space. check that's available before locking in a specific dataset in `docs/episode.md`.

## recommendations

ideas noticed in passing, outside whatever the session's actual task was. need an explicit yes/no before anything happens on them — prune once decided so this doesn't pile up.

- none yet

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
