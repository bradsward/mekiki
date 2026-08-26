# State

no memory between sessions — this file plus ROADMAP.md and DECISIONS.md is it.

## where things are

M0's done. skeleton's in: pyproject (hatchling, ruff, mypy strict, pytest config), `src/mekiki` with a stub CLI (`--version` only, real subcommands land in M5), tests passing, CI matrix on 3.10-3.13, pre-commit config, apache-2.0 license, readme.

local venv at `.venv/` (gitignored) — ruff, mypy, pytest, pytest-cov, numpy, pyarrow, installed via `pip install -e ".[dev]"`. no uv on this machine, just venv + pip. reuse `.venv/Scripts/python.exe -m <tool>` rather than recreating it unless it's actually missing.

repo lives at github.com/bradsward/mekiki, private. checked this session: ruff check, ruff format --check, mypy --strict, pytest -q all pass, coverage 94% (bar's 90%).

## next

`docs/episode.md` and the `Episode`/`Frame` dataclasses (`src/mekiki/episode.py`) are both done — `ActionDimSpec`/`ActionSpaceSpec`, `Pose`, `Proprioception`, `CameraFrame` (lazy `read()`, never decoded at construction), `Frame`, `EpisodeMetadata`, `Episode` (iterable once, per the streaming rule). Shape/range validation lives in `__post_init__` on `Pose` and `Proprioception`. `tests/conftest.py` has a `make_clean_episode`/`make_clean_frame` factory meant to be reused as the "clean control" fixture once actual detectors (M2+) need one alongside their defect-injected cases.

next: a LeRobotDataset reader that produces real `Episode`s from an actual dataset, round-tripped against one small public example. RLDS/Open X-Embodiment reader after that's solid.

## open questions / risks

- haven't pulled any dataset down in this environment yet — the reader needs network access + some disk space. pick the smallest LeRobot-hosted example with proprio + at least one camera stream once that's confirmed.
- `docs/episode.md` says a reader that can't determine the action space with confidence should fail loudly rather than guess — worth double-checking that's actually enforceable for LeRobotDataset, since I haven't looked at a real one's feature schema yet to see how explicit it actually is about delta vs absolute.
- numpy is pinned to `<2.4` (see DECISIONS.md) purely because of a mypy/stub incompatibility — not a real dependency conflict. Remember to reconsider that pin once mypy catches up, so it doesn't quietly linger for years.

## recommendations

ideas noticed in passing, outside whatever the session's actual task was. need an explicit yes/no before anything happens on them — prune once decided so this doesn't pile up.

- none yet

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
2026-08-25 · M1 · docs/episode.md written (action space, frames, streaming rules) · next: Episode/Frame dataclasses
2026-08-25 · M1 · Episode/Frame dataclasses + clean-episode test fixture · next: LeRobotDataset reader
