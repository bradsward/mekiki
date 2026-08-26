# State

no memory between sessions — this file plus ROADMAP.md and DECISIONS.md is it.

## where things are

M0's done. skeleton's in: pyproject (hatchling, ruff, mypy strict, pytest config), `src/mekiki` with a stub CLI (`--version` only, real subcommands land in M5), tests passing, CI matrix on 3.10-3.13, pre-commit config, apache-2.0 license, readme.

local venv at `.venv/` (gitignored) — ruff, mypy, pytest, pytest-cov, numpy, pyarrow, installed via `pip install -e ".[dev]"`. no uv on this machine, just venv + pip. reuse `.venv/Scripts/python.exe -m <tool>` rather than recreating it unless it's actually missing.

repo lives at github.com/bradsward/mekiki, private. checked this session: ruff check, ruff format --check, mypy --strict, pytest -q all pass, coverage 94% (bar's 90%).

## next

`docs/episode.md` is written — action space conventions (per-dimension delta vs absolute, explicit units/frames), frame handling (readers preserve source frame labels, no silent conversion), camera streams (lazy decode, timestamps independent of control timestamp), streaming rules. next: the `Episode`/`Frame` dataclasses themselves, typed against that doc, then a LeRobotDataset reader round-tripped against one small public dataset. RLDS/Open X-Embodiment reader after that's solid.

## open questions / risks

- haven't pulled any dataset down in this environment yet — the reader needs network access + some disk space. pick the smallest LeRobot-hosted example with proprio + at least one camera stream once that's confirmed.
- `docs/episode.md` says a reader that can't determine the action space with confidence should fail loudly rather than guess — worth double-checking that's actually enforceable for LeRobotDataset, since I haven't looked at a real one's feature schema yet to see how explicit it actually is about delta vs absolute.

## recommendations

ideas noticed in passing, outside whatever the session's actual task was. need an explicit yes/no before anything happens on them — prune once decided so this doesn't pile up.

- none yet

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
2026-08-25 · M1 · docs/episode.md written (action space, frames, streaming rules) · next: Episode/Frame dataclasses
