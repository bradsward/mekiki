# State

no memory between sessions — this file plus ROADMAP.md and DECISIONS.md is it.

## where things are

M0's done. skeleton's in: pyproject (hatchling, ruff, mypy strict, pytest config), `src/mekiki` with a stub CLI (`--version` only, real subcommands land in M5), tests passing, CI matrix on 3.10-3.13, pre-commit config, apache-2.0 license, readme.

local venv at `.venv/` (gitignored) — ruff, mypy, pytest, pytest-cov, numpy, pyarrow, installed via `pip install -e ".[dev]"`. no uv on this machine, just venv + pip. reuse `.venv/Scripts/python.exe -m <tool>` rather than recreating it unless it's actually missing.

repo lives at github.com/bradsward/mekiki, private. ruff check, ruff format --check, mypy --strict, pytest -q all pass, coverage 99% (bar's 90%).

## next

`docs/episode.md` and the `Episode`/`Frame` dataclasses (`src/mekiki/episode.py`) are both done — `ActionDimSpec`/`ActionSpaceSpec`, `Pose`, `Proprioception`, `CameraFrame` (lazy `read()`, never decoded at construction), `Frame`, `EpisodeMetadata`, `Episode` (iterable once, per the streaming rule). Shape/range validation lives in `__post_init__` on `Pose` and `Proprioception`. `tests/conftest.py` has a `make_clean_episode`/`make_clean_frame` factory meant to be reused as the "clean control" fixture once actual detectors (M2+) need one alongside their defect-injected cases.

`Proprioception` got revised same-day: `ee_poses`/`grippers` are now `dict[str, ...]` keyed by end-effector name (so zero, one, or two — bimanual — all work) instead of one required `ee_pose`/`gripper`, `joint_positions` is optional, and there's an `extra: dict[str, NDArray]` escape hatch for state that isn't joint/pose/gripper shaped (PushT's raw 2D position lives there). Reasoning in DECISIONS.md — the deciding factor was that even ALOHA-style bimanual arms broke the old single-`ee_pose` shape, not just non-arm tasks.

`src/mekiki/readers/lerobot.py` now reads a LeRobotDataset directory end to end: `meta/info.json`, the episode index, the action-space fail-loud gate, and `read_episodes()` turning `data/**.parquet` rows into real `Frame`/`Proprioception` objects. State columns (e.g. `observation.state`) are never guessed into joints/ee/gripper — they land in `Proprioception.extra`, keyed by their LeRobotDataset column name, same "don't guess" principle as the action space. Checked twice against real data (`lerobot/pusht`, not committed — network fixture outside the repo): once for metadata/episode-index only, again end-to-end for `read_episodes()` — 206/206 episodes, 25650/25650 frames match `info.total_frames` exactly, first frame's action/state/camera-resolution spot-checked by hand.

camera pixels are the one thing still not real: `Frame.images` correctly reports that a camera exists (name, resolution) but `.read()` raises `NotImplementedError` — real LeRobotDataset cameras are video-encoded (pusht's is av1-in-mp4) and mekiki has no decoder yet. Deliberately deferred rather than half-built; see recommendations.

## open questions / risks

- haven't looked at RLDS/Open X-Embodiment's real on-disk schema yet — no reason to assume it's any closer to what `docs/episode.md` expects than LeRobotDataset turned out to be. check before writing that reader, same way this session checked LeRobotDataset instead of assuming.
- numpy is pinned to `<2.4` (see DECISIONS.md) purely because of a mypy/stub incompatibility — not a real dependency conflict. Remember to reconsider that pin once mypy catches up, so it doesn't quietly linger for years.

## recommendations

ideas noticed in passing, outside whatever the session's actual task was. need an explicit yes/no before anything happens on them — prune once decided so this doesn't pile up.

- **Camera frames in real LeRobotDataset entries are video-encoded** (the pusht cameras are av1-in-mp4), not raw arrays in parquet. Decoding needs a real dependency (something like PyAV) that isn't installed yet. Not blocking anything else in M1 (metadata/state/action all work without it) — flagging so it's a deliberate addition whenever a check first needs actual pixels, rather than something that sneaks in.

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
2026-08-25 · M1 · docs/episode.md written (action space, frames, streaming rules) · next: Episode/Frame dataclasses
2026-08-25 · M1 · Episode/Frame dataclasses + clean-episode test fixture · next: LeRobotDataset reader
2026-08-25 · M1 · LeRobotDataset info/episode-index reader + action-space gate, checked against real lerobot/pusht data · next: resolve Proprioception-shape question (see recommendations), then frame-data reading
2026-08-25 · M1 · generalized Proprioception for bimanual/no-arm state (dict-keyed ee_poses/grippers, extra escape hatch) · next: frame-data parquet reading in the LeRobotDataset reader
2026-08-26 · M1 · LeRobotDataset reader now builds real Frame/Proprioception objects, verified end-to-end against lerobot/pusht (206 episodes, 25650 frames) · next: RLDS/Open X-Embodiment reader, or camera decoding if that becomes urgent first
