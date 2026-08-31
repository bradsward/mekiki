# State

no memory between sessions — this file plus ROADMAP.md and DECISIONS.md is it.

## where things are

M0's done. skeleton's in: pyproject (hatchling, ruff, mypy strict, pytest config), `src/mekiki` with a stub CLI (`--version` only, real subcommands land in M5), tests passing, CI matrix on 3.10-3.13, pre-commit config, apache-2.0 license, readme.

local venv at `.venv/` (gitignored) — ruff, mypy, pytest, pytest-cov, numpy, pyarrow, installed via `pip install -e ".[dev]"`. no uv on this machine, just venv + pip. reuse `.venv/Scripts/python.exe -m <tool>` rather than recreating it unless it's actually missing.

repo lives at github.com/bradsward/mekiki, private. ruff check, ruff format --check, mypy --strict, pytest -q all pass, coverage 99% (bar's 90%).

## next

`docs/episode.md` and the `Episode`/`Frame` dataclasses (`src/mekiki/episode.py`) are both done — `ActionDimSpec`/`ActionSpaceSpec`, `Pose`, `Proprioception`, `CameraFrame` (lazy `read()`, never decoded at construction), `Frame`, `EpisodeMetadata`, `Episode` (iterable once, per the streaming rule). Shape/range validation lives in `__post_init__` on `Pose` and `Proprioception`. `tests/conftest.py` has a `make_clean_episode`/`make_clean_frame` factory meant to be reused as the "clean control" fixture once actual detectors (M2+) need one alongside their defect-injected cases.

`Proprioception` got revised same-day: `ee_poses`/`grippers` are now `dict[str, ...]` keyed by end-effector name (so zero, one, or two — bimanual — all work) instead of one required `ee_pose`/`gripper`, `joint_positions` is optional, and there's an `extra: dict[str, NDArray]` escape hatch for state that isn't joint/pose/gripper shaped (PushT's raw 2D position lives there). Reasoning in DECISIONS.md — the deciding factor was that even ALOHA-style bimanual arms broke the old single-`ee_pose` shape, not just non-arm tasks.

`src/mekiki/readers/lerobot.py` now reads a LeRobotDataset directory end to end, across **both** on-disk layouts the format actually uses in the wild — not just v3.0. State columns (e.g. `observation.state`) are never guessed into joints/ee/gripper — they land in `Proprioception.extra`, keyed by their LeRobotDataset column name, same "don't guess" principle as the action space. Verified against three real datasets (not committed — network fixtures outside the repo):

- v3.0, `lerobot/pusht`: 206/206 episodes, 25650/25650 frames match `info.total_frames`.
- v2.0, `IPEC-COMMUNITY/bridge_orig_lerobot` (this project's own README's `bridge_v2` example): all 53192/53192 episodes and 1893026/1893026 frames match, first two episodes' actual frame data spot-checked by hand (actions, 4 camera keys, language instructions, `robot_embodiment` correctly picked up as `"widowx"` from the dataset's own `robot_type` rather than defaulting to `"unknown"`).
- v2.1, `IPEC-COMMUNITY/berkeley_cable_routing_lerobot`: confirmed layout-compatible with v2.0 for what this reader touches.

decided to do v2.x first (done above), keep native RLDS on the roadmap but after it — reasoning and the reader-internals design tradeoffs are in DECISIONS.md.

camera pixels are still the one thing not real across either layout: `Frame.images` correctly reports that a camera exists (name, resolution) but `.read()` raises `NotImplementedError` — real cameras are video-encoded (av1) and mekiki has no decoder yet. Still deliberately deferred; see parked.

M2 (temporal integrity) started: split it into four sub-checks in `ROADMAP.md` (non-monotonic timestamps, control frequency jitter, dropped frames, camera/proprio desync) since it wasn't broken down before. First one done: `src/mekiki/checks/temporal.py`'s `check_timestamp_monotonicity` — streams an episode once (doesn't materialize all frames), reports `min_delta_seconds` against a `threshold_seconds` plus which frame indices violated it, never a bare pass/fail. Tested against both a clean control (`make_clean_episode`) and precisely-injected defects (duplicate timestamp → 0.0 delta, out-of-order → negative delta, sub-threshold gap only caught with a stricter threshold), plus checked against a real `lerobot/pusht` episode (161 frames, 0 violations, ~0.1s deltas matching its 10Hz rate).

## open questions / risks

- numpy is pinned to `<2.4` (see DECISIONS.md) purely because of a mypy/stub incompatibility — not a real dependency conflict. Remember to reconsider that pin once mypy catches up, so it doesn't quietly linger for years.

## parked

ideas noticed in passing, outside whatever the session's actual task was. not deciding these mid-task — come back and actually think them through before acting, prune once resolved so this doesn't pile up.

- **Camera frames in real LeRobotDataset entries are video-encoded** (av1-in-mp4 in every dataset checked so far), not raw arrays in parquet. Decoding needs a real dependency (something like PyAV) that isn't installed yet. Not blocking anything else in M1 (metadata/state/action all work without it) — a deliberate addition whenever a check first needs actual pixels, not something to reach for casually.
- **RLDS-native decoding needs `tensorflow`/`tensorflow_datasets` — a dependency in the same weight class as the torch dependency this project already refuses to make core.** Full findings in `docs/rlds.md`: the format is TFRecord-wrapped TFDS serialization, nested/sequence-encoded in a way that isn't independently documented — reading it correctly without TF itself means hand-rolling a decoder with nothing to validate it against, which is a bad idea for a tool whose whole job is catching silent data corruption. Priority's settled (v2.x support went first) but not the dependency question — if/when RLDS-native gets picked up, the sane path is a `mekiki[rlds]` extra (mirroring `mekiki[coverage]`) that pulls in `tensorflow_datasets`, never core. Still not sure it's worth it at all given how much OXE data the LeRobotDataset mirrors already cover.

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
2026-08-25 · M1 · docs/episode.md written (action space, frames, streaming rules) · next: Episode/Frame dataclasses
2026-08-25 · M1 · Episode/Frame dataclasses + clean-episode test fixture · next: LeRobotDataset reader
2026-08-25 · M1 · LeRobotDataset info/episode-index reader + action-space gate, checked against real lerobot/pusht data · next: resolve Proprioception-shape question (see recommendations), then frame-data reading
2026-08-25 · M1 · generalized Proprioception for bimanual/no-arm state (dict-keyed ee_poses/grippers, extra escape hatch) · next: frame-data parquet reading in the LeRobotDataset reader
2026-08-26 · M1 · LeRobotDataset reader now builds real Frame/Proprioception objects, verified end-to-end against lerobot/pusht (206 episodes, 25650 frames) · next: RLDS/Open X-Embodiment reader, or camera decoding if that becomes urgent first
2026-08-26 · M1 · researched real RLDS/OXE format (docs/rlds.md), added codebase_version guard after finding real data (bridge_orig_lerobot) is LeRobotDataset v2.x not v3.0 · next: decide RLDS-native priority/dependency, then LeRobotDataset v2.x support
2026-08-26 · M1 · LeRobotDataset v2.0/v2.1 support added (decided this goes before native RLDS), verified against real bridge_orig_lerobot (53192 episodes, 1893026 frames match) and berkeley_cable_routing_lerobot (v2.1) · next: RLDS-native reader (still haven't committed to the TF dependency) or camera decoding, whichever becomes relevant first
2026-08-27 · M2 · split M2 into 4 sub-checks, implemented the first (timestamp monotonicity), verified against real lerobot/pusht data · next: control frequency jitter or dropped frames
2026-08-27 · infra · pulled the nightly scheduled task — every run since it was set up died silently at a broken sleep step and never actually did anything, so it wasn't earning its keep. everything in this repo so far is from working sessions directly, not unattended runs. revisit with a simpler design later.
