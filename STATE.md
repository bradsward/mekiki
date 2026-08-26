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

checked RLDS/Open X-Embodiment's real format before writing a reader for it (per the open question below) — findings written up properly in `docs/rlds.md` rather than just here, since it's real research worth keeping. Short version: it's genuinely more work and a heavier dependency than LeRobotDataset was. Two things came out of that check that change next steps — both below.

## open questions / risks

- numpy is pinned to `<2.4` (see DECISIONS.md) purely because of a mypy/stub incompatibility — not a real dependency conflict. Remember to reconsider that pin once mypy catches up, so it doesn't quietly linger for years.

## recommendations

ideas noticed in passing, outside whatever the session's actual task was. need an explicit yes/no before anything happens on them — prune once decided so this doesn't pile up.

- **Camera frames in real LeRobotDataset entries are video-encoded** (the pusht cameras are av1-in-mp4), not raw arrays in parquet. Decoding needs a real dependency (something like PyAV) that isn't installed yet. Not blocking anything else in M1 (metadata/state/action all work without it) — flagging so it's a deliberate addition whenever a check first needs actual pixels, rather than something that sneaks in.
- **RLDS-native decoding needs `tensorflow`/`tensorflow_datasets` — a dependency in the same weight class as the torch dependency this project already refuses to make core.** Full findings in `docs/rlds.md`: the format is TFRecord-wrapped TFDS serialization, nested/sequence-encoded in a way that isn't independently documented — reading it correctly without TF itself means hand-rolling a decoder with nothing to validate it against, which is a bad idea for a tool whose whole job is catching silent data corruption. If RLDS-native support happens, the sane path is a `mekiki[rlds]` extra (mirroring `mekiki[coverage]`) that pulls in `tensorflow_datasets`, never core. Want your sign-off on that dependency before it lands, same as the Proprioception call.
- **Open X-Embodiment may not need the native RLDS reader at all, or at least not soon.** Most/all OXE datasets already have community LeRobotDataset conversions on the Hugging Face Hub — including `IPEC-COMMUNITY/bridge_orig_lerobot`, which is literally this project's own README's `bridge_v2` example. If that's an acceptable path to real OXE data, RLDS-native work could drop in priority (or off the board entirely) in favor of the LeRobotDataset v2.x item below, which reaches the same data with no new heavy dependency. Reordered `ROADMAP.md` to put v2.x support ahead of the RLDS reader on that basis, since reordering is pre-authorized — but skipping RLDS-native entirely is a bigger call than a reorder and still needs your word.
- **LeRobotDataset v2.x (a different on-disk layout, not v3.0) is real and common** — checked several Hub mirrors, `v2.0`/`v2.1` outnumbered `v3.0` among the ones sampled, including `bridge_orig_lerobot`. The reader now fails loudly and clearly on it (`read_info` checks `codebase_version`, verified against the real file) rather than the confusing generic error it would've thrown before — but nothing about actually *reading* v2.x is built yet. That's the concrete next coding task on the reader, not a recommendation; added to `ROADMAP.md`.

## log

2026-08-25 · M0 · repo skeleton, tooling, CI, license, readme · next: docs/episode.md then Episode dataclasses + LeRobot reader (M1)
2026-08-25 · infra · pushed to github (private), nightly build session scheduled · next: M1
2026-08-25 · M1 · docs/episode.md written (action space, frames, streaming rules) · next: Episode/Frame dataclasses
2026-08-25 · M1 · Episode/Frame dataclasses + clean-episode test fixture · next: LeRobotDataset reader
2026-08-25 · M1 · LeRobotDataset info/episode-index reader + action-space gate, checked against real lerobot/pusht data · next: resolve Proprioception-shape question (see recommendations), then frame-data reading
2026-08-25 · M1 · generalized Proprioception for bimanual/no-arm state (dict-keyed ee_poses/grippers, extra escape hatch) · next: frame-data parquet reading in the LeRobotDataset reader
2026-08-26 · M1 · LeRobotDataset reader now builds real Frame/Proprioception objects, verified end-to-end against lerobot/pusht (206 episodes, 25650 frames) · next: RLDS/Open X-Embodiment reader, or camera decoding if that becomes urgent first
2026-08-26 · M1 · researched real RLDS/OXE format (docs/rlds.md), added codebase_version guard after finding real data (bridge_orig_lerobot) is LeRobotDataset v2.x not v3.0 · next: your call on the RLDS-native/dependency recommendations, then LeRobotDataset v2.x support
