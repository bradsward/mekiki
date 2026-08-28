# Roadmap

reorder whenever, don't delete anything without a note in [DECISIONS.md](DECISIONS.md).

- [x] M0 — repo skeleton: pyproject, ruff/mypy/pytest, CI, pre-commit, apache-2.0, readme stub. done.
- [ ] M1 — episode abstraction. one in-memory rep for timestamped observations, actions, proprio, camera streams, metadata. readers for LeRobotDataset and RLDS/Open X-Embodiment.
      write `docs/episode.md` first — action space conventions, delta vs absolute, frame handling all get decided here and everything downstream inherits it. pull one small public dataset and round-trip it before moving on.
      - [x] `docs/episode.md`
      - [x] `Episode` / `Frame` dataclasses, typed, streaming-friendly
      - [x] LeRobotDataset reader — round-tripped against three real datasets across both on-disk layouts the format actually uses:
        - [x] v3.0 (`lerobot/pusht`, 206/206 episodes, 25650/25650 frames match) — shared multi-episode data files, `meta/episodes/**.parquet` index
        - [x] v2.0 (`IPEC-COMMUNITY/bridge_orig_lerobot`, 53192/53192 episodes, 1893026/1893026 frames match, incl. `bridge_v2` — this project's own README example) — one parquet file per episode, `meta/episodes.jsonl` index
        - [x] v2.1 layout-compatibility spot-checked against `IPEC-COMMUNITY/berkeley_cable_routing_lerobot`
        - [x] `meta/info.json` parsing + episode index, for both layouts
        - [x] action-space fail-loud gate: info.json never declares delta-vs-absolute/unit/frame, so a caller-supplied `ActionSpaceSpec` is required and validated against the actual action dimensionality
        - [x] frame-data parquet reading into `Frame`/`Proprioception` — unlabeled state columns go to `Proprioception.extra` rather than being guessed into joints/ee/gripper
        - [x] reject unsupported `codebase_version` loudly instead of misreading
        - [x] `robot_embodiment` defaults to the dataset's own declared `robot_type` (e.g. `"widowx"`) instead of always `"unknown"`, caller can still override
        - [ ] camera pixel decoding — real datasets use video-encoded cameras (e.g. av1); `CameraFrame` correctly represents that a camera exists (name, resolution) but `.read()` raises `NotImplementedError` until a decode dependency is added. deliberately deferred to whenever a check first needs actual pixels (M6 at the latest).
      - [ ] RLDS / Open X-Embodiment reader — real format is TFRecord-wrapped `tensorflow_datasets` serialization, not a simple flat format; genuinely needs `tensorflow`/`tensorflow_datasets` (or an unvalidatable hand-rolled decoder) to read correctly. see `docs/rlds.md` and the recommendation in `STATE.md` before starting this — it may be lower priority than it looks, since most OXE data is already reachable via LeRobotDataset mirrors on the Hub (which raises the v2.x item above in priority instead).
- [ ] M2 — temporal integrity. cheap to implement, immediately useful — first real checks, not just plumbing. every check reports a magnitude + threshold, never a bare pass/fail, and gets a clean-control fixture plus a defect-injected-at-known-magnitude fixture.
      - [x] non-monotonic timestamps — a frame's timestamp doesn't strictly increase from the previous one (duplicate or out-of-order). foundational: jitter/frequency stats downstream assume this holds. `mekiki.checks.temporal.check_timestamp_monotonicity`, verified clean (0 violations) against a real `lerobot/pusht` episode.
      - [ ] control frequency jitter — how much consecutive-frame deltas deviate from the episode's own nominal rate.
      - [ ] dropped frames — a delta much larger than the nominal rate, implying missing frames in between.
      - [ ] camera/proprio desync — gap between a `CameraFrame.timestamp` and its owning `Frame.timestamp`, honoring `timestamp_is_measured` (docs/episode.md) so "no gap measured" isn't confused with "verified in sync."
- [ ] M3 — action-state consistency (the flagship one). forward-integrate commanded actions, diff against recorded proprio, report residuals in real units. `docs/consistency.md` first — tolerance has to scale with control freq and robot type, don't rush it.
- [ ] M4 — idle + segmentation. hesitation, pre-motion dead time, post-task dwell. report recoverable fraction per episode.
- [ ] M5 — report + CLI. `mekiki audit`, `mekiki report`. per-episode scorecards + corpus summary.
- [ ] M6 — visual checks. dead/frozen streams, occlusion, exposure failure, camera drift. classical CV only, no learned models in core.
- [ ] M7 — coverage analysis (the other flagship). embed the state-action space, estimate density, report sparse regions in plain language.
- [ ] M8 — near-duplicate detection. repeated demos, autonomous rollouts mislabeled as human, scripted resets counted as episodes.
- [ ] M9 — filter + export. apply a policy, write filtered LeRobot output, keep full provenance of what got dropped and why.
- [ ] M10 — docs site. quickstart against a real dataset, under 5 min.
- [ ] M11 — public release + corpus audit. pypi + a writeup running mekiki across the major open datasets, numbers published even where unflattering.
- [ ] M12 — training-impact proof. train filtered vs unfiltered on a public benchmark (LIBERO or similar), publish the delta.
