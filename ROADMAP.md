# Roadmap

Reorder as we learn. Delete nothing without a note in [DECISIONS.md](DECISIONS.md).

- [x] **M0 — Foundation.** Repo skeleton, `pyproject.toml`, ruff + mypy +
      pytest, GitHub Actions, pre-commit, Apache-2.0 license, README stub.
- [ ] **M1 — Episode abstraction.** A common in-memory representation:
      timestamped observations, actions, proprioception, camera streams,
      metadata. Readers for LeRobotDataset and RLDS/Open X-Embodiment. Write
      `docs/episode.md` before any code — get action space conventions,
      delta-vs-absolute, and frame handling right here, because everything
      downstream inherits these choices. Pull one small public dataset and
      make it round-trip before moving on.
      - [ ] `docs/episode.md`: action space conventions, delta-vs-absolute,
            frame/coordinate handling.
      - [ ] `Episode` / `Frame` core dataclasses (typed, immutable where
            practical), streaming-friendly.
      - [ ] LeRobotDataset reader, round-tripped against one small public
            dataset.
      - [ ] RLDS / Open X-Embodiment reader.
- [ ] **M2 — Temporal integrity.** Dropped frames, non-monotonic timestamps,
      control frequency jitter, cross-stream desync between cameras and
      proprioception.
- [ ] **M3 — Action-state consistency (flagship).** Forward-integrate
      commanded actions, compare against recorded proprioception, report
      residuals in physical units. Write `docs/consistency.md` first,
      including how tolerance scales with control frequency and robot type.
- [ ] **M4 — Idle and segmentation.** Operator hesitation, pre-motion dead
      time, post-task dwell. Report recoverable fraction per episode.
- [ ] **M5 — Report and CLI.** `mekiki audit`, `mekiki report`. Per-episode
      scorecards plus corpus-level summary.
- [ ] **M6 — Visual checks.** Dead/frozen streams, workspace occlusion,
      exposure failure, camera pose drift. Classical CV only in core.
- [ ] **M7 — Coverage analysis (flagship).** Embed the state-action space,
      estimate density, report sparse regions in actionable terms.
- [ ] **M8 — Near-duplicate detection.** Repeated demos, autonomous rollouts
      mislabeled as human demos, scripted resets counted as episodes.
- [ ] **M9 — Filter and export.** Apply a policy, write filtered LeRobot
      output with full provenance of what was dropped and why.
- [ ] **M10 — Docs site.** Quickstart against a real public dataset, under
      five minutes.
- [ ] **M11 — Public release and the corpus audit.** PyPI + writeup: run
      mekiki across major open datasets, publish the numbers, invite
      correction.
- [ ] **M12 — Training-impact proof (capstone).** Train filtered vs.
      unfiltered on a public benchmark (LIBERO or similar), publish the delta.
