# Roadmap

reorder whenever, don't delete anything without a note in [DECISIONS.md](DECISIONS.md).

- [x] M0 — repo skeleton: pyproject, ruff/mypy/pytest, CI, pre-commit, apache-2.0, readme stub. done.
- [ ] M1 — episode abstraction. one in-memory rep for timestamped observations, actions, proprio, camera streams, metadata. readers for LeRobotDataset and RLDS/Open X-Embodiment.
      write `docs/episode.md` first — action space conventions, delta vs absolute, frame handling all get decided here and everything downstream inherits it. pull one small public dataset and round-trip it before moving on.
      - [x] `docs/episode.md`
      - [ ] `Episode` / `Frame` dataclasses, typed, streaming-friendly
      - [ ] LeRobotDataset reader, round-tripped against a small public dataset
      - [ ] RLDS / Open X-Embodiment reader
- [ ] M2 — temporal integrity. dropped frames, non-monotonic timestamps, control freq jitter, camera/proprio desync.
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
