# mekiki

Find the bad demonstrations before you train on them.

Robot learning runs on human teleoperation data, and that data is full of
garbage: operator hesitation recorded as intentional motion, failed grasps
kept as successes, clock skew between camera and proprioception streams,
action labels that don't match what the arm actually did, whole episodes
where a camera died. Nobody finds out until a policy trains for nine hours
and behaves strangely, and even then nobody knows which episodes caused it.

mekiki reads demonstration datasets, runs a battery of physically-grounded
checks, and tells you which episodes are lying to you. That's the target —
here's the shape of it:

```bash
mekiki audit ~/data/bridge_v2/
# 12,847 episodes scanned
#   1,203 (9.4%)  action-state mismatch beyond tolerance
#     892 (6.9%)  >40% idle frames
#     311 (2.4%)  camera stream desync >100ms
#      67 (0.5%)  wrist camera fully occluded
# → 2,301 episodes flagged. mekiki report --open

mekiki coverage ~/data/bridge_v2/
# sparse region: gripper-closed approach angles beyond 35° from vertical
# → 41 episodes covering 8% of reachable workspace. collect here next.
```

This CLI doesn't exist yet — see [Status](#status) for what does.

## Status

Pre-alpha. The `audit`/`coverage`/`report` commands above are the target
interface (landing in M5); right now mekiki is a Python library. What
actually works today:

```python
from pathlib import Path
from mekiki.episode import ActionDimSpec
from mekiki.readers.lerobot import read_episodes
from mekiki.checks.temporal import check_timestamp_monotonicity

action_space = (
    ActionDimSpec("x", "delta", "m", "ee"),
    ActionDimSpec("y", "delta", "m", "ee"),
)
for episode in read_episodes(Path("~/data/pusht").expanduser(), action_space):
    result = check_timestamp_monotonicity(episode)
    if result.violation_indices:
        print(episode.metadata.episode_id, result.violation_indices)
```

Reads real LeRobotDataset directories (v2.0/v2.1/v3.0 layouts) into a typed
`Episode`/`Frame` model, with one temporal-integrity check so far. See
[ROADMAP.md](ROADMAP.md) for what's built and what's next, and
[STATE.md](STATE.md) for the running log.

## Scope

mekiki is a **pre-training data quality layer**: read demonstrations, detect
corruption and low-value data, quantify coverage, filter, export.

mekiki is **not** a training framework, a simulator, a teleoperation tool, a
dataset host, or an annotation platform. It never gates CI, never enforces a
spec, and never issues pass/fail verdicts on robot behavior — it reports,
humans decide.

## Install

```bash
pip install -e ".[dev]"
```

## Development

```bash
ruff check .
ruff format --check .
mypy --strict src/
pytest -q
```

## License

Apache-2.0
