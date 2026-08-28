# Decisions

log of calls that aren't obvious from the code, so future-me doesn't redo or re-litigate them.

## 2026-08-25 — M0 tooling

- build backend: hatchling. nothing exotic, no compiled bits, standard `src/` layout.
- numpy + pyarrow only in core deps. scikit-learn goes behind `mekiki[coverage]` once M7 needs density estimation — core stays under six deps, torch never becomes a runtime dep.
- CLI entry point (`mekiki.cli:main`) stubbed now instead of waiting for M5. packaging/entry-point wiring is cheap to get right early and cheap to test; real subcommands still land in M5.
- license: apache-2.0, matches the robotics ecosystem norm, pulled verbatim from apache.org.

## 2026-08-25 — numpy pinned below 2.4

`numpy>=2.4` ships stub files using PEP 695 `type` statements, which mypy 2.3.1 (latest available) can't parse when `python_version` is set below 3.12 — `mypy --strict` fails on `numpy/__init__.pyi` itself, not on our code. Pinned `numpy<2.4` in `pyproject.toml` until either mypy fixes this or we're ready to drop 3.10/3.11 support and bump the project's own `python_version` target. Revisit this pin periodically rather than forgetting it's there.

## 2026-08-25 — pyarrow untyped under mypy strict

pyarrow ships no `py.typed` marker and no inline types, so `mypy --strict` fails on the import itself, not on our code. `pyarrow-stubs` exists but its published versions (up to 20.x) trail our installed pyarrow (25.x) enough that pulling it in seemed like its own maintenance burden for a package we use narrowly. Added a scoped override (`[[tool.mypy.overrides]]` for `pyarrow.*`, `ignore_missing_imports = true`) instead — our own code touching pyarrow still gets checked, just not pyarrow's internals.

## 2026-08-25 — Proprioception generalized past single-arm-with-gripper

The original `Proprioception` (joint_positions + one ee_pose + one gripper, all required) broke on real data faster than expected — not just on non-arm tasks like PushT, but on bimanual arms too (ALOHA-style setups have two end-effectors and two grippers, which is mainstream in Open X-Embodiment, not an edge case). A rigid single-arm shape would've needed redesigning again the moment a bimanual dataset showed up.

Went with: every field optional, `ee_poses`/`grippers` keyed by short names (`"ee"`, or `"left"`/`"right"`) so the same type covers zero/one/two end-effectors, plus an `extra: dict[str, NDArray]` escape hatch for state that isn't joints/pose/gripper shaped at all (PushT's raw 2D position lives there). `extra` carries no assumed unit or frame — it's explicitly the "we don't know what this is" bucket, not a default path. `docs/episode.md` and `src/mekiki/episode.py` both updated together so they don't drift.

## 2026-08-26 — LeRobotDataset reader now rejects unsupported codebase_version

Checked real RLDS/OXE data before writing a reader for it (see `docs/rlds.md`) and found, along the way, that `IPEC-COMMUNITY/bridge_orig_lerobot` — the actual Bridge V2 dataset this project's README uses as its example — is LeRobotDataset `codebase_version: "v2.0"`, not `"v3.0"` like the only dataset the reader had been checked against (`lerobot/pusht`). v2.x uses a different on-disk layout entirely (`meta/episodes.jsonl`, not `meta/episodes/**.parquet`). The reader would have failed on it, but with a generic "no episode index parquet files" error that doesn't say why — confusing, not wrong, but not the fail-loudly-with-a-clear-reason standard the rest of this reader holds itself to. Added an explicit `codebase_version` check in `read_info` that names the real problem and points at `docs/rlds.md`. Verified against the real `bridge_orig_lerobot` info.json, not just a synthetic fixture.

## 2026-08-26 — LeRobotDataset v2.0/v2.1 support added, not just detected

Per the owner's call (prioritize this over the native RLDS/TFRecord reader, since it needs no heavy dependency and unblocks real data immediately): implemented the v2.x layout on top of the v2/v3 rejection guard from the previous entry, rather than just leaving it rejected.

Design: `LeRobotEpisodeRef` gained a resolved `data_relative_path` (computed once from `LeRobotInfo.data_path`'s version-specific template — v3.0 and v2.x use different placeholder names, `chunk_index`/`file_index` vs. `episode_chunk`/`episode_index`) and made `dataset_from_index`/`dataset_to_index` optional — `None` for v2.x, where each episode owns its whole file and no row-range slicing is needed. `read_episode_refs` dispatches on `codebase_version` to a v2 (`meta/episodes.jsonl`, line-delimited JSON) or v3 (`meta/episodes/**.parquet`) implementation; `read_episodes` itself needed no branching once refs carry a resolved path and an optional row range.

Verified against real data, not just the synthetic fixture: `IPEC-COMMUNITY/bridge_orig_lerobot` (v2.0) — all 53,192 episodes and 1,893,026 frames match `info.json`'s own totals exactly, first two episodes' frame data spot-checked by hand (action vectors, camera keys, language instructions all correct). `IPEC-COMMUNITY/berkeley_cable_routing_lerobot` (v2.1) confirmed layout-compatible with v2.0 for what this reader touches, so both versions are supported rather than just v2.0.

Small side improvement while in there: `read_episodes`'s `robot_embodiment` now defaults to the dataset's own declared `info.robot_type` (e.g. `bridge_orig`'s real value, `"widowx"`) instead of always `"unknown"` regardless of what the file actually says — a caller can still override it. Low-risk since, unlike action-space semantics, a wrong robot label doesn't corrupt any computed check, it's just a name.

<!-- log IP-BOUNDARY here whenever a session drifts toward CI gating, verdicts, or safety-eval territory and gets reverted -->
