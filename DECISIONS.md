# Decisions

log of calls that aren't obvious from the code, so future-me doesn't redo or re-litigate them.

## 2026-08-25 — M0 tooling

- build backend: hatchling. nothing exotic, no compiled bits, standard `src/` layout.
- numpy + pyarrow only in core deps. scikit-learn goes behind `mekiki[coverage]` once M7 needs density estimation — core stays under six deps, torch never becomes a runtime dep.
- CLI entry point (`mekiki.cli:main`) stubbed now instead of waiting for M5. packaging/entry-point wiring is cheap to get right early and cheap to test; real subcommands still land in M5.
- license: apache-2.0, matches the robotics ecosystem norm, pulled verbatim from apache.org.

<!-- log IP-BOUNDARY here whenever a session drifts toward CI gating, verdicts, or safety-eval territory and gets reverted -->
