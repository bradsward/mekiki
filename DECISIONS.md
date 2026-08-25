# Decisions

Append-only log of non-obvious calls: scope boundaries, rejected approaches,
and anything a future session would otherwise redo or re-litigate.

## 2026-08-25 — M0 tooling choices

- **Build backend: hatchling.** No compiled extensions, nothing exotic;
  hatchling is a light, standard choice for a `src/` layout package.
- **`numpy` and `pyarrow` as core runtime deps; `scikit-learn` behind
  `mekiki[coverage]`.** Per the project brief, core stays under six
  dependencies and torch/heavy ML never becomes a runtime requirement.
  Coverage analysis (M7) is the first place density estimation is likely to
  want sklearn — deferred there as an extra, not core.
- **CLI entry point (`mekiki.cli:main`) stubbed in M0, not deferred to M5.**
  Packaging and entry-point wiring are cheap to get right early and cheap to
  regression-test; the actual subcommands (`audit`, `report`, `coverage`)
  still land in M5 as planned.
- **License: Apache-2.0**, matching the robotics ecosystem norm, fetched
  verbatim from apache.org rather than paraphrased.

<!-- Log IP-BOUNDARY entries here whenever a session drifts toward CI
     gating, verdicts, or safety-evaluation territory and gets reverted. -->
