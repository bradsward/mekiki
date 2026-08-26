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

<!-- log IP-BOUNDARY here whenever a session drifts toward CI gating, verdicts, or safety-eval territory and gets reverted -->
