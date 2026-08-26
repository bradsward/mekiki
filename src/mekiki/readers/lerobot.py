"""Reader for the LeRobotDataset on-disk format.

Targets the v3.0 layout used by current LeRobot Hub datasets: per-dataset
``meta/info.json``, an episode index at ``meta/episodes/chunk-*/file-*.parquet``,
and frame data at ``data/chunk-*/file-*.parquet`` — one data file commonly
holds many episodes' rows concatenated, distinguished by a global ``index``
column and an ``episode_index`` column. Schema confirmed against a real
dataset (``lerobot/pusht``) rather than assumed from documentation, since
LeRobotDataset has changed on-disk layout across versions before.

This module currently covers metadata and episode discovery only — enough
to validate a dataset against ``docs/episode.md``'s action-space rule before
any frame is read. It stops short of building ``Frame``/``Proprioception``
objects; see ``STATE.md`` for why (real datasets' state layouts don't all
fit the current ``Proprioception`` shape, and that needs a decision before
more code is built on top of it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from mekiki.episode import ActionSpaceSpec

#: LeRobotDataset format versions this reader has been checked against.
#: A dataset reporting a different ``codebase_version`` may still work, but
#: hasn't been verified — see `read_info`.
SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)


@dataclass(frozen=True, slots=True)
class LeRobotInfo:
    """Parsed ``meta/info.json``.

    Attributes:
        codebase_version: LeRobotDataset format version, e.g. ``"v3.0"``.
        fps: Nominal control frequency in Hz.
        total_episodes: Number of episodes in the dataset.
        total_frames: Number of frames across all episodes.
        data_path: Template for locating frame-data parquet files, e.g.
            ``"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"``.
        video_path: Template for locating per-camera video files.
        features: Raw ``"features"`` mapping from info.json, keyed by
            feature name (``"observation.state"``, ``"action"``, camera
            keys, ...). Kept as the source dict rather than re-typed, since
            its shape varies by dataset and only a couple of fields
            (``action``'s length) are actually consumed here.
        action_dims: Number of dimensions in the ``"action"`` feature.
            info.json declares this feature's shape but never its
            per-dimension semantics (delta vs. absolute, unit, frame) — see
            `validate_action_space`.
    """

    codebase_version: str
    fps: float
    total_episodes: int
    total_frames: int
    data_path: str
    video_path: str
    features: dict[str, dict[str, Any]]
    action_dims: int


def read_info(dataset_dir: Path) -> LeRobotInfo:
    """Parse ``<dataset_dir>/meta/info.json``.

    Args:
        dataset_dir: Root of a local LeRobotDataset directory (containing
            ``meta/``, ``data/``, and usually ``videos/``).

    Returns:
        The parsed info file.

    Raises:
        FileNotFoundError: ``meta/info.json`` doesn't exist under
            ``dataset_dir``.
        ValueError: The file exists but is missing the ``"action"``
            feature, or that feature has no usable shape — this reader has
            nothing to validate an action space against in that case.

    Example:
        >>> from pathlib import Path
        >>> info = read_info(Path("~/data/pusht").expanduser())  # doctest: +SKIP
        >>> info.fps
        10.0
    """
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"no meta/info.json under {dataset_dir} — is this a LeRobotDataset directory?"
        )
    raw: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
    features: dict[str, dict[str, Any]] = raw["features"]
    action_feature = features.get("action")
    if action_feature is None or "shape" not in action_feature:
        raise ValueError(
            f"{info_path} has no usable 'action' feature — cannot validate an "
            "action space against it"
        )
    action_shape = action_feature["shape"]
    action_dims = int(action_shape[0]) if action_shape else 0
    return LeRobotInfo(
        codebase_version=raw["codebase_version"],
        fps=float(raw["fps"]),
        total_episodes=int(raw["total_episodes"]),
        total_frames=int(raw["total_frames"]),
        data_path=raw["data_path"],
        video_path=raw["video_path"],
        features=features,
        action_dims=action_dims,
    )


def validate_action_space(info: LeRobotInfo, action_space: ActionSpaceSpec | None) -> None:
    """Check a caller-supplied action space against the dataset's shape.

    ``meta/info.json`` records the ``"action"`` feature's dimensionality
    but never whether each dimension is a delta or an absolute target, its
    physical unit, or its coordinate frame — that semantic knowledge isn't
    present anywhere in the LeRobotDataset format and has to come from the
    caller (the dataset's card, paper, or collection code). Per
    docs/episode.md, a reader that can't determine the action space must
    fail loudly rather than default to a guess — this function is that
    gate. It does not and cannot infer ``action_space`` itself.

    Args:
        info: Parsed dataset info, from `read_info`.
        action_space: The action space the caller asserts this dataset
            uses, or ``None`` if they haven't supplied one.

    Raises:
        ValueError: ``action_space`` is ``None``, empty, or doesn't have
            exactly as many dimensions as the dataset's ``"action"``
            feature.

    Example:
        >>> from mekiki.episode import ActionDimSpec
        >>> info = LeRobotInfo(
        ...     codebase_version="v3.0", fps=10.0, total_episodes=1,
        ...     total_frames=1, data_path="", video_path="", features={},
        ...     action_dims=2,
        ... )
        >>> validate_action_space(info, None)
        Traceback (most recent call last):
            ...
        ValueError: no action_space supplied for a LeRobotDataset — meta/info.json ...
    """
    if not action_space:
        raise ValueError(
            "no action_space supplied for a LeRobotDataset — meta/info.json declares "
            f"{info.action_dims} action dimension(s) but never their delta-vs-absolute "
            "mode, unit, or frame. Supply an ActionSpaceSpec derived from this "
            "dataset's documentation rather than guessing (docs/episode.md)."
        )
    if len(action_space) != info.action_dims:
        raise ValueError(
            f"action_space has {len(action_space)} dimension(s) but this dataset's "
            f"'action' feature has {info.action_dims} — they must match exactly."
        )


@dataclass(frozen=True, slots=True)
class LeRobotEpisodeRef:
    """One row of ``meta/episodes/**.parquet``: where to find one episode.

    Attributes:
        episode_index: Index of this episode within the dataset.
        data_chunk_index: Chunk component of the parquet file holding this
            episode's frame data (see `LeRobotInfo.data_path`).
        data_file_index: File component of that same parquet file.
        dataset_from_index: First row (inclusive) of this episode's frames
            in that file's global ``index`` column.
        dataset_to_index: Last row (exclusive) of this episode's frames.
        length: Number of frames in this episode
            (``dataset_to_index - dataset_from_index``).
        tasks: Language instruction(s) recorded for this episode.
    """

    episode_index: int
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    length: int
    tasks: tuple[str, ...]


def read_episode_refs(dataset_dir: Path) -> list[LeRobotEpisodeRef]:
    """Read the episode index from every ``meta/episodes/**.parquet`` file.

    Args:
        dataset_dir: Root of a local LeRobotDataset directory.

    Returns:
        One `LeRobotEpisodeRef` per episode, ordered by ``episode_index``.
        A dataset's episode index can be split across several parquet
        files (mirroring how ``meta/episodes/`` is itself chunked); all of
        them are read and merged.

    Raises:
        FileNotFoundError: no parquet files found under
            ``<dataset_dir>/meta/episodes``.
    """
    episodes_dir = dataset_dir / "meta" / "episodes"
    files = sorted(episodes_dir.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode index parquet files under {episodes_dir}")

    refs: list[LeRobotEpisodeRef] = []
    for path in files:
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "length",
                "tasks",
            ],
        )
        for row in table.to_pylist():
            refs.append(
                LeRobotEpisodeRef(
                    episode_index=row["episode_index"],
                    data_chunk_index=row["data/chunk_index"],
                    data_file_index=row["data/file_index"],
                    dataset_from_index=row["dataset_from_index"],
                    dataset_to_index=row["dataset_to_index"],
                    length=row["length"],
                    tasks=tuple(row["tasks"]),
                )
            )
    refs.sort(key=lambda ref: ref.episode_index)
    return refs
