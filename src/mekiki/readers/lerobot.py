"""Reader for the LeRobotDataset on-disk format.

Targets the v3.0 layout used by current LeRobot Hub datasets: per-dataset
``meta/info.json``, an episode index at ``meta/episodes/chunk-*/file-*.parquet``,
and frame data at ``data/chunk-*/file-*.parquet`` — one data file commonly
holds many episodes' rows concatenated, distinguished by a global ``index``
column and an ``episode_index`` column. Schema confirmed against a real
dataset (``lerobot/pusht``) rather than assumed from documentation, since
LeRobotDataset has changed on-disk layout across versions before.

``read_episodes`` builds real ``Frame``/``Proprioception`` objects from the
frame-data parquet files. It deliberately does not guess which columns are
joint positions, an end-effector pose, or a gripper — ``meta/info.json``
never says, the same way it never says whether ``action`` is a delta or an
absolute target. Every non-special numeric column ends up in
``Proprioception.extra`` instead, keyed by its LeRobotDataset feature name.
Camera frames backed by encoded video are represented (so a check can see a
camera exists and read its declared resolution) but decoding is not
implemented yet — ``CameraFrame.read()`` raises `NotImplementedError` for
those; see the camera-decoding recommendation in ``STATE.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from mekiki.episode import (
    ActionSpaceSpec,
    CameraFrame,
    Episode,
    EpisodeMetadata,
    Frame,
    Proprioception,
)

#: Data-file columns with fixed meaning, excluded from `Proprioception.extra`.
_KNOWN_COLUMNS = frozenset(
    {"action", "episode_index", "frame_index", "timestamp", "index", "task_index"}
)

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


def _video_camera_features(info: LeRobotInfo) -> dict[str, dict[str, Any]]:
    """Feature entries whose pixels live outside the data parquet files."""
    return {
        name: feature
        for name, feature in info.features.items()
        if feature.get("dtype") in ("video", "image")
    }


def _make_undecoded_reader(camera_name: str) -> Callable[[], NDArray[np.uint8]]:
    """Build a `CameraFrame.read` that fails loudly instead of decoding.

    A plain closure over the loop variable would be wrong (`name=name`
    default-arg tricks aside, mypy also can't infer an unannotated lambda's
    type) — a small named function keeps both correct and typed.
    """

    def read() -> NDArray[np.uint8]:
        raise NotImplementedError(
            f"decoding camera {camera_name!r} isn't implemented yet — LeRobotDataset "
            "stores it as encoded video, and mekiki doesn't have a video decoder "
            "wired up (see STATE.md recommendations). Frame.images[...] still "
            "carries resolution/timestamp metadata; only pixel access is unavailable."
        )

    return read


def read_episodes(
    dataset_dir: Path,
    action_space: ActionSpaceSpec,
    *,
    dataset_name: str | None = None,
    robot_embodiment: str = "unknown",
) -> Iterator[Episode]:
    """Read every episode in a LeRobotDataset directory.

    Streams one episode at a time (per docs/episode.md); within an episode,
    proprioception/action/timestamp are read eagerly (cheap — a few floats
    per frame) but camera pixels are never decoded here.

    State columns (e.g. ``observation.state``) are never assumed to be
    joint positions, an end-effector pose, or a gripper — LeRobotDataset
    doesn't declare that semantic mapping any more than it declares
    action-space semantics, so guessing it would violate the same
    fail-loudly-don't-guess rule `validate_action_space` already enforces
    for actions. Every such column lands in
    ``Frame.proprioception.extra[column_name]`` instead, as a raw
    ``float64`` array with no assumed unit or frame.

    Args:
        dataset_dir: Root of a local LeRobotDataset directory.
        action_space: Action space for this dataset — validated via
            `validate_action_space` before any frame is read.
        dataset_name: Value for `EpisodeMetadata.dataset_name`. Defaults to
            ``dataset_dir.name``.
        robot_embodiment: Value for `EpisodeMetadata.robot_embodiment`.
            LeRobotDataset's ``info.json`` carries a ``robot_type`` field in
            practice but it isn't guaranteed present or meaningful (real
            data has been seen with it set to ``"unknown"``), so it's a
            caller-supplied override rather than something this reader
            trusts blindly.

    Yields:
        One `Episode` per episode in the dataset, in ``episode_index`` order.

    Raises:
        ValueError: `action_space` doesn't match the dataset (see
            `validate_action_space`).
        FileNotFoundError: metadata or a referenced data file is missing.

    Example:
        >>> from pathlib import Path
        >>> from mekiki.episode import ActionDimSpec
        >>> action_space = (
        ...     ActionDimSpec("x", "absolute", "normalized", "unknown"),
        ...     ActionDimSpec("y", "absolute", "normalized", "unknown"),
        ... )
        >>> episodes = read_episodes(
        ...     Path("~/data/pusht").expanduser(), action_space
        ... )  # doctest: +SKIP
        >>> first = next(episodes)  # doctest: +SKIP
    """
    info = read_info(dataset_dir)
    validate_action_space(info, action_space)
    resolved_dataset_name = dataset_name if dataset_name is not None else dataset_dir.name
    camera_features = _video_camera_features(info)
    refs = read_episode_refs(dataset_dir)

    data_tables: dict[tuple[int, int], pa.Table] = {}
    for ref in refs:
        key = (ref.data_chunk_index, ref.data_file_index)
        if key not in data_tables:
            rel_path = info.data_path.format(
                chunk_index=ref.data_chunk_index, file_index=ref.data_file_index
            )
            data_tables[key] = pq.read_table(dataset_dir / rel_path)
        table = data_tables[key]
        mask = pc.and_(
            pc.greater_equal(table["index"], ref.dataset_from_index),
            pc.less(table["index"], ref.dataset_to_index),
        )
        rows = table.filter(mask).to_pylist()
        rows.sort(key=lambda row: row["index"])

        extra_columns = [
            c for c in table.column_names if c not in _KNOWN_COLUMNS and c not in camera_features
        ]
        language_instruction = ref.tasks[0] if ref.tasks else None

        frames = [
            _row_to_frame(
                row,
                is_first=(i == 0),
                is_last=(i == len(rows) - 1),
                extra_columns=extra_columns,
                camera_features=camera_features,
                language_instruction=language_instruction,
            )
            for i, row in enumerate(rows)
        ]
        metadata = EpisodeMetadata(
            episode_id=str(ref.episode_index),
            dataset_name=resolved_dataset_name,
            robot_embodiment=robot_embodiment,
            action_space=action_space,
            source_format="lerobot",
        )
        yield Episode(metadata=metadata, frames=frames)


def _row_to_frame(
    row: dict[str, Any],
    *,
    is_first: bool,
    is_last: bool,
    extra_columns: list[str],
    camera_features: dict[str, dict[str, Any]],
    language_instruction: str | None,
) -> Frame:
    extra = {name: np.asarray(row[name], dtype=np.float64) for name in extra_columns}
    proprioception = Proprioception(
        joint_positions=None,
        joint_velocities=None,
        ee_poses={},
        grippers={},
        extra=extra,
    )
    images: dict[str, CameraFrame] = {}
    for name, feature in camera_features.items():
        shape = feature.get("shape", [0, 0])
        height, width = int(shape[0]), int(shape[1])
        images[name] = CameraFrame(
            read=_make_undecoded_reader(name),
            timestamp=float(row["timestamp"]),
            resolution=(height, width),
            timestamp_is_measured=False,
        )
    success = bool(row["next.success"]) if is_last and "next.success" in row else None
    return Frame(
        timestamp=float(row["timestamp"]),
        proprioception=proprioception,
        action=np.asarray(row["action"], dtype=np.float64),
        images=images,
        is_first=is_first,
        is_last=is_last,
        success=success,
        language_instruction=language_instruction,
    )
