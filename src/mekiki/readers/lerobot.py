"""Reader for the LeRobotDataset on-disk format.

Handles the two LeRobotDataset layouts actually seen in the wild, both
checked against real datasets rather than assumed from documentation
(LeRobotDataset has changed on-disk layout across versions before, and
does not always agree with itself about it):

- **v3.0** (checked against ``lerobot/pusht``): a shared episode index at
  ``meta/episodes/chunk-*/file-*.parquet``, and frame data at
  ``data/chunk-*/file-*.parquet`` where one data file commonly holds many
  episodes' rows concatenated, distinguished by a global ``index`` column.
- **v2.0/v2.1** (checked against ``IPEC-COMMUNITY/bridge_orig_lerobot`` and
  ``IPEC-COMMUNITY/berkeley_cable_routing_lerobot``): a flat
  ``meta/episodes.jsonl`` (one JSON object per line: episode_index, tasks,
  length) and one parquet file *per episode* at
  ``data/chunk-*/episode_{episode_index:06d}.parquet`` — no from/to index
  slicing needed, the whole file is the episode. This is the layout most
  community conversions of Open X-Embodiment datasets use on the Hugging
  Face Hub, including this project's own README example (Bridge V2).

Any other ``codebase_version`` is rejected loudly by `read_info` rather than
read speculatively — see `SUPPORTED_CODEBASE_VERSIONS`.

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

#: LeRobotDataset format versions this reader has been checked against real
#: data for. A dataset reporting a different ``codebase_version`` is
#: rejected by `read_info` rather than read speculatively.
SUPPORTED_CODEBASE_VERSIONS = ("v2.0", "v2.1", "v3.0")


@dataclass(frozen=True, slots=True)
class LeRobotInfo:
    """Parsed ``meta/info.json``.

    Attributes:
        codebase_version: LeRobotDataset format version, e.g. ``"v3.0"``.
        fps: Nominal control frequency in Hz.
        total_episodes: Number of episodes in the dataset.
        total_frames: Number of frames across all episodes.
        chunks_size: Episodes (v2.x) or files (v3.0) per chunk directory —
            needed to resolve v2.x's ``data_path`` template, which encodes
            a chunk number derived from ``episode_index // chunks_size``
            rather than storing it explicitly anywhere.
        data_path: Template for locating frame-data parquet files, e.g.
            ``"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"``
            (v3.0) or ``"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"``
            (v2.x) — the placeholder names differ by version, which is why
            path resolution is version-aware (see `read_episode_refs`)
            rather than a single generic `.format()` call.
        video_path: Template for locating per-camera video files.
        robot_type: Free-text robot/embodiment identifier from info.json
            (e.g. ``"widowx"``), or ``"unknown"`` if absent. Unlike action
            semantics this carries no correctness risk if wrong — it's a
            label, not something a check computes against — so it's used
            as-is rather than requiring a caller override.
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
    chunks_size: int
    data_path: str
    video_path: str
    robot_type: str
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
            nothing to validate an action space against in that case. Also
            raised if ``codebase_version`` isn't one this reader has been
            checked against real data for — see
            `SUPPORTED_CODEBASE_VERSIONS`. Reading an unrecognized version
            further would risk silently misreading rather than failing
            clearly, so this stops immediately instead.

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
    codebase_version = raw["codebase_version"]
    if codebase_version not in SUPPORTED_CODEBASE_VERSIONS:
        raise ValueError(
            f"{info_path} reports codebase_version {codebase_version!r}, but this reader "
            f"has only been checked against real data for {SUPPORTED_CODEBASE_VERSIONS!r} "
            "— see docs/rlds.md for how those were verified. Reading an unrecognized "
            "version further here would risk silently misreading rather than failing "
            "clearly, so this stops now instead."
        )
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
        codebase_version=codebase_version,
        fps=float(raw["fps"]),
        total_episodes=int(raw["total_episodes"]),
        total_frames=int(raw["total_frames"]),
        chunks_size=int(raw["chunks_size"]),
        data_path=raw["data_path"],
        video_path=raw["video_path"],
        robot_type=str(raw.get("robot_type") or "unknown"),
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
        ...     total_frames=1, chunks_size=1000, data_path="", video_path="",
        ...     robot_type="unknown", features={}, action_dims=2,
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
    """One entry from the episode index: where to find one episode's data.

    Comes from ``meta/episodes/**.parquet`` (v3.0) or ``meta/episodes.jsonl``
    (v2.x) — see `read_episode_refs`.

    Attributes:
        episode_index: Index of this episode within the dataset.
        data_relative_path: Path, relative to the dataset root, to the
            parquet file holding this episode's frame data — already
            resolved from `LeRobotInfo.data_path`'s version-specific
            template, so callers never need to know which placeholder
            names that template uses.
        dataset_from_index: First row (inclusive) of this episode's frames
            in that file's global ``index`` column, when the file holds
            more than one episode's rows (v3.0). ``None`` when each episode
            has its own file (v2.x) — the whole file belongs to this
            episode, no slicing needed.
        dataset_to_index: Last row (exclusive), or ``None`` — see above.
        length: Number of frames in this episode.
        tasks: Language instruction(s) recorded for this episode.
    """

    episode_index: int
    data_relative_path: str
    dataset_from_index: int | None
    dataset_to_index: int | None
    length: int
    tasks: tuple[str, ...]


def read_episode_refs(
    dataset_dir: Path, info: LeRobotInfo | None = None
) -> list[LeRobotEpisodeRef]:
    """Read the episode index, however this dataset's version stores it.

    Args:
        dataset_dir: Root of a local LeRobotDataset directory.
        info: Already-parsed `LeRobotInfo`, to avoid re-reading
            ``meta/info.json`` when the caller has it already (e.g.
            `read_episodes`). Read fresh via `read_info` if omitted.

    Returns:
        One `LeRobotEpisodeRef` per episode, ordered by ``episode_index``.

    Raises:
        FileNotFoundError: the episode index (``meta/episodes/**.parquet``
            for v3.0, ``meta/episodes.jsonl`` for v2.x) doesn't exist under
            ``dataset_dir``.
    """
    if info is None:
        info = read_info(dataset_dir)
    if info.codebase_version == "v3.0":
        return _read_episode_refs_v3(dataset_dir, info)
    return _read_episode_refs_v2(dataset_dir, info)


def _read_episode_refs_v3(dataset_dir: Path, info: LeRobotInfo) -> list[LeRobotEpisodeRef]:
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
            data_relative_path = info.data_path.format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
            )
            refs.append(
                LeRobotEpisodeRef(
                    episode_index=row["episode_index"],
                    data_relative_path=data_relative_path,
                    dataset_from_index=row["dataset_from_index"],
                    dataset_to_index=row["dataset_to_index"],
                    length=row["length"],
                    tasks=tuple(row["tasks"]),
                )
            )
    refs.sort(key=lambda ref: ref.episode_index)
    return refs


def _read_episode_refs_v2(dataset_dir: Path, info: LeRobotInfo) -> list[LeRobotEpisodeRef]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"no meta/episodes.jsonl under {dataset_dir} for a {info.codebase_version} "
            "LeRobotDataset"
        )
    refs: list[LeRobotEpisodeRef] = []
    with episodes_path.open(encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            episode_chunk = episode_index // info.chunks_size
            data_relative_path = info.data_path.format(
                episode_chunk=episode_chunk, episode_index=episode_index
            )
            refs.append(
                LeRobotEpisodeRef(
                    episode_index=episode_index,
                    data_relative_path=data_relative_path,
                    dataset_from_index=None,
                    dataset_to_index=None,
                    length=int(row["length"]),
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
    robot_embodiment: str | None = None,
) -> Iterator[Episode]:
    """Read every episode in a LeRobotDataset directory.

    Streams one episode at a time (per docs/episode.md); within an episode,
    proprioception/action/timestamp are read eagerly (cheap — a few floats
    per frame) but camera pixels are never decoded here. Works against
    either on-disk layout this reader supports (v3.0's shared multi-episode
    data files, or v2.x's one-file-per-episode layout) transparently — see
    the module docstring.

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
            Defaults to ``info.robot_type`` (the dataset's own declared
            value, or ``"unknown"`` if it doesn't have one) — pass this to
            override.

    Yields:
        One `Episode` per episode in the dataset, in ``episode_index`` order.

    Raises:
        ValueError: `action_space` doesn't match the dataset, or
            ``codebase_version`` isn't supported (see `validate_action_space`,
            `read_info`).
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
    resolved_robot_embodiment = (
        robot_embodiment if robot_embodiment is not None else info.robot_type
    )
    camera_features = _video_camera_features(info)
    refs = read_episode_refs(dataset_dir, info)

    data_tables: dict[str, pa.Table] = {}
    for ref in refs:
        if ref.data_relative_path not in data_tables:
            data_tables[ref.data_relative_path] = pq.read_table(
                dataset_dir / ref.data_relative_path
            )
        table = data_tables[ref.data_relative_path]

        if ref.dataset_from_index is not None and ref.dataset_to_index is not None:
            mask = pc.and_(
                pc.greater_equal(table["index"], ref.dataset_from_index),
                pc.less(table["index"], ref.dataset_to_index),
            )
            rows = table.filter(mask).to_pylist()
        else:
            rows = table.to_pylist()
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
            robot_embodiment=resolved_robot_embodiment,
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
