"""Tests for the LeRobotDataset metadata/episode-index reader.

The on-disk fixture built here mirrors the real v3.0 schema confirmed
against ``lerobot/pusht`` on the Hugging Face Hub (column names, nesting,
and the fact that ``meta/info.json`` never declares action-space semantics)
— see ``src/mekiki/readers/lerobot.py`` and ``STATE.md`` for how that was
checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mekiki.episode import ActionDimSpec
from mekiki.readers.lerobot import (
    read_episode_refs,
    read_info,
    validate_action_space,
)

_ACTION_SPACE = (
    ActionDimSpec("x", "absolute", "m", "base_link"),
    ActionDimSpec("y", "absolute", "m", "base_link"),
)


def _write_dataset(root: Path, *, action_dims: int = 2) -> None:
    """Build a minimal on-disk LeRobotDataset directory for testing.

    Two episodes, no video files and no frame-data parquet — this reader
    only touches ``meta/info.json`` and ``meta/episodes/**.parquet`` so far.
    """
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [action_dims]},
            "action": {"dtype": "float32", "shape": [action_dims]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "dataset_from_index": [0, 3],
            "dataset_to_index": [3, 5],
            "length": [3, 2],
            "tasks": [["pick up the cube"], ["pick up the cube"]],
        }
    )
    pq.write_table(table, episodes_dir / "file-000.parquet")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    _write_dataset(tmp_path)
    return tmp_path


def test_read_info_parses_core_fields(dataset_dir: Path) -> None:
    info = read_info(dataset_dir)
    assert info.codebase_version == "v3.0"
    assert info.fps == 10.0
    assert info.total_episodes == 2
    assert info.total_frames == 5
    assert info.action_dims == 2


def test_read_info_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError, match=r"info\.json"):
        read_info(Path("does/not/exist"))


def test_read_info_missing_action_feature_raises(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 10,
                "total_episodes": 0,
                "total_frames": 0,
                "data_path": "",
                "video_path": "",
                "features": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action"):
        read_info(tmp_path)


def test_validate_action_space_raises_when_not_supplied(dataset_dir: Path) -> None:
    info = read_info(dataset_dir)
    with pytest.raises(ValueError, match="no action_space supplied"):
        validate_action_space(info, None)


def test_validate_action_space_raises_on_dimension_mismatch(dataset_dir: Path) -> None:
    info = read_info(dataset_dir)
    wrong_length = (ActionDimSpec("x", "absolute", "m", "base_link"),)
    with pytest.raises(ValueError, match="dimension"):
        validate_action_space(info, wrong_length)


def test_validate_action_space_accepts_matching_spec(dataset_dir: Path) -> None:
    info = read_info(dataset_dir)
    validate_action_space(info, _ACTION_SPACE)  # must not raise


def test_read_episode_refs_returns_all_episodes_in_order(dataset_dir: Path) -> None:
    refs = read_episode_refs(dataset_dir)
    assert [ref.episode_index for ref in refs] == [0, 1]
    assert refs[0].dataset_from_index == 0
    assert refs[0].dataset_to_index == 3
    assert refs[0].length == 3
    assert refs[0].tasks == ("pick up the cube",)
    assert refs[1].dataset_from_index == 3
    assert refs[1].dataset_to_index == 5


def test_read_episode_refs_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="episode index"):
        read_episode_refs(tmp_path)
