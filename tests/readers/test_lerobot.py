"""Tests for the LeRobotDataset metadata/episode-index reader.

The on-disk fixture built here mirrors the real v3.0 schema confirmed
against ``lerobot/pusht`` on the Hugging Face Hub (column names, nesting,
and the fact that ``meta/info.json`` never declares action-space semantics)
— see ``src/mekiki/readers/lerobot.py`` and ``STATE.md`` for how that was
checked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mekiki.episode import ActionDimSpec
from mekiki.readers.lerobot import (
    read_episode_refs,
    read_episodes,
    read_info,
    validate_action_space,
)

_ACTION_SPACE = (
    ActionDimSpec("x", "absolute", "m", "base_link"),
    ActionDimSpec("y", "absolute", "m", "base_link"),
)


def _write_dataset(root: Path, *, action_dims: int = 2) -> None:
    """Build a minimal on-disk LeRobotDataset directory for testing.

    Two episodes (3 frames + 2 frames), one non-video state column
    (deliberately unlabeled, like a real dataset's ``observation.state``)
    and one video-backed camera feature, so both the "state goes to
    Proprioception.extra" and "camera exists but isn't decoded" paths get
    exercised.
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
            "observation.image": {
                "dtype": "video",
                "shape": [64, 64, 3],
                "names": ["height", "width", "channel"],
            },
            "action": {"dtype": "float32", "shape": [action_dims]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    episodes_table = pa.table(
        {
            "episode_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "dataset_from_index": [0, 3],
            "dataset_to_index": [3, 5],
            "length": [3, 2],
            "tasks": [["pick up the cube"], ["place the cube"]],
        }
    )
    pq.write_table(episodes_table, episodes_dir / "file-000.parquet")

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    data_table = pa.table(
        {
            "observation.state": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0]],
            "action": [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [10.1, 10.1], [11.1, 11.1]],
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1],
            "next.success": [False, False, True, False, True],
            "index": [0, 1, 2, 3, 4],
            "task_index": [0, 0, 0, 1, 1],
        }
    )
    pq.write_table(data_table, data_dir / "file-000.parquet")


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
    assert info.chunks_size == 1000
    assert info.robot_type == "unknown"


def test_read_info_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError, match=r"info\.json"):
        read_info(Path("does/not/exist"))


def test_read_info_rejects_unsupported_codebase_version(tmp_path: Path) -> None:
    # a hypothetical future version this reader hasn't been checked against
    # -- must fail clearly, not silently misread. (v2.0/v2.1/v3.0 are all
    # real and supported -- see test_lerobot_v2.py and the fixture above.)
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v4.0",
                "fps": 10,
                "total_episodes": 1,
                "total_frames": 1,
                "data_path": "",
                "video_path": "",
                "features": {"action": {"dtype": "float32", "shape": [2]}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"v4\.0"):
        read_info(tmp_path)


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


def test_read_episode_refs_missing_directory_raises(dataset_dir: Path) -> None:
    # dataset_dir has a valid meta/info.json (v3.0) but no meta/episodes/ --
    # the failure should be about the missing episode index specifically,
    # not a generic "no info.json" error from an earlier step.
    shutil.rmtree(dataset_dir / "meta" / "episodes")
    with pytest.raises(FileNotFoundError, match="episode index"):
        read_episode_refs(dataset_dir)


def test_read_episodes_yields_expected_episode_and_frame_counts(dataset_dir: Path) -> None:
    episodes = list(read_episodes(dataset_dir, _ACTION_SPACE))
    assert len(episodes) == 2
    assert [len(list(ep)) for ep in episodes] == [3, 2]


def test_read_episodes_frame_boundaries_and_timestamps(dataset_dir: Path) -> None:
    first_episode = next(iter(read_episodes(dataset_dir, _ACTION_SPACE)))
    frames = list(first_episode)
    assert frames[0].is_first and not frames[0].is_last
    assert frames[-1].is_last and not frames[-1].is_first
    assert [f.timestamp for f in frames] == pytest.approx([0.0, 0.1, 0.2])


def test_read_episodes_unlabeled_state_goes_to_extra_not_guessed(dataset_dir: Path) -> None:
    first_episode = next(iter(read_episodes(dataset_dir, _ACTION_SPACE)))
    frame = next(iter(first_episode))
    assert frame.proprioception.joint_positions is None
    assert frame.proprioception.ee_poses == {}
    assert frame.proprioception.grippers == {}
    assert frame.proprioception.extra["observation.state"].tolist() == [0.0, 0.0]


def test_read_episodes_action_matches_action_space_length(dataset_dir: Path) -> None:
    first_episode = next(iter(read_episodes(dataset_dir, _ACTION_SPACE)))
    frame = next(iter(first_episode))
    assert frame.action.shape == (len(_ACTION_SPACE),)
    assert frame.action.tolist() == pytest.approx([0.1, 0.1])


def test_read_episodes_success_only_set_on_last_frame(dataset_dir: Path) -> None:
    first_episode = next(iter(read_episodes(dataset_dir, _ACTION_SPACE)))
    frames = list(first_episode)
    assert [f.success for f in frames] == [None, None, True]


def test_read_episodes_language_instruction_from_episode_tasks(dataset_dir: Path) -> None:
    episodes = list(read_episodes(dataset_dir, _ACTION_SPACE))
    assert all(f.language_instruction == "pick up the cube" for f in episodes[0])
    assert all(f.language_instruction == "place the cube" for f in episodes[1])


def test_read_episodes_camera_present_but_undecoded(dataset_dir: Path) -> None:
    first_episode = next(iter(read_episodes(dataset_dir, _ACTION_SPACE)))
    frame = next(iter(first_episode))
    camera = frame.images["observation.image"]
    assert camera.resolution == (64, 64)
    with pytest.raises(NotImplementedError, match=r"observation\.image"):
        camera.read()


def test_read_episodes_rejects_missing_action_space(dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="no action_space supplied"):
        list(read_episodes(dataset_dir, None))  # type: ignore[arg-type]
