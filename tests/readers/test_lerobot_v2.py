"""Tests for LeRobotDataset v2.x support: one parquet file per episode,
episode index at ``meta/episodes.jsonl`` rather than ``meta/episodes/**.parquet``.

The fixture mirrors the real schema confirmed against
``IPEC-COMMUNITY/bridge_orig_lerobot`` (v2.0) and
``IPEC-COMMUNITY/berkeley_cable_routing_lerobot`` (v2.1) on the Hugging Face
Hub — see ``src/mekiki/readers/lerobot.py``'s module docstring and
``docs/rlds.md`` for how those were checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mekiki.episode import ActionDimSpec
from mekiki.readers.lerobot import read_episode_refs, read_episodes, read_info

_ACTION_SPACE = (
    ActionDimSpec("x", "delta", "m", "ee"),
    ActionDimSpec("y", "delta", "m", "ee"),
)


def _write_dataset_v2(root: Path, *, codebase_version: str = "v2.0") -> None:
    """Build a minimal on-disk v2.x LeRobotDataset directory for testing.

    Two episodes (3 frames + 2 frames), each its own parquet file, with a
    global continuous ``index`` column spanning both files — confirmed
    against real data that this is how v2.x actually numbers frames even
    though each episode lives in a separate file.
    """
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "codebase_version": codebase_version,
        "robot_type": "widowx",
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "chunks_size": 1000,
        "fps": 5,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.images.image_0": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "rgb"],
            },
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    episodes_lines = [
        json.dumps({"episode_index": 0, "tasks": ["put spoon in basket"], "length": 3}),
        json.dumps({"episode_index": 1, "tasks": ["open the drawer"], "length": 2}),
    ]
    # a blank line in the middle is harmless and worth tolerating -- real
    # JSONL files sometimes pick up a stray trailing newline from tooling
    (meta_dir / "episodes.jsonl").write_text(
        f"{episodes_lines[0]}\n\n{episodes_lines[1]}\n", encoding="utf-8"
    )

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    episode_0 = pa.table(
        {
            "observation.state": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            "action": [[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]],
            "timestamp": [0.0, 0.2, 0.4],
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
        }
    )
    pq.write_table(episode_0, data_dir / "episode_000000.parquet")
    episode_1 = pa.table(
        {
            "observation.state": [[10.0, 10.0], [11.0, 11.0]],
            "action": [[10.1, 10.1], [11.1, 11.1]],
            "timestamp": [0.0, 0.2],
            "frame_index": [0, 1],
            "episode_index": [1, 1],
            "index": [3, 4],
            "task_index": [1, 1],
        }
    )
    pq.write_table(episode_1, data_dir / "episode_000001.parquet")


@pytest.fixture
def dataset_dir_v2(tmp_path: Path) -> Path:
    _write_dataset_v2(tmp_path)
    return tmp_path


def test_read_info_parses_v2_fields(dataset_dir_v2: Path) -> None:
    info = read_info(dataset_dir_v2)
    assert info.codebase_version == "v2.0"
    assert info.robot_type == "widowx"
    assert info.chunks_size == 1000
    assert info.action_dims == 2


def test_read_episode_refs_v2_resolves_per_episode_paths(dataset_dir_v2: Path) -> None:
    refs = read_episode_refs(dataset_dir_v2)
    assert [ref.episode_index for ref in refs] == [0, 1]
    assert refs[0].data_relative_path == "data/chunk-000/episode_000000.parquet"
    assert refs[1].data_relative_path == "data/chunk-000/episode_000001.parquet"
    # no shared-file slicing needed for v2.x -- each episode owns its file
    assert refs[0].dataset_from_index is None
    assert refs[0].dataset_to_index is None
    assert refs[0].length == 3
    assert refs[0].tasks == ("put spoon in basket",)


def test_read_episode_refs_v2_missing_episodes_jsonl_raises(dataset_dir_v2: Path) -> None:
    (dataset_dir_v2 / "meta" / "episodes.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match=r"episodes\.jsonl"):
        read_episode_refs(dataset_dir_v2)


def test_read_episodes_v2_end_to_end_frame_counts_and_content(dataset_dir_v2: Path) -> None:
    episodes = list(read_episodes(dataset_dir_v2, _ACTION_SPACE))
    assert len(episodes) == 2
    assert [len(list(ep)) for ep in episodes] == [3, 2]

    first_frames = list(episodes[0])
    assert first_frames[0].is_first and first_frames[-1].is_last
    assert [f.timestamp for f in first_frames] == pytest.approx([0.0, 0.2, 0.4])
    assert first_frames[0].proprioception.extra["observation.state"].tolist() == [0.0, 0.0]
    assert all(f.language_instruction == "put spoon in basket" for f in first_frames)

    second_frames = list(episodes[1])
    assert all(f.language_instruction == "open the drawer" for f in second_frames)


def test_read_episodes_v2_robot_embodiment_defaults_to_info_robot_type(
    dataset_dir_v2: Path,
) -> None:
    episode = next(iter(read_episodes(dataset_dir_v2, _ACTION_SPACE)))
    assert episode.metadata.robot_embodiment == "widowx"


def test_read_episodes_v2_robot_embodiment_override_wins(dataset_dir_v2: Path) -> None:
    episode = next(
        iter(read_episodes(dataset_dir_v2, _ACTION_SPACE, robot_embodiment="franka_panda"))
    )
    assert episode.metadata.robot_embodiment == "franka_panda"


def test_read_episodes_v2_camera_present_but_undecoded(dataset_dir_v2: Path) -> None:
    frame = next(iter(next(iter(read_episodes(dataset_dir_v2, _ACTION_SPACE)))))
    camera = frame.images["observation.images.image_0"]
    assert camera.resolution == (256, 256)
    with pytest.raises(NotImplementedError):
        camera.read()


def test_read_episodes_v21_layout_also_supported(tmp_path: Path) -> None:
    # v2.1 is layout-compatible with v2.0 for what this reader touches --
    # confirmed against a real v2.1 dataset (berkeley_cable_routing_lerobot),
    # not assumed.
    _write_dataset_v2(tmp_path, codebase_version="v2.1")
    episodes = list(read_episodes(tmp_path, _ACTION_SPACE))
    assert len(episodes) == 2
