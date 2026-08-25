"""The raw-frame datahouse contracts: geometry and causal action alignment."""
import numpy as np
import pytest

from contra_policy.frame_house import FrameEpisodeDataset, FrameHouse

ROOT = "~/code/contra_nes_data/game_trace/datahouse"


@pytest.fixture(scope="module")
def house():
    try:
        return FrameHouse(ROOT, weapon="laser")
    except Exception as exc:                       # pragma: no cover - box without data
        pytest.skip(f"no frame release available: {exc}")


def test_release_is_the_published_laser_set(house):
    assert (len(house), house.declared[1]) == (10293, 1075404)
    assert house.frame_size == (224, 240)
    assert len(house.shards) == 42


def test_native_geometry_is_patch16_divisible(house):
    h, w = house.frame_size
    assert h % 16 == 0 and w % 16 == 0, "a resize would be needed and is refused"


def test_frames_and_actions_are_equal_length(house):
    uid = house.episodes[0]["uid"]
    assert len(house.raw_actions(uid)) == house.length(uid)


def test_seeking_a_window_matches_a_full_decode(house):
    uid = house.episodes[0]["uid"]
    whole = house.frames(uid, 0, house.length(uid))
    assert np.array_equal(house.frames(uid, 3, 4), whole[3:7])


def test_action_shift_matches_the_token_path(house):
    """``frames[i]`` is the screen *after* action ``i``, so the target is ``actions[i+1]``.

    Pinned here because the token dataset applies the same +1; if either moves alone the
    two paths silently train on different pairs.
    """
    uid = house.episodes[0]["uid"]
    item = FrameEpisodeDataset(house, [uid])[0]
    raw = house.raw_actions(uid)
    n = house.length(uid) - 1
    assert item["action"].shape[0] == n
    assert np.array_equal(item["action"].numpy(), raw[1:1 + n])
    assert item["mask"].sum().item() == n


def test_no_goal_row(house):
    """Frame arrays are ``length``, not ``length + 1`` — index i is decision i."""
    uid = house.episodes[0]["uid"]
    assert house.frames(uid).shape[0] == house.length(uid)


