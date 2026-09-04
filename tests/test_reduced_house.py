"""Policy-facing contracts for the data-owned reduced Laser representation."""

import numpy as np
import pytest

from contra_policy.reduced_house import (ReducedFeatureEpisodeDataset,
                                         ReducedFeatureHouse)

ROOT = "~/code/contra_nes_data/game_trace/datahouse"
SHA = "f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c"


@pytest.fixture(scope="module")
def house():
    try:
        return ReducedFeatureHouse(ROOT, encoder_sha256=SHA)
    except Exception as exc:  # pragma: no cover - environment without the release
        pytest.skip(f"reduced feature release unavailable: {exc}")


def test_release_contract(house):
    assert len(house.shards) == 42
    assert house.declared == (10293, 1075404)
    assert house.encoder_sha256 == SHA


def test_features_and_actions_are_aligned_and_shifted(house):
    uid = house.episodes[0]["uid"]
    features, actions = house.features(uid), house.raw_actions(uid)
    assert features.shape == (house.length(uid), 256, 4, 4)
    assert features.dtype == np.float16
    assert actions.shape == (house.length(uid),)
    item = ReducedFeatureEpisodeDataset(house, [uid])[0]
    assert np.array_equal(item["action"].numpy(), actions[1:])
    assert item["features"].shape[0] == house.length(uid) - 1
    assert item["mask"].sum().item() == house.length(uid) - 1
