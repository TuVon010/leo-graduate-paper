import numpy as np
import pytest

from leo_routing.agents.rollout_buffer import compute_gae
from leo_routing.agents.settings import rl_settings


def test_gae_stops_at_episode_boundaries():
    advantages, returns = compute_gae([1, 2, 3], [0.5, 0.6, 0.7], [0.6, 0, 0], [False, True, True], 1, 1)
    assert advantages == pytest.approx([2.5, 1.4, 2.3])
    assert returns == pytest.approx([3, 2, 3])


def test_gae_bootstraps_only_a_continuing_rollout():
    _, continuing = compute_gae([1], [2], [5], [False], 0.9, 0.95)
    _, terminal = compute_gae([1], [2], [5], [True], 0.9, 0.95)
    assert continuing == pytest.approx([5.5])
    assert terminal == pytest.approx([1])
    assert np.isfinite(continuing).all()


@pytest.mark.parametrize("patch", [{"gamma": 1.2}, {"hidden_dim": 15}, {"train_seeds": [100]},
                                  {"use_mask": "false"}, {"learning_rate": 0}, {"encodre": "gat"}])
def test_bad_rl_settings_rejected(patch):
    with pytest.raises(ValueError):
        rl_settings({"rl": patch})
