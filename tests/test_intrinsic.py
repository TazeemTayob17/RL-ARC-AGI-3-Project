"""Phase 4 I1 (count-based novelty bonus): unit and mechanism-activation checks (plan 6 bug procedure steps 2-3)."""
import math

import numpy as np
import pytest

from src import ppo as ppo_mod
from src.adapter import NO_ACTION, Observation, Transition, build_mask
from src.config import RunConfig
from src.intrinsic import NoveltyCounter
from src.run import run

DQN_FAST = {"warmup": 32, "train_every": 4, "target_update": 100, "eps_decay_actions": 100, "eps_end": 0.05}
PPO_FAST = {"rollout": 32, "minibatches": 2, "epochs": 2}


def _frame(v):
    f = np.zeros((64, 64), np.int8)
    f[0, 0] = v
    return f


def _tr(a, b, progress=0, next_progress=0):
    mask = build_mask([1, 2, 3, 4])
    o = Observation(_frame(a), _frame(a), mask, NO_ACTION, progress, progress, 5)
    n = Observation(_frame(b), _frame(a), mask, 1, next_progress, next_progress, 5)
    return Transition(o, 1, 0.0, n, False, False, {})


def test_bonus_is_beta_over_sqrt_count_and_zero_on_loops():
    c = NoveltyCounter(0.1)
    assert c.bonus(_tr(0, 1)) == pytest.approx(0.1)             # new state, n = 1
    assert c.bonus(_tr(1, 0)) == 0.0                            # back to where the last step started: loop
    assert c.bonus(_tr(0, 1)) == 0.0                            # undo/redo again: still a loop
    assert c.bonus(_tr(1, 2)) == pytest.approx(0.1)             # genuinely new
    assert c.bonus(_tr(2, 3)) == pytest.approx(0.1)
    assert c.bonus(_tr(3, 2)) == 0.0                            # loop
    assert c.bonus(_tr(2, 4)) == pytest.approx(0.1)
    assert c.bonus(_tr(4, 5)) == pytest.approx(0.1)
    assert c.bonus(_tr(5, 3)) == pytest.approx(0.1 / math.sqrt(2))   # revisit (not a 2-step loop): n(3) = 2
    assert c.summary()["loop_zeroed"] == 3


def test_scoped_by_level_and_cleared_on_reset():
    c = NoveltyCounter(0.1)
    c.bonus(_tr(0, 1))
    assert c.bonus(_tr(5, 1, progress=0, next_progress=1)) == pytest.approx(0.1)   # same grid, new level: new state
    c.reset()
    assert c.counts == {} and c.summary()["calls"] == 0
    assert c.bonus(_tr(0, 1)) == pytest.approx(0.1)


def test_beta_zero_touches_no_state():
    c = NoveltyCounter(0.0)
    assert c.bonus(_tr(0, 1)) == 0.0 and c.counts == {} and c.summary()["calls"] == 0
    with pytest.raises(ValueError):
        NoveltyCounter(-1.0)


def _actions(cfg, tmp_path, tag):
    rec = run(cfg, out_dir=str(tmp_path / tag))
    return np.load(tmp_path / tag / rec["identity"]["run_id"] / "actions.npy"), rec


@pytest.mark.parametrize("algo,fast", [("dqn", DQN_FAST), ("ppo", PPO_FAST)])
def test_beta_zero_reproduces_baseline_exactly(tmp_path, algo, fast):
    base = RunConfig(algo=algo, game="synthetic:nav", budget=150, params=dict(fast))
    zero = RunConfig(algo=algo, game="synthetic:nav", budget=150, params={**fast, "intrinsic_beta": 0.0})
    a, _ = _actions(base, tmp_path, "base")
    b, _ = _actions(zero, tmp_path, "zero")
    assert np.array_equal(a, b)


def test_dqn_bonus_reaches_targets_and_changes_behaviour_after_updates(tmp_path):
    base = RunConfig(algo="dqn", game="synthetic:nav", budget=300, params=dict(DQN_FAST))
    treat = RunConfig(algo="dqn", game="synthetic:nav", budget=300, params={**DQN_FAST, "intrinsic_beta": 0.5})
    a, ra = _actions(base, tmp_path, "base")
    b, rb = _actions(treat, tmp_path, "treat")
    ib = rb["learner"]["intrinsic"]
    assert ib["nonzero"] > 0 and ib["total_bonus"] > 0 and ra["learner"]["intrinsic"]["nonzero"] == 0
    first = int(np.argmax(a != b))
    assert a[:DQN_FAST["warmup"]].tolist() == b[:DQN_FAST["warmup"]].tolist()   # no learning signal yet: identical
    assert (a != b).any() and first >= DQN_FAST["warmup"]                        # diverges only after updates
    assert rb["learner"]["gradient_steps"] > 0 and rb["learner"]["q_mean_ema"] != ra["learner"]["q_mean_ema"]


def test_ppo_bonus_reaches_advantages_and_changes_behaviour_after_updates(tmp_path, monkeypatch):
    seen = []
    orig = ppo_mod.compute_gae
    monkeypatch.setattr(ppo_mod, "compute_gae",
                        lambda r, v, d, lv, g, lam: (seen.append(np.asarray(r).copy()), orig(r, v, d, lv, g, lam))[1])
    base = RunConfig(algo="ppo", game="synthetic:nav", budget=200, params=dict(PPO_FAST))
    treat = RunConfig(algo="ppo", game="synthetic:nav", budget=200, params={**PPO_FAST, "intrinsic_beta": 0.5})
    a, ra = _actions(base, tmp_path, "base")
    n_base = len(seen)
    b, rb = _actions(treat, tmp_path, "treat")
    assert any(r.max() > 0 for r in seen[n_base:])                    # bonus is inside the rewards used for GAE
    assert rb["learner"]["intrinsic"]["nonzero"] > 0
    assert a[:PPO_FAST["rollout"]].tolist() == b[:PPO_FAST["rollout"]].tolist()   # first rollout collected pre-update
    assert (a != b).any()
