"""Phase 5 Improvement 2 (candidate A, event-terminated repeat option): unit and mechanism checks (plan 7.3 card)."""
import numpy as np
import pytest

from src import ppo as ppo_mod
from src.adapter import NO_ACTION, Observation, Transition, build_mask
from src.config import RunConfig
from src.dqn_v1.replay import NStepAccumulator, StepTransition
from src.learner import RandomPolicy
from src.options import EventRepeat
from src.ppo import compute_gae
from src.run import run

MASK = build_mask([1, 2, 3, 4])
DQN_FAST = {"warmup": 32, "train_every": 4, "target_update": 100, "eps_decay_actions": 100, "eps_end": 0.05}
PPO_FAST = {"rollout": 32, "minibatches": 2, "epochs": 2}


def _frame(v):
    f = np.zeros((64, 64), np.int8)
    f[0, 0] = v
    return f


def _tr(a, b, action=1, reward=0.0, terminal=False, truncated=False, mask=MASK):
    o = Observation(_frame(a), _frame(a), mask, NO_ACTION, 0, 0, 5)
    n = Observation(_frame(b), _frame(a), mask, action, 0, 0, 5)
    return Transition(o, action, reward, n, terminal, truncated, {"state": "NOT_FINISHED"})


class Recorder:
    """Stub inner learner: records what it is shown."""
    def __init__(self):
        self.acts, self.seen, self.finished = 0, [], False

    def start_run(self, seed): ...
    def act(self, obs):
        self.acts += 1
        return 1

    def observe(self, t): self.seen.append(t)
    def finish(self, final_obs): self.finished = True
    def diagnostics(self): return {}
    def state_dict(self): return {}
    def load_state_dict(self, d): ...


def _drive(wrapper, transitions):
    """Feed transitions as the run loop would; returns the actions the wrapper asked for before each observe()."""
    asked = []
    for t in transitions:
        asked.append(wrapper.act(t.obs))
        wrapper.observe(t)
    return asked


def _wrap(**kw):
    rec = Recorder()
    w = EventRepeat(rec, **{"cap": 4, "prob": 1.0, **kw})
    w.start_run(0)
    return w, rec


# ---------------------------------------------------------------- stop rules
def test_repeats_until_no_effect_and_hands_over_one_option():
    w, rec = _wrap()
    asked = _drive(w, [_tr(0, 1), _tr(1, 2), _tr(2, 2)])             # third repetition changes nothing
    assert asked == [1, 1, 1] and rec.acts == 1                      # the inner learner decided once
    assert len(rec.seen) == 1
    t = rec.seen[0]
    assert t.info["k"] == 3 and len(t.info["steps"]) == 3
    assert t.obs.frame[0, 0] == 0 and t.next_obs.frame[0, 0] == 2 and t.action == 1
    assert w.stops["no_effect"] == 1 and w.diagnostics()["option"]["repeated_steps"] == 2


def test_cap_includes_the_first_step():
    w, rec = _wrap(cap=3)
    _drive(w, [_tr(0, 1), _tr(1, 2), _tr(2, 3)])
    assert rec.seen[0].info["k"] == 3 and w.stops["cap"] == 1


def test_cycle_level_up_terminal_illegal_and_budget_stop_the_option():
    cases = {
        "cycle": [_tr(0, 1), _tr(1, 0)],
        "level_up": [_tr(0, 1), _tr(1, 2, reward=1.0)],
        "terminal": [_tr(0, 1), _tr(1, 2, terminal=True)],
        "budget": [_tr(0, 1), _tr(1, 2, truncated=True)],
        "illegal": [_tr(0, 1, mask=build_mask([1, 3]))],   # index 1 is ACTION2, absent from this mask
    }
    for reason, ts in cases.items():
        w, rec = _wrap()
        _drive(w, ts)
        assert w.stops[reason] == 1, reason
        assert len(rec.seen) == 1 and rec.seen[0].info.get("k", 1) == len(ts), reason


def test_first_step_without_effect_is_never_repeated_and_forwarded_unchanged():
    w, rec = _wrap()
    t = _tr(0, 0)
    _drive(w, [t])
    assert rec.seen == [t] and rec.seen[0] is t and w.stops["no_effect"] == 1


def test_prob_zero_and_cap_one_forward_every_transition_unchanged():
    for kw in ({"prob": 0.0}, {"cap": 1}):
        w, rec = _wrap(**kw)
        ts = [_tr(0, 1), _tr(1, 2), _tr(2, 3)]
        asked = _drive(w, ts)
        assert asked == [1, 1, 1] and rec.acts == 3
        assert all(a is b for a, b in zip(rec.seen, ts)) and w.options == 0


def test_engagement_is_drawn_once_per_decision_with_its_own_rng():
    w, rec = _wrap(prob=0.5)
    for i in range(200):
        w.act(None)
        w.observe(_tr(0, 1))
        while w._repeating:
            w.act(None)
            w.observe(_tr(0, 0))
    frac = w.options / 200
    assert 0.35 < frac < 0.65


def test_macro_reward_is_discounted_sum_and_finish_flushes_a_partial_option():
    w, rec = _wrap(gamma=0.9)
    _drive(w, [_tr(0, 1), _tr(1, 2, reward=0.0), _tr(2, 3, reward=0.0)])     # option still open (cap 4)
    assert not rec.seen
    w.finish(None)
    assert rec.finished and len(rec.seen) == 1 and rec.seen[0].info["k"] == 3
    w2, rec2 = _wrap(gamma=0.9)
    _drive(w2, [_tr(0, 1), _tr(1, 2, reward=1.0)])
    assert rec2.seen[0].reward == pytest.approx(0.9)                         # level-up on the 2nd step: gamma**1 * 1


# ---------------------------------------------------------------- learners discount an option by gamma**k
def test_nstep_accumulator_uses_gamma_to_the_number_of_primitive_steps():
    acc = NStepAccumulator(1, 0.9)
    z = np.zeros(2, np.float32)
    mk = lambda steps: StepTransition(z, (1,), z, (1, 0, 0), 0.0, z, (1,), False, steps)
    assert acc.append(mk(1))[0].gamma_n == pytest.approx(0.9)
    assert acc.append(mk(3))[0].gamma_n == pytest.approx(0.9 ** 3)


def test_gae_with_per_step_discount_matches_scalar_when_all_ones():
    r = np.array([0, 0, 1, 0], np.float32)
    v = np.array([.1, .2, .3, .4], np.float32)
    d = np.array([0, 0, 0, 0], np.float32)
    a = compute_gae(r, v, d, 0.5, 0.99, 0.95)
    b = compute_gae(r, v, d, 0.5, np.full(4, 0.99, np.float32), 0.95)
    assert np.allclose(a, b, atol=1e-6)
    c = compute_gae(r, v, d, 0.5, np.array([0.99, 0.99 ** 3, 0.99, 0.99], np.float32), 0.95)
    assert not np.allclose(a, c)


def test_dqn_v0_stores_gamma_to_the_k():
    from src.dqn import DQN
    cfg = RunConfig(algo="dqn", game="synthetic:nav", budget=10, params=DQN_FAST).resolved()
    L = DQN(cfg.arch, cfg.params)
    L.start_run(0)
    t = _tr(0, 1)
    L.act(t.obs)
    L.observe(t)
    macro = Transition(t.obs, 1, 0.0, t.next_obs, False, False, {"k": 3, "steps": [t, t, t]})
    L.act(t.obs)
    L.observe(macro)
    assert L.disc[0] == pytest.approx(0.99) and L.disc[1] == pytest.approx(0.99 ** 3)


# ---------------------------------------------------------------- end-to-end on the real run loop
def _cfg(algo, game, budget, seed=0, params=None, version="V0"):
    return RunConfig(algo=algo, game=game, budget=budget, seed=seed, version=version, params=params or {})


def test_wrapped_learner_with_options_off_reproduces_the_baseline_actions(tmp_path):
    for algo, extra in (("dqn", DQN_FAST), ("ppo", PPO_FAST)):
        base = run(_cfg(algo, "synthetic:nav", 300, params=extra), out_dir=str(tmp_path / "b"))
        for off in ({"option_cap": 8, "option_prob": 0.0}, {"option_cap": 1}):
            r = run(_cfg(algo, "synthetic:nav", 300, params={**extra, **off}), out_dir=str(tmp_path / "w"))
            a = np.load(tmp_path / "b" / base["identity"]["run_id"] / "actions.npy")
            b = np.load(tmp_path / "w" / r["identity"]["run_id"] / "actions.npy")
            assert np.array_equal(a, b), (algo, off)


def test_option_reaches_a_goal_that_random_play_cannot_and_every_step_is_charged():
    budget = 1500
    plain = [run(_cfg("random_type", "synthetic:track", budget, seed=s)) for s in range(3)]
    opt = []
    for s in range(3):
        cfg = _cfg("random_type", "synthetic:track", budget, seed=s, params={})
        cfg = cfg.resolved()
        learner = EventRepeat(RandomPolicy("type"), cap=8, prob=0.5, gamma=0.99)
        opt.append(run(cfg, learner=learner))
    assert sum(r["counters"]["high_water"] for r in opt) > 3 * max(1, sum(r["counters"]["high_water"] for r in plain))
    for r in opt:
        o = r["learner"]["option"]
        assert r["counters"]["scorer_actions"] == budget == o["primitive_steps"]       # every step charged
        assert o["options_longer_than_one"] > 0 and max(map(int, o["length_histogram"])) <= o["cap"]
        assert o["stop_reasons"]["level_up"] > 0


@pytest.mark.parametrize("algo,extra", [("dqn", DQN_FAST), ("ppo", PPO_FAST)])
def test_options_run_through_dqn_and_ppo_and_change_behaviour(algo, extra, tmp_path):
    kw = {**extra, "option_cap": 8, "option_prob": 0.5}
    base = run(_cfg(algo, "synthetic:track", 600, params=extra), out_dir=str(tmp_path / "b"))
    r = run(_cfg(algo, "synthetic:track", 600, params=kw), out_dir=str(tmp_path / "o"))
    assert r["status"] == "budget" and r["error"] is None
    o = r["learner"]["option"]
    assert o["options_longer_than_one"] > 0 and o["primitive_steps"] == 600 and o["decisions"] < 600
    assert r["learner"]["gradient_steps"] > 0
    a = np.load(tmp_path / "b" / base["identity"]["run_id"] / "actions.npy")
    b = np.load(tmp_path / "o" / r["identity"]["run_id"] / "actions.npy")
    assert not np.array_equal(a, b)
