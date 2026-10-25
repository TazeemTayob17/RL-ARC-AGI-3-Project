"""I4: state graph, frontier option wrapper, semi-MDP hand-over, DQN / PPO integration, off-mode equivalence."""
import numpy as np
import pytest
import torch

from src.adapter import NO_ACTION, OPTION_ACT, Observation, Transition, build_mask
from src.config import RunConfig
from src.frontier import MAX_PATH, FrontierOption, StateGraph, node_key
from src.model import ActorCritic, collate
from src.ppo import evaluate_actions
from src.run import run


def _obs(v, progress=0, avail=(1, 2)):
    f = np.zeros((64, 64), dtype=np.int8)
    f[10, 10] = v
    return Observation(f, f.copy(), build_mask(avail), NO_ACTION, progress, 0, 3)


def _t(o, a, o2, terminal=False, reward=0.0):
    return Transition(o, a, reward, o2, terminal, False, {"state": "GAME_OVER" if terminal else "NOT_FINISHED"})


def test_graph_plan_is_the_current_node_when_it_has_untried_actions():
    g = StateGraph()
    path, nodes, free = g.plan(_obs(1))
    assert path == [] and nodes == [] and free == [0, 1]                    # ACTION1 and ACTION2 are simple indices 0 and 1


def test_graph_walks_to_the_nearest_node_with_an_untried_action():
    g = StateGraph()
    a, b, c = _obs(1), _obs(2), _obs(3)
    g.record(_t(a, 0, b))
    g.record(_t(a, 1, a))                                                    # a is exhausted (both actions tried)
    g.record(_t(b, 0, c))                                                    # b: action 1 still untried
    path, nodes, free = g.plan(a)
    assert path == [0] and nodes == [node_key(b)] and free == [1]
    g.record(_t(b, 1, b))
    g.record(_t(c, 0, c)); g.record(_t(c, 1, c))
    assert g.plan(a) is None                                                 # everything reachable is tried: no frontier


def test_graph_never_walks_through_terminal_nodes_and_respects_the_path_limit():
    g = StateGraph()
    a, dead = _obs(1), _obs(2)
    g.record(_t(a, 0, dead, terminal=True)); g.record(_t(a, 1, a))
    assert g.plan(a) is None
    chain = [_obs(10 + i) for i in range(MAX_PATH + 3)]
    for x, y in zip(chain, chain[1:]):
        g.record(_t(x, 0, y)); g.record(_t(x, 1, x))
    g.record(_t(chain[-1], 0, chain[-1]))                                    # only the far end could still have action 1
    assert g.plan(chain[0]) is None                                          # farther than MAX_PATH: not offered


def test_graph_filters_untried_actions_by_the_current_mask():
    g = StateGraph()
    full = _obs(1, avail=(1, 2))
    g.ensure(full)
    narrowed = Observation(full.frame, full.prev_frame, build_mask((2,)), NO_ACTION, 0, 0, 3)
    assert g.plan(narrowed)[2] == [1]


class _Scripted:
    """Inner learner that takes the option whenever it is offered, else ACTION1; records what it is shown."""
    name = "scripted"

    def start_run(self, seed): self.seen, self.offered = [], 0
    def act(self, obs):
        self.offered += int(obs.option_ok)
        return OPTION_ACT if obs.option_ok else 0
    def observe(self, t): self.seen.append(t)
    def finish(self, final_obs): ...
    def diagnostics(self): return {}
    def state_dict(self): return {}
    def load_state_dict(self, d): ...


def _run_stub(mode, budget=240, task="nav"):
    inner = _Scripted()
    rec = run(RunConfig(algo="random_type", game=f"synthetic:{task}", budget=budget), learner=FrontierOption(inner, mode=mode))
    return rec, inner


def test_inner_sees_the_option_as_one_semi_markov_transition_and_every_step_is_charged():
    rec, inner = _run_stub("learned")
    assert rec["counters"]["scorer_actions"] == 240
    options = [t for t in inner.seen if t.action == OPTION_ACT]
    assert options and all(t.info["k"] == len(t.info["steps"]) for t in options)
    assert sum(t.info["k"] for t in inner.seen) <= 240                       # plain transitions have k absent (= 1)
    assert sum(t.info.get("k", 1) for t in inner.seen) == rec["learner"]["frontier"]["primitive_steps"]
    for t in options:
        steps = t.info["steps"]
        assert t.obs is not steps[0].obs and t.obs.option_ok                 # the starting observation says the option was legal
        assert t.reward == pytest.approx(sum(0.99 ** i * s.reward for i, s in enumerate(steps)))
        assert t.next_obs.frame is steps[-1].next_obs.frame
    f = rec["learner"]["frontier"]
    assert f["options_started"] == len(options) > 0 and f["option_steps"] > 0


def test_forced_mode_explores_more_distinct_states_than_never_offering_the_option():
    forced, inner = _run_stub("forced", budget=300)
    off, _ = _run_stub("off", budget=300)
    assert forced["learner"]["frontier"]["options_started"] > 0
    assert not [t for t in inner.seen if t.action == OPTION_ACT]            # forced options are not shown to the learner
    assert forced["diagnostics"]["unique_states"] >= off["diagnostics"]["unique_states"]


def test_deviation_ends_the_option_and_is_counted():
    inner = _Scripted()
    w = FrontierOption(inner, mode="learned")
    w.start_run(0)
    a, b, c, d = _obs(1), _obs(2), _obs(3), _obs(4)
    w.graph.record(_t(a, 0, b)); w.graph.record(_t(a, 1, a)); w.graph.record(_t(b, 1, b))
    oa = Observation(a.frame, a.prev_frame, a.mask, NO_ACTION, 0, 0, 3)
    assert w.act(oa) == 0                                                    # walks ACTION1 (index 0) toward b
    w.observe(_t(oa, 0, d))                                                  # lands in d, not b
    assert w.stops["deviation"] == 1 and inner.seen[-1].action == OPTION_ACT and inner.seen[-1].info["k"] == 1


def test_mode_off_is_bit_for_bit_the_unwrapped_learner():
    for algo, version in (("ppo", "V0"), ("dqn", "V1")):
        base = dict(algo=algo, version=version, game="synthetic:nav", seed=3, budget=140, threads=1)
        a = run(RunConfig(params={}, **base))
        b = run(RunConfig(params={"frontier_mode": "off"}, **base))
        assert a["diagnostics"]["progress_curve"] == b["diagnostics"]["progress_curve"]
        assert a["diagnostics"]["action_type_counts"] == b["diagnostics"]["action_type_counts"]
        assert a["diagnostics"]["unique_states"] == b["diagnostics"]["unique_states"]


@pytest.mark.parametrize("params", [{"frontier_mode": "learned"},
                                    {"frontier_mode": "learned", "object_clicks": True, "effect_memory": True,
                                     "intrinsic_beta": 0.01}])
def test_dqn_learns_with_the_option_in_its_choice_set(params):
    rec = run(RunConfig(algo="dqn", version="V1", game="synthetic:nav", seed=1, budget=260, threads=1, params=params))
    f = rec["learner"]["frontier"]
    assert rec["status"] in ("budget", "won") and rec["counters"]["scorer_actions"] == 260
    assert f["options_started"] > 0 and rec["learner"]["updates"] > 0
    assert rec["learner"]["option_explore_draws"] > 0


@pytest.mark.parametrize("params", [{"frontier_mode": "learned"},
                                    {"frontier_mode": "learned", "object_clicks": True, "effect_memory": True,
                                     "intrinsic_beta": 0.01}])
def test_ppo_learns_with_the_option_as_an_action_type(params):
    rec = run(RunConfig(algo="ppo", version="V0", game="synthetic:nav", seed=1, budget=300, threads=1, params=params))
    assert rec["status"] in ("budget", "won") and rec["counters"]["scorer_actions"] == 300
    assert rec["learner"]["frontier_option"] and rec["learner"]["option_type_picks"] > 0
    assert rec["learner"]["gradient_steps"] > 0


def test_ppo_option_type_probability_is_masked_when_the_option_is_not_legal():
    torch.manual_seed(0)
    net = ActorCritic({"id": "onehotcnn_stack2", "channels": 4, "context_dim": 8, "extra_types": 1})
    o = _obs(1)
    b = collate([o, o])
    acts = torch.tensor([OPTION_ACT, OPTION_ACT])
    lp, ent, _ = evaluate_actions(net, b, acts, torch.tensor([True, False]))
    assert lp[0].exp().item() > 0.0 and lp[1].item() < -1e6                  # legal vs masked
    assert torch.isfinite(ent).all()
    lp2, _, _ = evaluate_actions(net, b, torch.tensor([0, 0]), torch.tensor([True, True]))
    assert (lp2.exp() < 1).all()                                             # the option takes probability mass when legal
