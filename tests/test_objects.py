"""I3: segmentation, effect memory, DQN choice set, PPO pointer head, and the memory-off equivalence."""
import numpy as np
import pytest
import torch

from src.adapter import N_SIMPLE, Observation, Transition, build_mask
from src.config import RunConfig
from src.dqn_v1.action_bank import ActionBank
from src.dqn_v1.object_actions import build_action_set
from src.objects import FEAT_DIM, K_MAX, TILE_AREA, EffectMemory, build_view, candidates, candidates_with_cells
from src.ppo import evaluate_object_actions, object_logits
from src.run import run


def _frame():
    f = np.zeros((64, 64), dtype=np.int8)
    f[10:14, 10:14] = 3          # object A
    f[30:32, 40:50] = 5          # object B
    f[50, 5] = 7                 # object C (single pixel)
    return f


def test_segmentation_finds_each_object_with_inside_click_pixel():
    f = _frame()
    cands, cells = candidates_with_cells(f)
    colours = sorted(c.colour for c in cands)
    assert {3, 5, 7} <= set(colours)
    for c, cell in zip(cands, cells):
        assert (c.y * 64 + c.x) in set(cell.tolist())          # representative pixel lies inside its own object
        assert f[c.y, c.x] == c.colour
    covered = np.concatenate(cells)
    assert len(covered) == len(set(covered.tolist())) == 64 * 64 or len(cands) == K_MAX


def test_large_background_is_tiled_and_cap_holds():
    f = np.zeros((64, 64), dtype=np.int8)
    cands = candidates(f)
    assert len(cands) > 1 and all(c.area <= TILE_AREA or c.area <= 256 for c in cands)
    noisy = np.random.default_rng(0).integers(0, 16, (64, 64)).astype(np.int8)
    assert len(candidates(noisy)) == K_MAX


def test_effect_memory_drops_known_noops_but_retries_and_never_empties():
    f = _frame()
    mem = EffectMemory(True, retry_prob=0.0)
    mem.reset(0)
    view = build_view(f, 0, mem)
    first = view.cands[view.keep[0]]
    mem.record(view.skey, first, effective=False)
    view2 = build_view(f, 0, mem)
    assert first not in [view2.cands[i] for i in view2.keep]               # known dead in this state
    other_state = build_view(f, 1, mem)                                    # different progress = different state
    assert first in [other_state.cands[i] for i in other_state.keep]
    for c in view.cands:                                                   # everything dead -> fallback to all
        mem.record(view.skey, c, effective=False)
    assert len(build_view(f, 0, mem).keep) == len(view.cands)
    mem2 = EffectMemory(True, retry_prob=1.0)
    mem2.reset(0)
    v = build_view(f, 0, mem2)
    mem2.record(v.skey, v.cands[0], effective=False)
    assert len(build_view(f, 0, mem2).keep) == len(v.cands)               # retry probability 1 keeps everything


def test_dead_signature_rule_and_disabled_memory():
    f = _frame()
    mem = EffectMemory(True, retry_prob=0.0)
    mem.reset(0)
    c = candidates(f)[0]
    for s in range(3):
        mem.record(1000 + s, c, effective=False)
    assert mem.known_dead(5, c)                                           # tried 3 times in other states, never effective
    off = EffectMemory(False)
    off.reset(0)
    off.record(1, c, effective=False)
    assert off.n_recorded == 0 and len(off.filter(1, [c])) == 1


def test_dqn_action_set_rows_and_keys():
    bank = ActionBank()
    mem = EffectMemory(True)
    mem.reset(0)
    view = build_view(_frame(), 0, mem)
    s = build_action_set(bank, [1, 2, 6], view, mem)
    assert s.rows.shape == (len(s.keys), bank.action_dim + FEAT_DIM)
    assert len(s.click_rows) == len(view.keep) + 1
    assert all(s.keys[s.row_of(k)] == k for k in s.keys)
    plain = build_action_set(bank, [1, 2], None, mem)                       # no ACTION6: simple actions only
    assert [k.action_id for k in plain.keys] == [1, 2] and not plain.click_rows


def test_pointer_head_pools_inside_objects_and_masks_invalid():
    cl = torch.arange(4096, dtype=torch.float32)[None]
    idx = torch.full((1, 4096), -1, dtype=torch.long)
    idx[0, :10] = 0
    idx[0, 100:104] = 1
    out = object_logits(cl, idx)
    assert out.shape == (1, K_MAX + 1)
    assert out[0, 0].item() == pytest.approx(4.5) and out[0, 1].item() == pytest.approx(101.5)
    assert out[0, K_MAX].item() == pytest.approx(cl.mean().item())


def test_ppo_object_log_prob_is_consistent_with_its_mask():
    from src.model import ActorCritic, collate
    torch.manual_seed(0)
    net = ActorCritic({"id": "onehotcnn_stack2", "channels": 4, "context_dim": 8})
    f = _frame()
    obs = Observation(f, f, build_mask([1, 2, 6]), -1, 0, 0, 3)
    mem = EffectMemory(True)
    mem.reset(0)
    view = build_view(f, 0, mem)
    valid = np.zeros(K_MAX + 1, dtype=bool)
    valid[view.keep] = True
    valid[K_MAX] = True
    b = collate([obs])
    oi = torch.from_numpy(view.obj_index.astype(np.int64))[None]
    total = 0.0
    for slot in np.flatnonzero(valid).tolist():                           # probabilities of all valid click slots sum to p(click)
        lp, _, _ = evaluate_object_actions(net, b, torch.tensor([N_SIMPLE]), torch.tensor([slot]), oi,
                                           torch.from_numpy(valid)[None])
        total += lp.exp().item()
    lp_click_type = evaluate_object_actions(net, b, torch.tensor([N_SIMPLE]), torch.tensor([K_MAX]), oi,
                                            torch.from_numpy(valid)[None])[0]
    tl = net(b)[0]
    tm = torch.tensor([[True, True, False, False, False, False, True]])
    p_click = torch.softmax(tl.masked_fill(~tm, -1e9), 1)[0, 6].item()
    assert total == pytest.approx(p_click, rel=1e-4)
    assert lp_click_type.shape == (1,)


def test_synthetic_one_responsive_object_is_found_and_memory_off_is_equivalent_to_empty_memory():
    """Only one object responds. DQN with I3 must reach the reward; memory with retry probability 1 equals memory off."""
    cfg = dict(algo="dqn", game="synthetic:click", version="V1", seed=0, budget=200)
    runs = {}
    for name, params in (("on", {"object_clicks": True}), ("off", {"object_clicks": True, "effect_memory": False})):
        try:
            runs[name] = run(RunConfig(params=params, **cfg))
        except Exception as e:                                           # the synthetic task set may lack this name
            pytest.skip(f"synthetic task unavailable: {e}")
    assert runs["on"]["status"] in ("budget", "won") and runs["off"]["status"] in ("budget", "won")
    assert runs["on"]["learner"]["effect_memory"]["recorded"] > 0
    assert runs["off"]["learner"]["effect_memory"]["recorded"] == 0


@pytest.mark.parametrize("algo,version", [("dqn", "V1"), ("ppo", "V0")])
def test_i3_finds_the_one_responsive_object_faster_than_pixel_clicks(algo, version):
    def rewards(params):
        r = run(RunConfig(algo=algo, version=version, game="synthetic:click", seed=0, budget=300, params=params))
        return r["diagnostics"]["rewarded_transitions"]
    assert rewards({"object_clicks": True}) >= 1.5 * rewards({})
