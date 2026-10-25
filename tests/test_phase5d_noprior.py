import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_phase5b import plan as plan5b
from scripts.run_phase5d import BETA, plan
from src.adapter import NO_ACTION, Observation, build_mask
from src.checkpoint_resolver import resolve_prior
from src.config import DQN_V1_DEFAULTS
from src.dqn_v1 import DQNV1
from src.dqn_v1.prior import random_prior

ROOT = Path(__file__).resolve().parents[1]
GAME = json.loads((ROOT / "configs" / "splits.json").read_text())["dev"][0]


def _obs(avail=(1, 2, 3, 4, 6, 7), seed=1):
    f = np.random.default_rng(seed).integers(0, 13, (64, 64), dtype=np.int8)
    return Observation(f, f.copy(), build_mask(avail), NO_ACTION, 0, 0, 5)


def _learner(params):
    spec = resolve_prior("ls20")
    return DQNV1({**DQN_V1_DEFAULTS, **params}, prior_path=spec.path, fold=spec.fold, seed_offset=spec.seed_offset)


def test_random_prior_is_fixed_game_independent_and_leaves_the_rng_stream_alone():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    a, b = random_prior(torch.device("cpu")), random_prior(torch.device("cpu"))
    assert torch.equal(torch.random.get_rng_state(), before)
    for (n, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(x, y), n
    assert not any(p.requires_grad for p in a.parameters())


def test_random_prior_differs_from_the_pretrained_prior_and_the_learner_acts_legally():
    d = _learner({"prior_mode": "random"})
    d.start_run(3)
    pre = _learner({})
    pre.start_run(3)
    assert not torch.equal(next(iter(d.prior.state_dict().values())), next(iter(pre.prior.state_dict().values())))
    for avail in ((6,), (1, 2, 3, 4), (3, 4, 6, 7)):
        o = _obs(avail, 9)
        assert o.mask[d.act(o)]
        d.pending = None
    assert d.diagnostics()["prior_checkpoint"].startswith("none")
    assert not pre.diagnostics()["prior_checkpoint"].startswith("none")


def test_plan_is_v3_and_baseline_params_plus_prior_mode_only():
    v3 = {c.seed: c for c in plan5b([GAME], [0, 1], 1500, ["dqn_v3"])}
    for c in plan([GAME], [0, 1], 1500, ["dqn_v3np"]):
        assert c.params == {**v3[c.seed].params, "prior_mode": "random", "stage": "V3np"}
        assert c.params["intrinsic_beta"] == BETA and (c.version, c.arch) == (v3[c.seed].version, v3[c.seed].arch)
    for c in plan([GAME], [0], 1500, ["dqn_v0np"]):
        assert c.params == {"prior_mode": "random", "stage": "V0np"}
