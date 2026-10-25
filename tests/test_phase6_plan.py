import json
from pathlib import Path

import pytest

from scripts.run_phase4 import plan as plan4
from scripts.run_phase5b import plan as plan5b
from scripts.run_phase5d import plan as plan5d
from scripts.run_phase6 import CONDITIONS, plan

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "configs" / "splits.json").read_text())


def test_only_reserved_games_are_accepted():
    with pytest.raises(ValueError):
        plan([SPLIT["dev"][0]], [0], 1500, ["dqn_v0"])
    assert len(plan(SPLIT["reserved"], [0, 1, 2, 3, 4], 1500, list(CONDITIONS))) == 8 * 5 * 9


def test_final_versions_equal_the_development_versions_except_the_stage_label():
    g, d = SPLIT["reserved"][0], [SPLIT["dev"][0]]
    strip = lambda p: {k: v for k, v in p.items() if k != "stage"}
    cases = [("dqn_v0", plan4("budget", d, [0], 1500, ["dqn"], 0.0)[0]),
             ("dqn_v1", plan4("i1", d, [0], 1500, ["dqn_i1"])[0]),
             ("ppo_v1", plan4("i1", d, [0], 1500, ["ppo_i1"])[0]),
             ("dqn_v2", plan5b(d, [0], 1500, ["dqn_v3"])[0]),
             ("ppo_v2", plan5b(d, [0], 1500, ["ppo_v3"])[0]),
             ("dqn_a2", plan5b(d, [0], 1500, ["dqn_a3"])[0]),
             ("ppo_a2", plan5b(d, [0], 1500, ["ppo_a3"])[0]),
             ("dqn_v2np", plan5d(d, [0], 1500, ["dqn_v3np"])[0])]
    for cond, dev in cases:
        c = plan([g], [0], 1500, [cond])[0]
        assert strip(c.params) == strip(dev.params), cond
        assert (c.algo, c.version, c.budget, c.arch, c.threads, c.updates_disabled) == \
               (dev.algo, dev.version, dev.budget, dev.arch, dev.threads, dev.updates_disabled), cond


def test_ppo_has_no_prior_free_condition():
    with pytest.raises(ValueError):
        plan([SPLIT["reserved"][0]], [0], 1500, ["ppo_v2np"])
