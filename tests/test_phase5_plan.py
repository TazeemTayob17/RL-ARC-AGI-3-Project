import json
from pathlib import Path

import pytest

from scripts.run_phase4 import plan as plan4
from scripts.run_phase5 import BETA, OPTION_CAP, OPTION_PROB, plan

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "configs" / "splits.json").read_text())


def test_v2_equals_phase4_i1_config_plus_option_and_a2_equals_baseline_plus_option():
    game = SPLIT["dev"][0]
    i1 = {(c.algo, c.seed): c for c in plan4("i1", [game], [0, 1], 1500, ["dqn_i1", "ppo_i1"])}
    for c in plan([game], [0, 1], 1500, ["dqn_v2", "ppo_v2"]):
        b = i1[(c.algo, c.seed)]
        assert (c.version, c.budget, c.arch, c.threads, c.updates_disabled) == \
               (b.version, b.budget, b.arch, b.threads, b.updates_disabled)
        assert c.params == {**b.params, "option_cap": OPTION_CAP, "option_prob": OPTION_PROB, "stage": "V2"}
        assert c.params["intrinsic_beta"] == BETA and c.hash() != b.hash()
    base = {(c.algo, c.seed): c for c in plan4("budget", [game], [0, 1], 1500, ["dqn", "ppo"], 0.0)}
    for c in plan([game], [0, 1], 1500, ["dqn_a2", "ppo_a2"]):
        b = base[(c.algo, c.seed)]
        assert (c.version, c.budget, c.arch, c.threads) == (b.version, b.budget, b.arch, b.threads)
        assert c.params == {"option_cap": OPTION_CAP, "option_prob": OPTION_PROB, "stage": "A2"}      # no I1 in A2


def test_reserved_games_and_bad_conditions_are_refused():
    with pytest.raises(ValueError):
        plan([SPLIT["reserved"][0]], [0], 1500, ["dqn_v2"])
    with pytest.raises(ValueError):
        plan(["ft09"], [0], 1500, ["dqn_i1"])
    with pytest.raises(ValueError):
        plan(["ft09"], [0], 0, ["dqn_v2"])


def test_option_parameters_are_the_predeclared_ones():
    assert (OPTION_CAP, OPTION_PROB) == (8, 0.5)
