import json
from pathlib import Path

import pytest

from scripts.run_phase3 import plan as plan3
from scripts.run_phase4 import BETA, BUDGET_GAMES, plan

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "configs" / "splits.json").read_text())


def test_i1_configs_differ_from_phase3_only_in_i1_params_and_tag():
    game = SPLIT["dev"][0]
    base = {(c.algo, c.seed): c for c in plan3([game], [0, 1], 1500, 32, ["dqn", "ppo"], "V1", 1)}
    for c in plan("i1", [game], [0, 1], 1500, ["dqn_i1", "ppo_i1"]):
        b = base[(c.algo, c.seed)]
        assert (c.version, c.budget, c.arch, c.threads, c.updates_disabled) == \
               (b.version, b.budget, b.arch, b.threads, b.updates_disabled)
        assert c.params == {"intrinsic_beta": BETA, "stage": "V1"} and c.hash() != b.hash()


def test_baseline_configs_keep_their_phase3_hash():
    game = SPLIT["dev"][0]
    for c3 in plan3([game], [0], 1500, 32, ["dqn", "ppo", "dqn_no_updates"], "V1", 1):
        assert c3.hash() == c3.resolved().hash()        # hash stable under resolution; no new fields leak in


def test_reserved_games_are_refused_and_budget_subset_is_dev():
    reserved = SPLIT["reserved"][0]
    with pytest.raises(ValueError):
        plan("i1", [reserved], [0], 1500, ["dqn_i1"])
    assert all(g in SPLIT["dev"] for g in BUDGET_GAMES.split(","))
    with pytest.raises(ValueError):
        plan("budget", ["ft09"], [0], 1500, ["dqn_i1"])
