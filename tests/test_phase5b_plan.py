import json
from pathlib import Path

import pytest

from scripts.run_phase4 import plan as plan4
from scripts.run_phase5b import BETA, plan

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "configs" / "splits.json").read_text())


def test_v3_equals_phase4_i1_config_plus_i3_and_a3_equals_baseline_plus_i3():
    game = SPLIT["dev"][0]
    i1 = {(c.algo, c.seed): c for c in plan4("i1", [game], [0, 1], 1500, ["dqn_i1", "ppo_i1"])}
    for c in plan([game], [0, 1], 1500, ["dqn_v3", "ppo_v3"]):
        b = i1[(c.algo, c.seed)]
        assert (c.version, c.budget, c.arch, c.threads, c.updates_disabled) == \
               (b.version, b.budget, b.arch, b.threads, b.updates_disabled)
        assert c.params == {**b.params, "object_clicks": True, "effect_memory": True, "stage": "V3"}
        assert c.params["intrinsic_beta"] == BETA
    for c in plan([game], [0, 1], 1500, ["dqn_a3", "ppo_a3"]):
        assert c.params == {"object_clicks": True, "effect_memory": True, "stage": "A3"}          # no I1 in A3
    for c in plan([game], [0], 1500, ["dqn_v3nm", "ppo_v3nm"]):
        assert c.params["effect_memory"] is False and c.params["object_clicks"] is True


def test_reserved_games_and_bad_conditions_are_refused():
    with pytest.raises(ValueError):
        plan([SPLIT["reserved"][0]], [0], 1500, ["dqn_v3"])
    with pytest.raises(ValueError):
        plan(["ft09"], [0], 1500, ["dqn_v2"])
