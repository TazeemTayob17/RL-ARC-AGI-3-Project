import json
from pathlib import Path

import pytest

from scripts.run_phase5b import plan as plan5b
from scripts.run_phase5c import BETA, plan

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "configs" / "splits.json").read_text())


def test_v4_equals_v3_config_plus_the_option_and_a4_equals_baseline_plus_the_option():
    game = SPLIT["dev"][0]
    v3 = {(c.algo, c.seed): c for c in plan5b([game], [0, 1], 1500, ["dqn_v3", "ppo_v3"])}
    for c in plan([game], [0, 1], 1500, ["dqn_v4", "ppo_v4"]):
        b = v3[(c.algo, c.seed)]
        assert (c.version, c.budget, c.arch, c.threads, c.updates_disabled) == \
               (b.version, b.budget, b.arch, b.threads, b.updates_disabled)
        assert c.params == {**b.params, "frontier_mode": "learned", "stage": "V4"}
        assert c.params["intrinsic_beta"] == BETA and c.params["object_clicks"] is True
    for c in plan([game], [0, 1], 1500, ["dqn_v4f", "ppo_v4f"]):
        assert c.params["frontier_mode"] == "forced" and c.params["stage"] == "V4f" and c.params["object_clicks"] is True
    for c in plan([game], [0, 1], 1500, ["dqn_a4", "ppo_a4"]):
        assert c.params == {"frontier_mode": "learned", "stage": "A4"}          # no I1 and no I3 in A4


def test_reserved_games_and_bad_conditions_are_refused():
    with pytest.raises(ValueError):
        plan([SPLIT["reserved"][0]], [0], 1500, ["dqn_v4"])
    with pytest.raises(ValueError):
        plan(["ft09"], [0], 1500, ["dqn_v3"])
