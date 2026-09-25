"""Phase 1 scorer probe (PROJECT_PLAN 3.1/3.2): how does the SDK scorecard count commands?

Fresh scorecard on ls20 (keyboard game), known command sequence; print scorecard after each phase.
"""
import json
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ls20")


def show(tag, sent):
    card = arc.get_scorecard()
    runs = [r for e in card.environments for r in e.runs]
    print(f"{tag:<44} sent={sent} | n_runs={len(runs)} | " + " ; ".join(
        f"actions={r.actions} resets={r.resets} lvl={r.levels_completed} state={r.state.value}"
        for r in runs))


sent = {"step": 0, "reset": 0}
env.reset(); sent["reset"] += 1
show("after first reset()", sent)
for _ in range(10):
    env.step(GameAction.ACTION1); sent["step"] += 1
show("after 10 x ACTION1", sent)
env.reset(); sent["reset"] += 1
show("after reset (level restart)", sent)
for _ in range(5):
    env.step(GameAction.ACTION1); sent["step"] += 1
show("after 5 x ACTION1", sent)
env.reset(); sent["reset"] += 1
env.reset(); sent["reset"] += 1
show("after 2 consecutive resets (full reset)", sent)
for _ in range(3):
    env.step(GameAction.ACTION1); sent["step"] += 1
show("after 3 x ACTION1", sent)
card = arc.get_scorecard()
print("card.score:", card.score, "| environments:", len(card.environments))
print(card.model_dump_json()[:1500])

# ---- Part 2: unavailable-action (no-op) and post-GAME_OVER accounting on ft09 (available=[6] only)
import numpy as np
print("--- part 2: ft09 no-op and post-GAME_OVER accounting ---")
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ft09")
env.reset()


def n_actions():
    card = arc.get_scorecard()
    return [(r.actions, r.resets) for e in card.environments for r in e.runs]


rng = np.random.RandomState(7)
sent = 0
r = env.step(GameAction.ACTION6, data={"x": 0, "y": 0}); sent += 1
print("after 1 valid ACTION6:", n_actions(), "sent", sent)
r = env.step(GameAction.ACTION1); sent += 1
print("after 1 unavailable ACTION1 (state=%s):" % r.state.value, n_actions(), "sent", sent)
while r.state.value != "GAME_OVER":
    r = env.step(GameAction.ACTION6, data={"x": int(rng.randint(64)), "y": int(rng.randint(64))}); sent += 1
print("at GAME_OVER:", n_actions(), "sent", sent)
for _ in range(3):
    r = env.step(GameAction.ACTION6, data={"x": 5, "y": 5}); sent += 1
print("after 3 steps sent during GAME_OVER (n_frames=%d):" % len(r.frame), n_actions(), "sent", sent)
