"""Phase 1 reset probe (PROJECT_PLAN 3.1).

Diagnostic replay of the recorded ft09 seed-7 random trajectory (level_transition.py) to reach
levels_completed=1, then: one reset after an action, an immediate second reset, and a reset
after GAME_OVER. Also reads the SDK scorecard to see how resets/actions are counted.
The replay is NOT an agent policy.
"""
import json
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ft09")
cmds = {"step": 0, "reset": 0}


def reset():
    cmds["reset"] += 1
    return env.reset()


def step(a, x=None, y=None):
    cmds["step"] += 1
    return env.step(a, data={"x": int(x), "y": int(y)}) if a.is_complex() else env.step(a)


def snap(tag, r, ref=None):
    fr = r.frame[-1] if len(r.frame) else None
    row = dict(tag=tag, state=r.state.value, levels_completed=r.levels_completed,
               win_levels=r.win_levels, n_frames=len(r.frame), full_reset=r.full_reset,
               avail=list(r.available_actions))
    if fr is not None and ref:
        row.update({k: bool(np.array_equal(fr, v)) for k, v in ref.items()})
    print(json.dumps(row))
    return row


f = reset()
level0_frame = f.frame[-1].copy()
acts = env.action_space
rng = np.random.RandomState(7)
prev = 0
for i in range(2100):
    a = acts[rng.randint(len(acts))]
    r = step(a, rng.randint(64), rng.randint(64)) if a.is_complex() else step(a)
    if r.levels_completed > prev:
        break
    if r.state.value in ("GAME_OVER", "WIN"):
        reset()
        acts = env.action_space or acts
    prev = r.levels_completed
assert r.levels_completed == 1, "replay did not reproduce level completion"
print(f"replayed to level 1 at step index {i}; commands so far {cmds}")
level1_first = r.frame[-1].copy()
snap("A. at level-1 start (replay end)", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})

rng2 = np.random.RandomState(11)
a = acts[rng2.randint(len(acts))]
r = step(a, rng2.randint(64), rng2.randint(64)) if a.is_complex() else step(a)
snap(f"B. after 1 action ({a.name})", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})

r = reset()
snap("C. reset #1 (mid level 1)", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})
r = reset()
snap("D. reset #2 immediately", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})

# progress again, then drive to GAME_OVER from level >= 1 and reset
n = 0
while r.state.value not in ("GAME_OVER", "WIN") and n < 3000:
    a = acts[rng2.randint(len(acts))]
    r = step(a, rng2.randint(64), rng2.randint(64)) if a.is_complex() else step(a)
    n += 1
snap(f"E. terminal after {n} more actions", r)
r = reset()
snap("F. reset after GAME_OVER", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})
r = reset()
snap("G. second reset after that", r, {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})

# ---- Part 2: reset after GAME_OVER while at levels_completed=1 (fresh env, same replay)
print("--- part 2: GAME_OVER at level 1 ---")
arc2 = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc2.make("ft09")
r = reset()
acts = env.action_space
rng = np.random.RandomState(7)
prev = 0
for i in range(2100):
    a = acts[rng.randint(len(acts))]
    r = step(a, rng.randint(64), rng.randint(64)) if a.is_complex() else step(a)
    if r.levels_completed > prev:
        break
    if r.state.value in ("GAME_OVER", "WIN"):
        reset()
        acts = env.action_space or acts
    prev = r.levels_completed
assert r.levels_completed == 1
rng3 = np.random.RandomState(5)
n = 0
while r.state.value not in ("GAME_OVER", "WIN") and n < 5000:
    a = acts[rng3.randint(len(acts))]
    r = step(a, rng3.randint(64), rng3.randint(64)) if a.is_complex() else step(a)
    n += 1
snap(f"H. terminal at level>=1 after {n} more actions", r)
r = reset()
snap("I. reset after GAME_OVER (from level 1)", r,
     {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})
r = reset()
snap("J. second reset after that", r,
     {"eq_level1_first": level1_first, "eq_level0_init": level0_frame})

print("env commands issued:", cmds)
card = arc.get_scorecard()
for e in card.environments:
    for run in e.runs:
        print("scorecard run:", run.model_dump_json())
