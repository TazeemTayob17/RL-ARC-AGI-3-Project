"""Probe: capture the exact ft09 level-completion transition (replayable, seed 7)."""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("ft09")
f = env.reset()
acts = env.action_space
rng = np.random.RandomState(7)
prev_lvl = 0
hist = []
for i in range(2100):
    a = acts[rng.randint(len(acts))]
    r = env.step(a, data={"x": int(rng.randint(64)), "y": int(rng.randint(64))}) \
        if a.is_complex() else env.step(a)
    hist.append((i, a.name, r.state.value, r.levels_completed, r.win_levels,
                 len(r.frame), list(r.available_actions), r.full_reset))
    if r.levels_completed > prev_lvl:
        print(f"*** LEVEL COMPLETION at step {i}: "
              f"levels_completed {prev_lvl} -> {r.levels_completed} (win_levels={r.win_levels})")
        print()
        print(f"{'step':>6} {'action':<8} {'state':<13} {'lvl':>3} {'win':>3} "
              f"{'n_frames':>8} {'avail':<12} {'full_reset'}")
        for row in hist[max(0, i - 4): i + 2]:
            mark = "  <== " if row[0] == i else "      "
            print(f"{row[0]:>6} {row[1]:<8} {row[2]:<13} {row[3]:>3} {row[4]:>3} "
                  f"{row[5]:>8} {str(row[6]):<12} {row[7]}{mark}")
        break
    if r.state.value in ("GAME_OVER", "WIN"):
        env.reset()
        acts = env.action_space or acts
    prev_lvl = r.levels_completed
else:
    print("no level completion within budget")
