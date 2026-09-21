"""Probe: GAME_OVER semantics, RESET recovery, and level progression.

Random policy with proper episode handling (reset on every terminal state),
recording GAME_OVER frequency and any `levels_completed` increment.
"""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

GAMES = ["ls20", "vc33", "ft09", "tr87", "sb26"]
BUDGET = 20000

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
summary = []
for game in GAMES:
    env = arc.make(game)
    f = env.reset()
    acts = env.action_space or [GameAction.ACTION1]
    rng = np.random.RandomState(7)
    probed = False
    episodes = 0
    n_go = n_win = 0
    best_lvl = 0
    first_lvl = None
    ep_len = []
    cur = 0
    for i in range(BUDGET):
        a = acts[rng.randint(len(acts))]
        r = env.step(a, data={"x": int(rng.randint(64)), "y": int(rng.randint(64))}) \
            if a.is_complex() else env.step(a)
        cur += 1
        if r.levels_completed > best_lvl:
            if first_lvl is None:
                first_lvl = (i, best_lvl, r.levels_completed, r.state.value, len(r.frame))
            best_lvl = r.levels_completed
        if r.state.value in ("GAME_OVER", "WIN"):
            if r.state.value == "GAME_OVER":
                n_go += 1
            else:
                n_win += 1
            if not probed:                       # one-time semantics probe
                probed = True
                print(f"\n[{game}] first {r.state.value} at step {i}")
                print(f"   available_actions: {r.available_actions}")
                try:
                    rr = env.step(GameAction.ACTION1)
                    print(f"   step(ACTION1) while terminal -> accepted, state={rr.state.value}, "
                          f"n_frames={len(rr.frame)}, frame_empty={rr.is_empty()}")
                except Exception as ex:
                    print(f"   step(ACTION1) while terminal -> {type(ex).__name__}: {str(ex)[:90]}")
            rs = env.reset()
            print_once = None
            acts = env.action_space or acts
            episodes += 1
            ep_len.append(cur)
            cur = 0
    summary.append((game, episodes, n_go, n_win, best_lvl, first_lvl,
                    int(np.mean(ep_len)) if ep_len else -1))

print()
print("=" * 86)
print(f"{'game':<7} {'episodes':>9} {'GAME_OVER':>10} {'WIN':>5} {'best lvl':>9} {'mean ep len':>12}  first level-up")
print("-" * 86)
for g, eps, go, win, bl, fl, ml in summary:
    fltxt = f"step {fl[0]}: {fl[1]}->{fl[2]}" if fl else "never"
    print(f"{g:<7} {eps:>9} {go:>10} {win:>5} {bl:>9} {ml:>12}  {fltxt}")
print()
print(f"(random policy, {BUDGET} steps/game, reset on every terminal state)")
