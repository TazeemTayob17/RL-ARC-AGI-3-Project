"""Probe: environment throughput with no rendering."""
import time
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

GAMES = ["ls20", "vc33", "ft09", "tr87"]
N = 2000

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
print(f"{'game':<7} {'steps':>6} {'raw step/s':>12} {'+numpy step/s':>15}")
print("-" * 44)
tot_raw = tot_np = 0.0
for game in GAMES:
    for convert in (False, True):
        env = arc.make(game)          # render_mode=None -> no rendering
        env.reset()
        acts = env.action_space or [GameAction.ACTION1]
        rng = np.random.RandomState(0)
        t0 = time.perf_counter()
        for _ in range(N):
            a = acts[rng.randint(len(acts))]
            r = env.step(a, data={"x": 32, "y": 32}) if a.is_complex() else env.step(a)
            if r.state.value in ("WIN", "GAME_OVER"):
                env.reset()
            if convert:
                _ = np.asarray(r.frame[-1], dtype=np.int8)   # what an agent actually pays
        dt = time.perf_counter() - t0
        if convert:
            sp_np = N / dt
        else:
            sp_raw = N / dt
    tot_raw += sp_raw; tot_np += sp_np
    print(f"{game:<7} {N:>6} {sp_raw:>12.0f} {sp_np:>15.0f}")

print("-" * 44)
print(f"{'mean':<7} {'':>6} {tot_raw/len(GAMES):>12.0f} {tot_np/len(GAMES):>15.0f}")
