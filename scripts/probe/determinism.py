"""Probe: same actions from same state -> same frames?

Replays one fixed action sequence over several fresh episodes and compares
SHA-256 of every returned grid, step by step.
"""
import hashlib
import sys
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

GAMES = ["ls20", "vc33", "ft09"]
N = 40


def rollout(arc, game, seed=0):
    env = arc.make(game, seed=seed)
    f = env.reset()
    acts = env.action_space or [GameAction.ACTION1]
    digests = []
    rng = np.random.RandomState(1234)          # fixed action sequence
    for i in range(N):
        a = acts[rng.randint(len(acts))]
        r = env.step(a, data={"x": int(rng.randint(64)), "y": int(rng.randint(64))}) \
            if a.is_complex() else env.step(a)
        blob = b"".join(np.ascontiguousarray(g, dtype=np.int8).tobytes() for g in r.frame)
        digests.append((a.name, r.state.value, r.levels_completed,
                        hashlib.sha256(blob).hexdigest()[:16]))
    return digests


arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
overall = True
for game in GAMES:
    a1 = rollout(arc, game)
    a2 = rollout(arc, game)
    same = a1 == a2
    overall &= same
    first_div = next((i for i, (x, y) in enumerate(zip(a1, a2)) if x != y), None)
    print(f"{game}: {N} steps, run1==run2 -> {same}"
          + ("" if same else f"  FIRST DIVERGENCE at step {first_div}: {a1[first_div]} vs {a2[first_div]}"))
    if "--dump" in sys.argv:
        print("   final:", a1[-1])

print()
print("IN-PROCESS DETERMINISM:", "PASS" if overall else "FAIL")

# emit digests for cross-process comparison
if "--emit" in sys.argv:
    for game in GAMES:
        d = rollout(arc, game)
        print(f"XPROC {game} {d[-1][3]} final_state={d[-1][1]} lvl={d[-1][2]}")
