"""Probe: action space across every public game.

Tests claims: GameAction members, ACTION7 exists, available_actions varies per
game, ACTION6 coordinate passing.
"""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

print("GameAction members:")
for a in GameAction:
    print(f"  {a.name:<8} id={a.value} type={a.action_type.__name__:<14} "
          f"simple={a.is_simple()} complex={a.is_complex()}")
print("ACTION7 exists    :", hasattr(GameAction, "ACTION7"))
print("all_simple        :", [a.name for a in GameAction.all_simple()])
print("all_complex       :", [a.name for a in GameAction.all_complex()])
print()

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
envs = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})

print(f"{'game':<7} {'win_levels':>10} {'available_actions':<24} {'n_frames@reset':>14}")
print("-" * 62)
rows = []
for g in envs:
    try:
        env = arc.make(g)
        f = env.reset()
        names = [GameAction.from_id(i).name.replace("ACTION", "A") for i in f.available_actions]
        rows.append((g, f.win_levels, f.available_actions))
        print(f"{g:<7} {f.win_levels:>10} {','.join(names):<24} {len(f.frame):>14}")
    except Exception as ex:
        print(f"{g:<7} ERROR {type(ex).__name__}: {ex}")

print()
print("DISTINCT available_actions sets:")
for s in sorted({tuple(r[2]) for r in rows}):
    members = [g for g, _, a in rows if tuple(a) == s]
    print(f"  {list(s)} -> {members}")
print()
print("Games exposing ACTION6 (click):", [g for g, _, a in rows if 6 in a])
print("Games exposing ACTION7 (undo):", [g for g, _, a in rows if 7 in a])
