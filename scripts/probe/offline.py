"""Probe: does OperationMode.OFFLINE work with no network calls?"""
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
envs = arc.get_environments()
print("OFFLINE mode: environments discovered =", len(envs))
print("ids:", sorted({e.game_id.split('-')[0] for e in envs}))

env = arc.make("ls20")
f = env.reset()
print("reset ok: state =", f.state.value, "| levels_completed =", f.levels_completed,
      "| win_levels =", f.win_levels, "| n_frames =", len(f.frame))
for _ in range(20):
    a = env.action_space[0]
    r = env.step(a)
print("20 steps ok: state =", r.state.value)
print("OFFLINE: PASS")
