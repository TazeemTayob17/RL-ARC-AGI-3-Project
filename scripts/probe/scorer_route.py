"""Probe: does the pinned SDK's local scorer have human baselines for the 25 public games?"""
import arc_agi
from arc_agi import OperationMode

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
envs = arc.get_environments()
print("N envs:", len(envs))
print("EnvironmentInfo fields:", list(type(envs[0]).model_fields))
n_have = 0
for e in envs:
    ba = e.baseline_actions
    ok = bool(ba)
    n_have += ok
    print(f"{e.game_id:<16} baseline_actions={ba} tags={e.tags}")
print("games with baselines:", n_have, "/", len(envs))
