"""Probe: enumerate locally-available ARC-AGI-3 environments.

Tests research claim: "25 public games" with a specific ID list.
"""
import arc_agi
from arc_agi import OperationMode

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
envs = arc.get_environments()

print(f"COUNT: {len(envs)}")
print(f"{'game_id':<24} {'short':<8} title")
print("-" * 70)
for e in sorted(envs, key=lambda x: x.game_id):
    short = e.game_id.split("-")[0]
    print(f"{e.game_id:<24} {short:<8} {getattr(e, 'title', '?')}")

print()
print("SHORT_IDS:", sorted({e.game_id.split("-")[0] for e in envs}))
