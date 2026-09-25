"""Deterministic development / reserved split of the 25 public games (PROJECT_PLAN 3.3).

Strata = action-set type at reset (docs/evidence/action_space.txt): click-only, keyboard-only, mixed.
Reserved = 8 games, allocated proportionally per stratum, picked with a fixed seed.
"Reserved" means reserved from FURTHER tuning; every game was already exposed (docs/EXPOSURE.md).
"""
import hashlib, json, random
import arc_agi
from arc_agi import OperationMode

SEED = 2026
N_RESERVED = {"click": 2, "keyboard": 2, "mixed": 4}
# Rule: a game whose completion trajectory was inspected (reset-probe replay) is never reserved.
TRAJECTORY_INSPECTED = {"ft09"}

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
strata = {"click": [], "keyboard": [], "mixed": []}
for e in sorted(arc.get_environments(), key=lambda e: e.game_id):
    g = e.game_id.split("-")[0]
    av = set(arc.make(g).reset().available_actions)
    has_click, has_key = 6 in av, bool(av - {6, 7} - {6}) and bool(av & {1, 2, 3, 4, 5})
    strata["click" if has_click and not has_key else "keyboard" if has_key and not has_click else "mixed"].append(g)

rng = random.Random(SEED)
reserved, dev = [], []
for s, games in strata.items():
    pick = set(rng.sample([g for g in games if g not in TRAJECTORY_INSPECTED], N_RESERVED[s]))
    reserved += sorted(pick)
    dev += sorted(set(games) - pick)
out = {"schema_version": 1, "status": "proposed - awaiting team confirmation", "seed": SEED,
       "strata": strata, "dev": sorted(dev), "reserved": sorted(reserved)}
out["hash"] = hashlib.sha256(json.dumps({k: out[k] for k in ("seed", "dev", "reserved")}, sort_keys=True).encode()).hexdigest()[:12]
json.dump(out, open("configs/splits.json", "w"), indent=2)
print(json.dumps(out, indent=2))
