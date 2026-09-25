"""Step 8: synthetic learning checks. Both learners must improve; random and updates-disabled must not.

  python scripts/synthetic_checks.py --tasks nav,click,cue --budget 6000 --seeds 0,1,2 --out results/synthetic
Reward per action over the final third of the run is the improvement metric (whole-run rate is also printed).
These are implementation checks, not ARC scores.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RunConfig
from src.run import run

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="nav,click,cue")
ap.add_argument("--algos", default="dqn,ppo")
ap.add_argument("--budget", type=int, default=6000)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--channels", type=int, default=8)
ap.add_argument("--dqn", default="{}", help="json overrides for DQN params")
ap.add_argument("--ppo", default="{}", help="json overrides for PPO params")
ap.add_argument("--out", default=None)
ap.add_argument("--controls", action="store_true", help="also run updates-disabled copies")
a = ap.parse_args()

ARCH = {"id": "onehotcnn_stack2", "channels": a.channels, "context_dim": 32}
PARAMS = {"dqn": {"eps_decay_actions": a.budget // 2, "replay_capacity": 20000, **json.loads(a.dqn)},
          "ppo": json.loads(a.ppo), "random_type": {}}


def rates(rec, budget):
    curve = rec["diagnostics"]["progress_curve"][1:]
    hits = [c[0] for c in curve]
    last = sum(h > 2 * budget / 3 for h in hits)
    return len(hits) / budget, last / (budget / 3)


rows = []
t0 = time.time()
for task in a.tasks.split(","):
    variants = [("random_type", False)] + [(al, False) for al in a.algos.split(",")]
    if a.controls:
        variants += [(al, True) for al in a.algos.split(",")]
    for algo, off in variants:
        whole, final = [], []
        for seed in map(int, a.seeds.split(",")):
            cfg = RunConfig(algo=algo, game=f"synthetic:{task}", budget=a.budget, seed=seed, arch=dict(ARCH),
                            params=PARAMS[algo], updates_disabled=off, version="V0", tag="synthetic")
            rec = run(cfg, out_dir=a.out)
            w, f = rates(rec, a.budget)
            whole.append(w); final.append(f)
        label = f"{algo}{'(updates disabled)' if off else ''}"
        rows.append((task, label, np.mean(whole), np.mean(final), final))
        print(f"{task:<6} {label:<28} reward/action whole={np.mean(whole):.4f} final-third={np.mean(final):.4f} "
              f"per-seed final={[round(x, 4) for x in final]}  [{time.time() - t0:.0f}s]", flush=True)
