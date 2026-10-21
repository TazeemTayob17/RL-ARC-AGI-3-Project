"""I1 activation check (plan 6, bug procedure step 2): did the bonus fire, reach learning, and change behaviour?

For every I1 run found in --dir, find the Phase 3 baseline run (same algo/game/seed/budget) and report:
  * bonus statistics (calls, nonzero, loop-zeroed, total bonus);
  * index of the first action that differs from the baseline trace (identical earlier actions are expected:
    an intrinsic reward can only change behaviour after training starts);
  * gradient steps and learning-signal diagnostics versus baseline.
Exit status 1 if any run has a zero bonus, or never diverges from its baseline.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def baseline_dir(rec, base_root):
    c = rec["identity"]["config"]
    for d in base_root.iterdir():
        p = d / "record.json"
        if not p.exists() or not d.name.startswith(f"{c['algo']}-{c['version']}-{c['game']}-s{c['seed']}-"):
            continue
        b = json.loads(p.read_text())["identity"]["config"]
        if (b["budget"] == c["budget"] and not b["updates_disabled"] and not b["params"].get("intrinsic_beta")
                and b["arch"] == c["arch"]):
            return d
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase4_smoke")
    ap.add_argument("--baseline", default="results/phase3_selected_baseline")
    a = ap.parse_args(argv)
    bad = 0
    for d in sorted(Path(a.dir).iterdir()):
        p = d / "record.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        learner = rec["learner"]
        intr = learner["intrinsic"]
        base = baseline_dir(rec, Path(a.baseline))
        line = {"run": d.name, "calls": intr["calls"], "nonzero": intr["nonzero"],
                "loop_zeroed": intr["loop_zeroed"], "total_bonus": round(intr["total_bonus"], 3),
                "gradient_steps": learner.get("gradient_steps")}
        if base is None:
            line["baseline"] = "NOT FOUND"
            bad += 1
        else:
            brec = json.loads((base / "record.json").read_text())
            x, y = np.load(d / "actions.npy"), np.load(base / "actions.npy")
            n = min(len(x), len(y))
            diff = np.flatnonzero(x[:n] != y[:n])
            line["first_divergence"] = int(diff[0]) if len(diff) else None
            line["differing_actions"] = int(len(diff))
            line["baseline_gradient_steps"] = brec["learner"].get("gradient_steps")
            for key in ("q_mean_ema", "loss_ema", "vf_ema", "entropy_ema"):
                if key in learner:
                    line[key] = (round(learner[key], 5) if learner[key] is not None else None,
                                 round(brec["learner"][key], 5) if brec["learner"].get(key) is not None else None)
            if not len(diff):
                bad += 1
        if intr["nonzero"] == 0:
            bad += 1
        print(json.dumps(line))
    print("ACTIVATION OK" if not bad else f"ACTIVATION PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
