"""I4 activation check (plan 6, bug procedure step 2): did the frontier option actually operate?

For every run in --dir report how often the option was offered and chosen, the steps spent inside options, the share of
walks that reached the expected node, and the first action that differs from the matching reference run (same algo, game,
seed): the Phase 5b V3 record for V4 and V4f, the Phase 3 baseline record for A4.

Exit 1 if a run has no frontier diagnostics, the option was never offered, a learned run never took the option while it
was offered (counted over the whole grid: fewer than 80% of learned runs with at least one option), a forced run never
took it, or a run whose option was offered never diverges from its reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase5b import plan as plan5b


def reference_dir(rec, p5b, phase3):
    c = rec["identity"]["config"]
    stage = rec["identity"]["stage"]
    if stage in ("V4", "V4f"):
        cond = f"{c['algo']}_v3"
        cfg = plan5b([c["game"]], [c["seed"]], c["budget"], [cond])[0]
        return Path(p5b) / cfg.run_id()
    for d in Path(phase3).glob(f"{c['algo']}-{c['version']}-{c['game']}-s{c['seed']}-*"):
        b = json.loads((d / "record.json").read_text())["identity"]["config"]
        if b["budget"] == c["budget"] and not b["updates_disabled"] and b["arch"] == c["arch"] and not b["params"].get("intrinsic_beta"):
            return d
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase5c_improvement4")
    ap.add_argument("--phase5b", default="results/phase5b_improvement3")
    ap.add_argument("--phase3", default="results/phase3_selected_baseline")
    a = ap.parse_args(argv)
    bad = learned = learned_with_option = 0
    for d in sorted(Path(a.dir).iterdir()):
        p = d / "record.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        stage = rec["identity"]["stage"]
        f = rec["learner"].get("frontier")
        line = {"run": d.name, "stage": stage}
        if not f:
            line["problem"] = "frontier wrapper not active"
            bad += 1
            print(json.dumps(line))
            continue
        line.update({"offered": f["decisions_with_option"], "chosen": f["options_started"],
                     "option_steps": f["option_steps"], "mean_len": round(f["mean_option_length"], 2),
                     "path_success": round(f["path_success_rate"], 3), "nodes": f["graph_nodes"],
                     "stops": f["stop_reasons"]})
        if f["decisions_with_option"] == 0:
            line["problem"] = "option never offered"
            bad += 1
        if stage == "V4f":
            if f["options_started"] == 0:
                line["problem"] = "forced option never taken"
                bad += 1
        else:
            learned += 1
            learned_with_option += f["options_started"] > 0
        ref = reference_dir(rec, a.phase5b, a.phase3)
        if ref is None or not (ref / "actions.npy").exists():
            line["reference"] = "NOT FOUND"
            bad += 1
        else:
            x, y = np.load(d / "actions.npy"), np.load(ref / "actions.npy")
            n = min(len(x), len(y))
            diff = np.flatnonzero(x[:n] != y[:n])
            line["first_divergence"] = int(diff[0]) if len(diff) else None
            if f["decisions_with_option"] > 0 and not len(diff):
                line["problem"] = "actions identical to the reference although the option was offered"
                bad += 1
        print(json.dumps(line))
    if learned and learned_with_option < 0.8 * learned:
        print(f"learned runs that took the option at least once: {learned_with_option}/{learned} (< 80%)")
        bad += 1
    print("ACTIVATION OK" if not bad else f"ACTIVATION PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
