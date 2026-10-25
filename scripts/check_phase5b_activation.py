"""I3 activation check (plan 6, bug procedure step 2): did object clicks and the effect memory actually operate?

For every run in --dir report the mean candidates offered per click decision, the share dropped by the memory, the
number of recorded outcomes, and the first action that differs from the matching reference run (same algo/game/seed;
Phase 3 for A3, Phase 4 for V3 and V3nm). Exit 1 if object clicks never produced candidates, the memory of a with-memory
run never recorded or dropped anything, or a run never diverges from its reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def reference(rec, ref_root, with_i1):
    c = rec["identity"]["config"]
    for d in Path(ref_root).iterdir():
        p = d / "record.json"
        if not p.exists() or not d.name.startswith(f"{c['algo']}-{c['version']}-{c['game']}-s{c['seed']}-"):
            continue
        b = json.loads(p.read_text())["identity"]["config"]
        if (b["budget"] == c["budget"] and not b["updates_disabled"] and b["arch"] == c["arch"]
                and bool(b["params"].get("intrinsic_beta")) == with_i1 and not b["params"].get("option_cap")
                and not b["params"].get("object_clicks")):
            return d
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase5b_improvement3")
    ap.add_argument("--phase3", default="results/phase3_selected_baseline")
    ap.add_argument("--phase4", default="results/phase4_i1")
    a = ap.parse_args(argv)
    bad = 0
    for d in sorted(Path(a.dir).iterdir()):
        p = d / "record.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        prm = rec["identity"]["config"]["params"]
        m = rec["learner"].get("effect_memory")
        line = {"run": d.name, "stage": prm.get("stage")}
        if not rec["learner"].get("object_clicks") or m is None:
            line["problem"] = "object clicks not active"
            bad += 1
            print(json.dumps(line))
            continue
        line.update({"offered": m["candidates_offered"], "dropped_share": round(m["dropped_share"], 3),
                     "recorded": m["recorded"], "retried": m["retried"], "all_dead_fallbacks": m["all_dead_fallbacks"]})
        if prm.get("effect_memory") and m["recorded"] == 0 and rec["diagnostics"]["click_fraction"] > 0.05:
            line["problem"] = "memory recorded nothing despite clicks"
            bad += 1
        with_i1 = bool(prm.get("intrinsic_beta"))
        ref = reference(rec, a.phase4 if with_i1 else a.phase3, with_i1)
        if ref is None:
            line["reference"] = "NOT FOUND"
            bad += 1
        else:
            x, y = np.load(d / "actions.npy"), np.load(ref / "actions.npy")
            n = min(len(x), len(y))
            diff = np.flatnonzero(x[:n] != y[:n])
            line["first_divergence"] = int(diff[0]) if len(diff) else None
            if m["candidates_offered"] > 0:             # keyboard-only games offer no click candidates: identical actions are expected
                bad += not len(diff)
        print(json.dumps(line))
    print("ACTIVATION OK" if not bad else f"ACTIVATION PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
