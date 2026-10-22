"""Option activation check (plan 6, bug procedure step 2): did the repeat option fire, and did it change behaviour?

For every run in --dir report option counts, mean steps per decision, stop reasons and the first action that differs
from the matching reference run (same algo/game/seed; --reference-dir). Exit 1 if an option never fired, every run
reports zero repeated steps, or a run's actions never diverge from its reference.
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
                and bool(b["params"].get("intrinsic_beta")) == with_i1 and not b["params"].get("option_cap")):
            return d
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase5_improvement2")
    ap.add_argument("--phase3", default="results/phase3_selected_baseline")
    ap.add_argument("--phase4", default="results/phase4_i1")
    a = ap.parse_args(argv)
    bad = 0
    for d in sorted(Path(a.dir).iterdir()):
        p = d / "record.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        o = rec["learner"].get("option")
        if o is None:
            print(json.dumps({"run": d.name, "problem": "no option diagnostics"}))
            bad += 1
            continue
        with_i1 = bool(rec["identity"]["config"]["params"].get("intrinsic_beta"))
        ref = reference(rec, a.phase4 if with_i1 else a.phase3, with_i1)
        line = {"run": d.name, "decisions": o["decisions"], "steps": o["primitive_steps"],
                "options_gt1": o["options_longer_than_one"], "mean_len": round(o["mean_option_length"], 3),
                "stops": o["stop_reasons"]}
        if ref is None:
            line["reference"] = "NOT FOUND"
            bad += 1
        else:
            x, y = np.load(d / "actions.npy"), np.load(ref / "actions.npy")
            n = min(len(x), len(y))
            diff = np.flatnonzero(x[:n] != y[:n])
            line["first_divergence"] = int(diff[0]) if len(diff) else None
            bad += not len(diff)
        if o["options_longer_than_one"] == 0:
            bad += 1
        print(json.dumps(line))
    print("ACTIVATION OK" if not bad else f"ACTIVATION PROBLEMS: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
