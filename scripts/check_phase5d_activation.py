"""Phase 5d activation check: every no-prior run must use the random frozen encoder and its actions must diverge from the
matched pretrained run (Phase 5b V3 record for V3np, Phase 3 baseline record for V0np). Exit 1 on any problem."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase5b import plan as plan5b


def reference(rec, p5b, phase3):
    c = rec["identity"]["config"]
    if rec["identity"]["stage"] == "V3np":
        return Path(p5b) / plan5b([c["game"]], [c["seed"]], c["budget"], ["dqn_v3"])[0].run_id()
    for d in Path(phase3).glob(f"dqn-{c['version']}-{c['game']}-s{c['seed']}-*"):
        b = json.loads((d / "record.json").read_text())["identity"]["config"]
        if b["budget"] == c["budget"] and not b["updates_disabled"] and b["arch"] == c["arch"] and not b["params"].get("intrinsic_beta"):
            return d
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase5d_noprior")
    ap.add_argument("--phase5b", default="results/phase5b_improvement3")
    ap.add_argument("--phase3", default="results/phase3_selected_baseline")
    a = ap.parse_args(argv)
    bad = n = 0
    for d in sorted(Path(a.dir).iterdir()):
        if not (d / "record.json").exists():
            continue
        n += 1
        rec = json.loads((d / "record.json").read_text())
        problem = None
        if not str(rec["learner"].get("prior_checkpoint", "")).startswith("none"):
            problem = "random prior not in use"
        else:
            ref = reference(rec, a.phase5b, a.phase3)
            if ref is None or not (ref / "actions.npy").exists():
                problem = "reference not found"
            else:
                x, y = np.load(d / "actions.npy"), np.load(ref / "actions.npy")
                m = min(len(x), len(y))
                if not len(np.flatnonzero(x[:m] != y[:m])):
                    problem = "actions identical to the pretrained run"
        if problem:
            bad += 1
            print(json.dumps({"run": d.name, "problem": problem}))
    print(f"{n} runs checked; " + ("ACTIVATION OK" if not bad else f"ACTIVATION PROBLEMS: {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
