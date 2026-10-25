"""R6 check: re-run stored runs with the current code and require the exact same action sequence and score.

A speed-up is only accepted if every stored run reproduces. Usage:
    python scripts/check_trace_reproduction.py results/phase3_selected_baseline/dqn-V1-ft09-s0-39a56b343c65 ...
Exit status 1 if any run differs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RunConfig   # noqa: E402
from src.run import run            # noqa: E402


def check(run_dir: Path) -> dict:
    ref = json.loads((run_dir / "record.json").read_text())
    ref_actions = np.load(run_dir / "actions.npy")
    cfg = RunConfig.from_dict(ref["identity"]["config"])
    with tempfile.TemporaryDirectory() as tmp:
        rec = run(cfg, out_dir=tmp)
        new_actions = np.load(Path(tmp) / rec["identity"]["run_id"] / "actions.npy")
    same_actions = bool(np.array_equal(ref_actions, new_actions))
    ref_score = (ref.get("official_score") or {}).get("score")
    new_score = (rec.get("official_score") or {}).get("score")
    return {"run": run_dir.name, "n_actions": [len(ref_actions), len(new_actions)], "identical_actions": same_actions,
            "same_score": ref_score == new_score, "wall_s": [ref["cost"]["wall_s"], rec["cost"]["wall_s"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    logging.disable(logging.INFO)
    bad = 0
    for d in args.run_dirs:
        out = check(d)
        ok = out["identical_actions"] and out["same_score"]
        bad += not ok
        print(("IDENTICAL " if ok else "DIFFERENT ") + json.dumps(out), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
