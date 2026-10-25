"""Diagnostic D1: was I1's novelty signal swamped by counters / status bars? (docs/further-improvements.md 3.4)

Replays the stored I1 action traces (results/phase4_i1, development games only) through the environment and recounts
how often a step lands in a "new" state, once with the exact-frame key I1 used and once with a key that ignores
volatile pixels. No learner is involved: the traces are replayed, so the comparison is exact.

Volatile pixel (learned online from frames only, no game knowledge): after 20 steps, a pixel that changed in at least
80% of the steps of EVERY action type tried 3+ times. A moving sprite fails the test (it does not change when its move
is blocked); a ticking counter passes it.

  python scripts/diagnostic_d1.py --games ar25,ft09 --out results/phase4_d1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import N_TYPES, action_type   # noqa: E402
from src.config import RunConfig                # noqa: E402
from src.run import make_session                # noqa: E402

MIN_STEPS, MIN_TYPE_STEPS, RATE = 20, 3, 0.8


class VolatileMask:
    def __init__(self):
        self.steps = np.zeros(N_TYPES, dtype=np.int64)
        self.changes = np.zeros((N_TYPES, 64, 64), dtype=np.int64)

    def update(self, a: int, before: np.ndarray, after: np.ndarray) -> None:
        t = action_type(a)
        self.steps[t] += 1
        self.changes[t] += before != after

    def mask(self) -> np.ndarray:
        if self.steps.sum() < MIN_STEPS:
            return np.zeros((64, 64), dtype=bool)
        used = np.flatnonzero(self.steps >= MIN_TYPE_STEPS)
        if used.size == 0:
            return np.zeros((64, 64), dtype=bool)
        rate = self.changes[used] / self.steps[used][:, None, None]
        return (rate >= RATE).all(axis=0)


def _bonus_stats(counts_hist: list[int]) -> dict:
    n = np.asarray(counts_hist, dtype=np.float64)
    b = 1.0 / np.sqrt(n)
    return {"new_state_rate": float((n == 1).mean()), "mean_bonus_factor": float(b.mean()), "std_bonus_factor": float(b.std())}


def replay(run_dir: Path) -> dict:
    rec = json.loads((run_dir / "record.json").read_text())
    cfg = RunConfig.from_dict(rec["identity"]["config"]).resolved()
    actions = np.load(run_dir / "actions.npy").astype(int).tolist()
    session = make_session(cfg)
    session.start_run(cfg.game, cfg.budget, cfg.env_seed)

    full: dict = {}
    online: dict = {}
    vol = VolatileMask()
    n_full, n_online = [], []
    prev_key_full = prev_key_online = None
    for a in actions:
        t = session.step(a)
        before, after = t.obs.frame, t.next_obs.frame
        vol.update(a, before, after)
        m = vol.mask()
        kf = (int(t.next_obs.progress), after.tobytes())
        ko = (int(t.next_obs.progress), np.where(m, 0, after).tobytes())
        full[kf] = full.get(kf, 0) + 1
        online[ko] = online.get(ko, 0) + 1
        n_full.append(full[kf]); n_online.append(online[ko])
    final = vol.mask()
    posthoc: dict = {}
    n_post = []
    session2 = make_session(cfg)
    session2.start_run(cfg.game, cfg.budget, cfg.env_seed)
    for a in actions:
        t = session2.step(a)
        k = (int(t.next_obs.progress), np.where(final, 0, t.next_obs.frame).tobytes())
        posthoc[k] = posthoc.get(k, 0) + 1
        n_post.append(posthoc[k])
    ys, xs = np.nonzero(final)
    faithful = (session.tracker.actions == rec["counters"]["scorer_actions"]
                and (session.scorecard() or {}).get("score") == (rec.get("official_score") or {}).get("score"))
    return {"run": run_dir.name, "algo": cfg.algo, "game": cfg.game, "seed": cfg.seed, "steps": len(actions),
            "replay_faithful": bool(faithful),
            "full": {"unique_states": len(full), **_bonus_stats(n_full)},
            "masked_online": {"unique_states": len(online), **_bonus_stats(n_online)},
            "masked_final": {"unique_states": len(posthoc), **_bonus_stats(n_post)},
            "mask_pixels": int(final.sum()),
            "mask_bbox": ([int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if final.any() else None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--src", default="results/phase4_i1")
    ap.add_argument("--out", default="results/phase4_d1")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    games = set(args.games.split(","))
    dev = set(json.loads((ROOT / "configs/splits.json").read_text())["dev"])
    if not games <= dev:
        raise SystemExit("D1 uses development games only")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted((ROOT / args.src).iterdir()):
        rec_path = d / "record.json"
        if not rec_path.exists():
            continue
        c = json.loads(rec_path.read_text())["identity"]["config"]
        if c["game"] not in games or (out / f"{d.name}.json").exists():
            continue
        r = replay(d)
        (out / f"{d.name}.json").write_text(json.dumps(r, indent=1))
        print(r["run"], r["replay_faithful"], r["full"]["unique_states"], r["masked_online"]["unique_states"],
              r["masked_final"]["unique_states"], r["mask_pixels"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
