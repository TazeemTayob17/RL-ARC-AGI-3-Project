"""Summarise the Phase 5d grid (DQN without the pretrained prior) with the criteria fixed before any run
(docs/PHASE5D_NOPRIOR_CRITERIA.md). Refuses incomplete grids and runs whose scorer count disagrees with the SDK total.

  python scripts/summarize_phase5d.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase5b import plan as plan5b
from scripts.run_phase5d import OUT, plan as plan5d
from scripts.summarize_phase4 import load, load_phase3_baseline, row
from scripts.summarize_phase5b import compare, extra

P5B = ROOT / "results/phase5b_improvement3"
P5D = ROOT / OUT
GAMES = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
SEEDS = [0, 1, 2]


def _rows(path, cfgs, loader=load):
    recs = {(c.game, c.seed): (loader(path, c) if loader is load else loader(c)) for c in cfgs}
    return recs, {gs: {**row(rec), **extra(rec)} for gs, rec in recs.items()}




def main() -> Path:
    recs, r = {}, {}
    for st in ("v3np", "v0np"):
        recs[st], r[st] = _rows(P5D, plan5d(GAMES, SEEDS, 1500, [f"dqn_{st}"]))
    _, r["v3"] = _rows(P5B, plan5b(GAMES, SEEDS, 1500, ["dqn_v3"]))
    _, r["ppo_v3"] = _rows(P5B, plan5b(GAMES, SEEDS, 1500, ["ppo_v3"]))
    from scripts.run_phase4 import plan as plan4
    _, r["v0"] = _rows(None, plan4("budget", GAMES, SEEDS, 1500, ["dqn"], 0.0), loader=load_phase3_baseline)
    not_random = sum(1 for st in recs for x in recs[st].values()
                     if not str(x["learner"].get("prior_checkpoint", "")).startswith("none"))
    out = [compare("V3np vs V3 (primary)", r["v3np"], r["v3"]), compare("V0np vs V0", r["v0np"], r["v0"]),
           compare("V3np vs V0np", r["v3np"], r["v0np"]), compare("V3np vs PPO V3", r["v3np"], r["ppo_v3"])]
    rows = [f"| {c['name']} | {c['b_mean']:.6f} | {c['a_mean']:.6f} | {c['paired_delta']:+.6f} | "
            f"[{c['bootstrap_95'][0]:+.6f}, {c['bootstrap_95'][1]:+.6f}] | {c['verdict']} | "
            f"{c['levels'][0]} -> {c['levels'][1]} | {c['progress_runs'][0]}/{c['progress_runs'][2]} -> "
            f"{c['progress_runs'][1]}/{c['progress_runs'][2]} | {c['median_first_progress'][0]} -> {c['median_first_progress'][1]} | "
            f"{c['frame_change_rate'][0]:.3f} -> {c['frame_change_rate'][1]:.3f} | "
            f"{len(c['regressed_games'])}: {', '.join(c['regressed_games']) or '-'} |" for c in out]
    lines = ["# Phase 5d results: DQN without the pretrained prior", "",
             f"Grid: {len(GAMES)} dev games x 3 seeds x 2 conditions = {len(GAMES) * 6} runs, 1500 scorer-counted actions, env seed 0. "
             "Random frozen encoder (fixed seed) instead of the pretrained prior; not tuned.",
             "Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.", "",
             "| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |",
             "|---|---:|---:|---:|---|---|---|---|---|---|---|", *rows, "",
             f"Activation: runs not using the random prior: {not_random}.", "",
             "## Limits", "",
             "Short development runs on exposed public games, one environment seed, three learner seeds. An interval crossing zero is inconclusive."]
    (P5D / "summary.json").write_text(json.dumps({"comparisons": out}, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE5D_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
