"""Summarize the agreed selected-baseline grid; refuse incomplete or invalid evidence."""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase3 import CONDITIONS, plan


def summarize(out=ROOT / "results/phase3_selected_baseline"):
    games = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
    configs = plan(games, [0, 1, 2], 1500, 32, list(CONDITIONS), "V1", 1)
    rows = []
    for cfg in configs:
        path = out / cfg.run_id() / "record.json"
        if not path.exists():
            raise RuntimeError(f"Incomplete grid: missing {path}")
        rec = json.loads(path.read_text())
        assert rec["identity"]["config_hash"] == cfg.hash(), path
        assert rec["status"] in ("budget", "won"), path
        ctr = rec["counters"]
        assert ctr["scorer_actions"] == ctr["sdk_total_actions"] <= cfg.budget, path
        assert rec["official_score"] is not None, path
        grads = rec["learner"].get("gradient_steps", 0)
        if cfg.updates_disabled:
            assert grads == 0, path
        elif cfg.algo in ("dqn", "ppo"):
            assert grads > 0, path
        rows.append({"condition": cfg.algo + ("_no_updates" if cfg.updates_disabled else ""),
                     "game": cfg.game, "seed": cfg.seed, "score": rec["official_score"]["score"],
                     "levels": rec["official_score"]["levels_completed"], "gradients": grads,
                     "actions": ctr["scorer_actions"], "wall_s": rec["cost"]["wall_s"],
                     **rec["diagnostics"]})
    means = {}
    lines = ["# Phase 3 selected baseline results", "",
             "Complete development grid: 17 games × 3 seeds × 6 conditions = 306 runs.",
             "Budget: 1500 scorer-counted commands; env seed 0; one CPU thread.",
             "Scores are official local SDK scores, not Kaggle results. All public games have prior development exposure.",
             "DQN implementation V1 is the adopted assignment baseline; PPO uses the existing provisional V0 architecture.", "",
             "| Condition | Mean game score | Total levels across runs | Progress runs | Mean gradients | Mean seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        rs = [r for r in rows if r["condition"] == condition]
        means[condition] = {"mean_score": statistics.mean(r["score"] for r in rs),
                            "levels": sum(r["levels"] for r in rs),
                            "progress_runs": sum(r["rewarded_transitions"] > 0 for r in rs),
                            "mean_gradients": statistics.mean(r["gradients"] for r in rs),
                            "mean_wall_s": statistics.mean(r["wall_s"] for r in rs)}
        m = means[condition]
        lines.append(f"| {condition} | {m['mean_score']:.6f} | {m['levels']} | {m['progress_runs']}/51 | {m['mean_gradients']:.1f} | {m['mean_wall_s']:.1f} |")
    lines += ["", "## Learning versus matching controls", ""]
    contrasts = {}
    for algo in ("dqn", "ppo"):
        deltas = []
        for game in games:
            a = statistics.mean(r["score"] for r in rows if r["condition"] == algo and r["game"] == game)
            b = statistics.mean(r["score"] for r in rows if r["condition"] == algo + "_no_updates" and r["game"] == game)
            deltas.append(a - b)
        rng = random.Random(2026)
        boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        contrasts[algo] = {"mean_score_delta": statistics.mean(deltas), "game_bootstrap_95pct": [boot[50], boot[1949]]}
        x = contrasts[algo]
        lines.append(f"- {algo}: mean score difference {x['mean_score_delta']:+.6f}; approximate game-cluster bootstrap 95% interval [{boot[50]:+.6f}, {boot[1949]:+.6f}].")
    lines += ["", "## Next improvement hypotheses", ""]
    for algo in ("dqn", "ppo"):
        rs = [r for r in rows if r["condition"] == algo]
        failures = sum(r["rewarded_transitions"] == 0 for r in rs)
        if failures > len(rs) / 2:
            hypothesis = ("test stagnation-triggered recovery/revisit control in addition to the existing structured explorer" if algo == "dqn"
                          else "test a matched salience-guided click-exploration treatment")
            lines.append(f"- {algo}: {failures}/51 runs found no progress. Candidate I1: {hypothesis}. Test against this frozen baseline; this is a hypothesis, not a demonstrated improvement.")
        else:
            lines.append(f"- {algo}: {failures}/51 runs found no progress. Inspect per-game progress curves and learning-control differences before selecting I1; aggregate scores alone do not identify the bottleneck.")
    lines += ["", "## Limits", "",
              "The teammate five-game figures are confirmed by archived scorecards (docs/TEAMMATE_COMPARISON_VERIFICATION.md); they are one-seed, short-budget whole-system evidence, not an isolated online-learning result. This bounded evaluation does not establish convergence or unseen-game generalization. The group accepted the baselines, split and budget and reported tutor approval; no matched RL tournament of architecture finalists is claimed. Competition deployment/testing remains later work. Existing DQN features cannot be credited again as later improvements."]
    (out / "summary.json").write_text(json.dumps({"means": means, "learning_contrasts": contrasts, "runs": rows}, indent=2))
    path = ROOT / "docs/PHASE3_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    summarize()
