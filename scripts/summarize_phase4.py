"""Summarise Phase 4 grids with the criteria fixed in docs/PROJECT_PLAN.md 7.2 before any run.

  python scripts/summarize_phase4.py i1       I1 vs the Phase 3 baseline (paired by game and seed)
  python scripts/summarize_phase4.py budget   budget-sensitivity diagnostic (baseline vs learning-off, larger budget)

Primary metric: paired mean official local score difference, game-cluster bootstrap 95% interval.
Secondary: levels completed, progress runs, regressions. An interval crossing zero is "inconclusive".
Refuses incomplete grids and runs whose scorer count disagrees with the SDK total.
"""
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase4 import BUDGET_GAMES, I1_CONDITIONS, plan

PHASE3 = ROOT / "results/phase3_selected_baseline"


def load(path: Path, cfg) -> dict:
    rec = json.loads((path / cfg.run_id() / "record.json").read_text()) if (path / cfg.run_id() / "record.json").exists() else None
    if rec is None:
        raise RuntimeError(f"Incomplete grid: missing {path / cfg.run_id()}")
    assert rec["identity"]["config_hash"] == cfg.hash(), cfg.run_id()
    assert rec["status"] in ("budget", "won"), (cfg.run_id(), rec["status"])
    ctr = rec["counters"]
    assert ctr["scorer_actions"] == ctr["sdk_total_actions"] <= cfg.budget, cfg.run_id()
    assert rec["official_score"] is not None, cfg.run_id()
    return rec


def row(rec) -> dict:
    return {"score": rec["official_score"]["score"], "levels": rec["official_score"]["levels_completed"],
            "progress": rec["diagnostics"]["rewarded_transitions"] > 0,
            "gradients": rec["learner"].get("gradient_steps", 0), "wall_s": rec["cost"]["wall_s"],
            "intrinsic": rec["learner"].get("intrinsic")}


def boot(deltas, k=2000):
    rng = random.Random(2026)
    s = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(k))
    return [s[int(0.025 * k)], s[int(0.975 * k) - 1]]


def verdict(lo, hi) -> str:
    return "above zero" if lo > 0 else "below zero" if hi < 0 else "crosses zero (inconclusive)"


def paired(rows_a, rows_b, games, seeds):
    deltas = {g: statistics.mean(rows_a[(g, s)]["score"] - rows_b[(g, s)]["score"] for s in seeds) for g in games}
    ci = boot(list(deltas.values()))
    return deltas, statistics.mean(deltas.values()), ci


def summarize_i1() -> Path:
    games = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
    seeds = [0, 1, 2]
    out = ROOT / "results/phase4_i1"
    lines = ["# Phase 4 results: Improvement 1 (count-based novelty bonus)", "",
             f"Grid: {len(games)} dev games × 3 seeds × 2 algorithms = {len(games) * 6} runs, 1500 scorer-counted actions, env seed 0.",
             "Treatment: bonus beta/sqrt(n) on exact observed grids (beta = 0.01, predeclared, not tuned), zero on action/undo loops.",
             "Matched baseline: Phase 3 records of the same games and seeds. Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.", "",
             "| Algorithm | Baseline mean score | I1 mean score | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels base → I1 | Progress runs base → I1 | Games regressed |",
             "|---|---:|---:|---:|---|---|---|---|---|"]
    summary = {}
    for algo in ("dqn", "ppo"):
        base_cfgs = plan("budget", games, seeds, 1500, [algo], 0.0)       # identical to Phase 3 baseline configs, tag aside
        i1_cfgs = plan("i1", games, seeds, 1500, [f"{algo}_i1"])
        rows_b, rows_i = {}, {}
        for cb, ci_ in zip(base_cfgs, i1_cfgs):
            rows_i[(ci_.game, ci_.seed)] = row(load(out, ci_))
            rows_b[(cb.game, cb.seed)] = row(load_phase3_baseline(cb))
        deltas, mean_d, ci = paired(rows_i, rows_b, games, seeds)
        mb = statistics.mean(r["score"] for r in rows_b.values())
        mi = statistics.mean(r["score"] for r in rows_i.values())
        lv = (sum(r["levels"] for r in rows_b.values()), sum(r["levels"] for r in rows_i.values()))
        pr = (sum(r["progress"] for r in rows_b.values()), sum(r["progress"] for r in rows_i.values()))
        reg = [g for g, d in deltas.items() if d < 0]
        n = len(rows_i)
        lines.append(f"| {algo} | {mb:.6f} | {mi:.6f} | {mean_d:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {verdict(*ci)} | "
                     f"{lv[0]} → {lv[1]} | {pr[0]}/{n} → {pr[1]}/{n} | {len(reg)}: {', '.join(reg) or '-'} |")
        summary[algo] = {"baseline_mean": mb, "i1_mean": mi, "paired_delta": mean_d, "bootstrap_95": ci,
                         "verdict": verdict(*ci), "levels": lv, "progress_runs": pr, "regressed_games": reg,
                         "per_game_delta": deltas,
                         "bonus_nonzero_mean": statistics.mean(r["intrinsic"]["nonzero"] for r in rows_i.values()),
                         "bonus_total_mean": statistics.mean(r["intrinsic"]["total_bonus"] for r in rows_i.values()),
                         "level1_reach_fraction": pr[1] / n}
    frac = {a: summary[a]["level1_reach_fraction"] for a in summary}
    lines += ["", "## I2 selection rule (fixed before Phase 4; docs/PROJECT_PLAN.md 7.2)", "",
              f"- Fraction of I1 runs completing Level 1: DQN {frac['dqn']:.3f}, PPO {frac['ppo']:.3f} (threshold 1/3).",
              "- Rule 1: " + ("fewer than one third reach Level 1 for both algorithms → choose **A** (event-terminated repeat option)."
                             if max(frac.values()) < 1 / 3 else
                             "at least one algorithm reaches Level 1 in a third or more of its runs → rule 1 does not fire; "
                             "rule 2 (effective actions rediscovered after level changes) needs a log inspection."),
              "", "## Mechanism activation", ""]
    for algo in ("dqn", "ppo"):
        s = summary[algo]
        lines.append(f"- {algo}: mean nonzero-bonus steps per run {s['bonus_nonzero_mean']:.0f}; mean total bonus {s['bonus_total_mean']:.2f}.")
    lines += ["", "## Limits", "",
              "Short development runs on exposed public games, one environment seed, three learner seeds. An interval crossing zero is reported as inconclusive. "
              "No Kaggle or unseen-game evidence."]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE4_RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def load_phase3_baseline(cfg) -> dict:
    """Phase 3 record for the same algo/game/seed/budget/arch with no I1 parameters (tag differs, so match by content)."""
    for d in PHASE3.glob(f"{cfg.algo}-{cfg.version}-{cfg.game}-s{cfg.seed}-*"):
        rec = json.loads((d / "record.json").read_text())
        c = rec["identity"]["config"]
        if c["budget"] == cfg.budget and not c["updates_disabled"] and c["arch"] == cfg.arch and not c["params"].get("intrinsic_beta"):
            assert rec["status"] in ("budget", "won"), d
            return rec
    raise RuntimeError(f"No Phase 3 baseline for {cfg.run_id()}")


def summarize_budget(budget: int = 5000) -> Path:
    games = BUDGET_GAMES.split(",")
    seeds = [0, 1, 2]
    out = ROOT / "results/phase4_budget"
    lines = ["# Phase 4 budget-sensitivity diagnostic (not an improvement)", "",
             f"Baseline vs learning-off at {budget} scorer-counted actions on {len(games)} predeclared dev games × 3 seeds × both algorithms.",
             "Question: does learning start to matter at a larger budget? Scores are official local SDK scores; public games are exposed.", "",
             "| Algorithm | Mean score (learning) | Mean score (off) | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels learn / off | Progress runs learn / off | Mean gradients |",
             "|---|---:|---:|---:|---|---|---|---|---:|"]
    summary = {}
    for algo in ("dqn", "ppo"):
        on_cfgs = plan("budget", games, seeds, budget, [algo])
        off_cfgs = plan("budget", games, seeds, budget, [f"{algo}_no_updates"])
        rows_on = {(c.game, c.seed): row(load(out, c)) for c in on_cfgs}
        rows_off = {(c.game, c.seed): row(load(out, c)) for c in off_cfgs}
        deltas, mean_d, ci = paired(rows_on, rows_off, games, seeds)
        n = len(rows_on)
        lines.append(f"| {algo} | {statistics.mean(r['score'] for r in rows_on.values()):.6f} | "
                     f"{statistics.mean(r['score'] for r in rows_off.values()):.6f} | {mean_d:+.6f} | "
                     f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {verdict(*ci)} | "
                     f"{sum(r['levels'] for r in rows_on.values())} / {sum(r['levels'] for r in rows_off.values())} | "
                     f"{sum(r['progress'] for r in rows_on.values())}/{n} / {sum(r['progress'] for r in rows_off.values())}/{n} | "
                     f"{statistics.mean(r['gradients'] for r in rows_on.values()):.0f} |")
        summary[algo] = {"paired_delta": mean_d, "bootstrap_95": ci, "per_game_delta": deltas}
    lines += ["", "If learning helps here, later matched comparisons are rerun at this budget; otherwise the 1500-action protocol stands."]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE4_BUDGET_DIAGNOSTIC.md"
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "i1"
    print(summarize_i1() if which == "i1" else summarize_budget())
