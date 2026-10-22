"""Summarise the Phase 5 grid (Improvement 2 = event-terminated repeat option) with the criteria fixed in
docs/PROJECT_PLAN.md 7.2 / 7.3 before any run.

  python scripts/summarize_phase5.py

Comparisons (paired by game and seed; primary metric = paired mean official local score difference, game-cluster
bootstrap 95% interval; secondary = levels completed, progress runs, games regressed):

  V2 vs V1   baseline + I1 + A  vs  baseline + I1     (Phase 4 records)   does A add anything on top of I1?
  A2 vs V0   baseline + A       vs  baseline          (Phase 3 records)   does A alone help?
  V2 vs A2   baseline + I1 + A  vs  baseline + A                          what does I1 add on top of A?

An interval crossing zero is reported as inconclusive. Mechanism checks come first: if the option never fires, the run
is not evidence about A. Refuses incomplete grids and runs whose scorer count disagrees with the SDK total.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase4 import plan as plan4
from scripts.run_phase5 import CONDITIONS, OPTION_CAP, OPTION_PROB, OUT, plan as plan5
from scripts.summarize_phase4 import boot, load, load_phase3_baseline, paired, row, verdict

P4_I1 = ROOT / "results/phase4_i1"
P5 = ROOT / OUT
GAMES = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
SEEDS = [0, 1, 2]


def option_stats(recs: list) -> dict:
    o = [r["learner"]["option"] for r in recs]
    n = len(o)
    reasons = {k: sum(x["stop_reasons"][k] for x in o) for k in o[0]["stop_reasons"]}
    return {"runs": n, "runs_with_option": sum(x["options_longer_than_one"] > 0 for x in o),
            "mean_options_per_run": statistics.mean(x["options_longer_than_one"] for x in o),
            "mean_option_length": statistics.mean(x["mean_option_length"] for x in o),
            "mean_decisions": statistics.mean(x["decisions"] for x in o),
            "mean_repeated_steps": statistics.mean(x["repeated_steps"] for x in o), "stop_reasons": reasons}


def compare(name, a_rows, b_rows):
    deltas, mean_d, ci = paired(a_rows, b_rows, GAMES, SEEDS)
    n = len(a_rows)
    reg = [g for g, d in deltas.items() if d < 0]
    return {"name": name, "a_mean": statistics.mean(r["score"] for r in a_rows.values()),
            "b_mean": statistics.mean(r["score"] for r in b_rows.values()), "paired_delta": mean_d,
            "bootstrap_95": ci, "verdict": verdict(*ci),
            "levels": (sum(r["levels"] for r in b_rows.values()), sum(r["levels"] for r in a_rows.values())),
            "progress_runs": (sum(r["progress"] for r in b_rows.values()), sum(r["progress"] for r in a_rows.values()), n),
            "regressed_games": reg, "per_game_delta": deltas}


def main() -> Path:
    lines = ["# Phase 5 results: Improvement 2 (event-terminated repeat option)", "",
             f"Grid: {len(GAMES)} dev games x 3 seeds x {len(CONDITIONS)} conditions = {len(GAMES) * 3 * len(CONDITIONS)} runs, "
             f"1500 scorer-counted actions, env seed 0. Option cap {OPTION_CAP}, engagement probability {OPTION_PROB} (predeclared, not tuned).",
             "Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.", ""]
    summary, act_lines, cmp_rows = {}, [], []
    for algo in ("dqn", "ppo"):
        v2 = {(c.game, c.seed): load(P5, c) for c in plan5(GAMES, SEEDS, 1500, [f"{algo}_v2"])}
        a2 = {(c.game, c.seed): load(P5, c) for c in plan5(GAMES, SEEDS, 1500, [f"{algo}_a2"])}
        v1 = {(c.game, c.seed): load(P4_I1, c) for c in plan4("i1", GAMES, SEEDS, 1500, [f"{algo}_i1"])}
        v0 = {(c.game, c.seed): load_phase3_baseline(c) for c in plan4("budget", GAMES, SEEDS, 1500, [algo], 0.0)}
        r = {k: {gs: row(rec) for gs, rec in d.items()} for k, d in (("v2", v2), ("a2", a2), ("v1", v1), ("v0", v0))}
        out = [compare(f"{algo} V2 vs V1", r["v2"], r["v1"]), compare(f"{algo} A2 vs V0", r["a2"], r["v0"]),
               compare(f"{algo} V2 vs A2", r["v2"], r["a2"])]
        for c in out:
            cmp_rows.append(f"| {c['name']} | {c['b_mean']:.6f} | {c['a_mean']:.6f} | {c['paired_delta']:+.6f} | "
                            f"[{c['bootstrap_95'][0]:+.6f}, {c['bootstrap_95'][1]:+.6f}] | {c['verdict']} | "
                            f"{c['levels'][0]} -> {c['levels'][1]} | {c['progress_runs'][0]}/{c['progress_runs'][2]} -> "
                            f"{c['progress_runs'][1]}/{c['progress_runs'][2]} | {len(c['regressed_games'])}: {', '.join(c['regressed_games']) or '-'} |")
        st = {"v2": option_stats(list(v2.values())), "a2": option_stats(list(a2.values()))}
        for k, s in st.items():
            act_lines.append(f"- {algo} {k.upper()}: option fired in {s['runs_with_option']}/{s['runs']} runs; mean {s['mean_options_per_run']:.0f} "
                             f"options longer than one per run; mean steps per decision {s['mean_option_length']:.2f}; "
                             f"mean decisions per run {s['mean_decisions']:.0f}; stop reasons {s['stop_reasons']}.")
        summary[algo] = {"comparisons": out, "option_stats": st}
    lines += ["| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Games regressed |",
              "|---|---:|---:|---:|---|---|---|---|---|", *cmp_rows, "",
              "## Mechanism activation", "", *act_lines, "",
              "A run whose option never fires is not evidence about A (failure rule in the 7.3 card).", "",
              "## Limits", "",
              "Short development runs on exposed public games, one environment seed, three learner seeds. The option is a "
              "behaviour-level mechanism with fixed cap and engagement probability, not a learned choice; any benefit is a "
              "benefit of that rule. Part of the evidence that motivated A came from the reserved game lp85 (disclosed in "
              "docs/EXPOSURE.md). An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence."]
    (P5 / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE5_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
