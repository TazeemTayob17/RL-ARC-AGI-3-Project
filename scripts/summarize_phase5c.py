"""Summarise the Phase 5c grid (Improvement 4 = frontier-directed exploration option) with the criteria fixed before any
run (docs/PHASE5C_I4_CRITERIA.md).

  python scripts/summarize_phase5c.py

Comparisons (paired by game and seed; primary metric = paired mean official local score difference, game-cluster
bootstrap 95% interval; secondary = levels, progress (Level-1) runs, actions to first progress, frame-change rate):

  V4 vs V3     baseline + I1 + I3 + I4  vs  baseline + I1 + I3   (Phase 5b records)   does the option add anything?
  A4 vs V0     baseline + I4            vs  baseline              (Phase 3 records)    does the option alone help?
  V4 vs A4                                                                              what do I1 + I3 add on top of I4?
  V4 vs V4f    learned vs always taking the option                                      does learning when to use it matter?
  V4f vs V3    training-free graph explorer (with fallback) vs the V3 learner
  V4 vs V1     cumulative I3 + I4 over the I1 stage                                      (Phase 4 records)

An interval crossing zero is reported as inconclusive. Refuses incomplete grids and runs whose scorer count disagrees
with the SDK total.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase4 import plan as plan4
from scripts.run_phase5b import plan as plan5b
from scripts.run_phase5c import CONDITIONS, OUT, plan as plan5c
from scripts.summarize_phase4 import load, load_phase3_baseline, row
from scripts.summarize_phase5b import compare, extra

P4_I1 = ROOT / "results/phase4_i1"
P5B = ROOT / "results/phase5b_improvement3"
P5C = ROOT / OUT
GAMES = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
SEEDS = [0, 1, 2]


def _rows(path, cfgs, loader=load):
    recs = {(c.game, c.seed): (loader(path, c) if loader is load else loader(c)) for c in cfgs}
    return recs, {gs: {**row(rec), **extra(rec)} for gs, rec in recs.items()}


def main() -> Path:
    lines = ["# Phase 5c results: Improvement 4 (frontier-directed exploration option)", "",
             f"Grid: {len(GAMES)} dev games x 3 seeds x {len(CONDITIONS)} conditions = {len(GAMES) * 3 * len(CONDITIONS)} runs, "
             "1500 scorer-counted actions, env seed 0. I4 constants fixed in src/frontier.py and src/dqn_v1/learner.py, not tuned.",
             "Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.", ""]
    summary, act_lines, cmp_rows = {}, [], []
    for algo in ("dqn", "ppo"):
        recs, r = {}, {}
        for st in ("v4", "a4", "v4f"):
            recs[st], r[st] = _rows(P5C, plan5c(GAMES, SEEDS, 1500, [f"{algo}_{st}"]))
        _, r["v3"] = _rows(P5B, plan5b(GAMES, SEEDS, 1500, [f"{algo}_v3"]))
        _, r["v1"] = _rows(P4_I1, plan4("i1", GAMES, SEEDS, 1500, [f"{algo}_i1"]))
        _, r["v0"] = _rows(None, plan4("budget", GAMES, SEEDS, 1500, [algo], 0.0), loader=load_phase3_baseline)
        out = [compare(f"{algo} V4 vs V3", r["v4"], r["v3"]), compare(f"{algo} A4 vs V0", r["a4"], r["v0"]),
               compare(f"{algo} V4 vs A4", r["v4"], r["a4"]), compare(f"{algo} V4 vs V4f", r["v4"], r["v4f"]),
               compare(f"{algo} V4f vs V3", r["v4f"], r["v3"]), compare(f"{algo} V4 vs V1", r["v4"], r["v1"])]
        for c in out:
            cmp_rows.append(f"| {c['name']} | {c['b_mean']:.6f} | {c['a_mean']:.6f} | {c['paired_delta']:+.6f} | "
                            f"[{c['bootstrap_95'][0]:+.6f}, {c['bootstrap_95'][1]:+.6f}] | {c['verdict']} | "
                            f"{c['levels'][0]} -> {c['levels'][1]} | {c['progress_runs'][0]}/{c['progress_runs'][2]} -> "
                            f"{c['progress_runs'][1]}/{c['progress_runs'][2]} | {c['median_first_progress'][0]} -> {c['median_first_progress'][1]} | "
                            f"{c['frame_change_rate'][0]:.3f} -> {c['frame_change_rate'][1]:.3f} | "
                            f"{len(c['regressed_games'])}: {', '.join(c['regressed_games']) or '-'} |")
        for st in ("v4", "a4", "v4f"):
            fs = [x["learner"]["frontier"] for x in recs[st].values()]
            offered = sum(f["decisions"] for f in fs)
            act_lines.append(
                f"- {algo} {st.upper()}: option offered at {sum(f['decisions_with_option'] for f in fs) / max(offered, 1):.1%} of {offered} decisions; "
                f"chosen {sum(f['options_started'] for f in fs)} times (runs with at least one: {sum(f['options_started'] > 0 for f in fs)}/{len(fs)}); "
                f"{sum(f['option_steps'] for f in fs)} of {sum(f['primitive_steps'] for f in fs)} steps inside options; "
                f"mean option length {statistics.mean(f['mean_option_length'] for f in fs):.2f}; "
                f"walks reaching the expected node {sum(f['walks_completed_as_expected'] for f in fs)}/{sum(f['walks_started'] for f in fs)}; "
                f"new states per 100 steps inside / outside options {statistics.mean(f['new_states_per_100_option_steps'] for f in fs):.1f} / "
                f"{statistics.mean(f['new_states_per_100_other_steps'] for f in fs):.1f}; "
                f"deviations {sum(f['stop_reasons']['deviation'] for f in fs)}.")
        summary[algo] = {"comparisons": out}
    lines += ["| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |",
              "|---|---:|---:|---:|---|---|---|---|---|---|---|", *cmp_rows, "",
              "## Mechanism activation", "", *act_lines, "",
              "## Limits", "",
              "Short development runs on exposed public games, one environment seed, three learner seeds. I4 constants "
              "(maximum walk length, exploration share of the option for DQN, uniform choice of the untried action) were fixed "
              "before any run. An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence."]
    (P5C / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE5C_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
