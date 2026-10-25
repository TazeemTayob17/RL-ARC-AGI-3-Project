"""Summarise the Phase 5b grid (Improvement 3 = object-level clicks with effect memory) with the criteria fixed before
any run (docs/PHASE5B_I3_CRITERIA.md).

  python scripts/summarize_phase5b.py

Comparisons (paired by game and seed; primary metric = paired mean official local score difference, game-cluster
bootstrap 95% interval; secondary = levels, progress (Level-1) runs, actions to first progress, frame-change rate):

  V3 vs V1    baseline + I1 + I3  vs  baseline + I1   (Phase 4 records)   does I3 add anything on top of I1?
  A3 vs V0    baseline + I3       vs  baseline        (Phase 3 records)   does I3 alone help?
  V3 vs A3                                                                 what does I1 add on top of I3?
  V3 vs V3nm  with vs without effect memory                               what does the memory add to object actions?
  V3nm vs V1  object actions alone vs pixel clicks                        what do object actions add by themselves?

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
from scripts.run_phase5b import CONDITIONS, OUT, plan as plan5b
from scripts.summarize_phase4 import load, load_phase3_baseline, paired, row, verdict

P4_I1 = ROOT / "results/phase4_i1"
P5B = ROOT / OUT
GAMES = json.loads((ROOT / "configs/splits.json").read_text())["dev"]
SEEDS = [0, 1, 2]


def extra(rec) -> dict:
    d = rec["diagnostics"]
    return {"first": d["first_reward_action"], "fcr": d["frame_change_rate"], "wall": rec["cost"]["wall_s"]}


def compare(name, a_rows, b_rows):
    deltas, mean_d, ci = paired(a_rows, b_rows, GAMES, SEEDS)
    n = len(a_rows)
    firsts = lambda rows: [r["first"] for r in rows.values() if r["first"] is not None]
    fa, fb = firsts(a_rows), firsts(b_rows)
    return {"name": name, "a_mean": statistics.mean(r["score"] for r in a_rows.values()),
            "b_mean": statistics.mean(r["score"] for r in b_rows.values()), "paired_delta": mean_d,
            "bootstrap_95": ci, "verdict": verdict(*ci),
            "levels": (sum(r["levels"] for r in b_rows.values()), sum(r["levels"] for r in a_rows.values())),
            "progress_runs": (sum(r["progress"] for r in b_rows.values()), sum(r["progress"] for r in a_rows.values()), n),
            "median_first_progress": (statistics.median(fb) if fb else None, statistics.median(fa) if fa else None),
            "frame_change_rate": (statistics.mean(r["fcr"] for r in b_rows.values()), statistics.mean(r["fcr"] for r in a_rows.values())),
            "wall_s": (statistics.mean(r["wall"] for r in b_rows.values()), statistics.mean(r["wall"] for r in a_rows.values())),
            "regressed_games": [g for g, d in deltas.items() if d < 0], "per_game_delta": deltas}


def main() -> Path:
    lines = ["# Phase 5b results: Improvement 3 (object-level clicks with effect memory)", "",
             f"Grid: {len(GAMES)} dev games x 3 seeds x {len(CONDITIONS)} conditions = {len(GAMES) * 3 * len(CONDITIONS)} runs, "
             "1500 scorer-counted actions, env seed 0. I3 constants fixed in src/objects.py, not tuned.",
             "Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.", ""]
    summary, act_lines, cmp_rows = {}, [], []
    for algo in ("dqn", "ppo"):
        def rows(path, cfgs, loader=load):
            out = {}
            for c in cfgs:
                rec = loader(path, c) if loader is load else loader(c)
                out[(c.game, c.seed)] = {**row(rec), **extra(rec)}
            return out
        recs = {}
        for st in ("v3", "a3", "v3nm"):
            for c in plan5b(GAMES, SEEDS, 1500, [f"{algo}_{st}"]):
                recs.setdefault(st, {})[(c.game, c.seed)] = load(P5B, c)
        r = {st: {gs: {**row(rec), **extra(rec)} for gs, rec in d.items()} for st, d in recs.items()}
        r["v1"] = rows(P4_I1, plan4("i1", GAMES, SEEDS, 1500, [f"{algo}_i1"]))
        r["v0"] = rows(None, plan4("budget", GAMES, SEEDS, 1500, [algo], 0.0), loader=load_phase3_baseline)
        out = [compare(f"{algo} V3 vs V1", r["v3"], r["v1"]), compare(f"{algo} A3 vs V0", r["a3"], r["v0"]),
               compare(f"{algo} V3 vs A3", r["v3"], r["a3"]), compare(f"{algo} V3 vs V3nm", r["v3"], r["v3nm"]),
               compare(f"{algo} V3nm vs V1", r["v3nm"], r["v1"])]
        for c in out:
            cmp_rows.append(f"| {c['name']} | {c['b_mean']:.6f} | {c['a_mean']:.6f} | {c['paired_delta']:+.6f} | "
                            f"[{c['bootstrap_95'][0]:+.6f}, {c['bootstrap_95'][1]:+.6f}] | {c['verdict']} | "
                            f"{c['levels'][0]} -> {c['levels'][1]} | {c['progress_runs'][0]}/{c['progress_runs'][2]} -> "
                            f"{c['progress_runs'][1]}/{c['progress_runs'][2]} | {c['median_first_progress'][0]} -> {c['median_first_progress'][1]} | "
                            f"{c['frame_change_rate'][0]:.3f} -> {c['frame_change_rate'][1]:.3f} | "
                            f"{len(c['regressed_games'])}: {', '.join(c['regressed_games']) or '-'} |")
        for st in ("v3", "a3", "v3nm"):
            m = [x["learner"]["effect_memory"] for x in recs[st].values()]
            offered = sum(x["candidates_offered"] for x in m)
            act_lines.append(f"- {algo} {st.upper()}: runs offering click candidates {sum(x['candidates_offered'] > 0 for x in m)}/{len(m)}; "
                             f"dropped as dead {sum(x['dropped_as_dead'] for x in m) / max(offered, 1):.1%} of {offered} offered; "
                             f"outcomes recorded {sum(x['recorded'] for x in m)}; retries {sum(x['retried'] for x in m)}.")
        summary[algo] = {"comparisons": out}
    lines += ["| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |",
              "|---|---:|---:|---:|---|---|---|---|---|---|---|", *cmp_rows, "",
              "## Mechanism activation", "", *act_lines, "",
              "A run that offered no click candidates (keyboard-only game) is not evidence about I3.", "",
              "## Limits", "",
              "Short development runs on exposed public games, one environment seed, three learner seeds. I3 constants "
              "(tile size, candidate cap, retry probability, dead-signature threshold) were fixed before any run. An interval "
              "crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence."]
    (P5B / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE5B_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
