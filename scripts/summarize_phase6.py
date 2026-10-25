"""Summarise the Phase 6 final evaluation on the reserved games with the protocol fixed in docs/PHASE6_FREEZE_RULE.md.

  python scripts/summarize_phase6.py

Refuses incomplete grids and runs whose scorer count disagrees with the SDK total. Validity gates (activation) are checked
before the tables are written; a failed gate raises and no result is written.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.summarize_phase5b as s5b
from scripts.run_phase6 import CONDITIONS, OUT, plan
from scripts.summarize_phase4 import load, row

P6 = ROOT / OUT
GAMES = json.loads((ROOT / "configs/splits.json").read_text())["reserved"]
SEEDS = [0, 1, 2, 3, 4]
s5b.GAMES, s5b.SEEDS = GAMES, SEEDS                  # compare() pairs over these games and seeds
COMPARISONS = {"dqn": [("v1", "v0"), ("v2", "v0"), ("v2", "v1"), ("v2", "a2"), ("v2np", "v2"), ("v2np", "v0")],
               "ppo": [("v1", "v0"), ("v2", "v0"), ("v2", "v1"), ("v2", "a2")]}


def gates(algo, st, recs) -> list:
    """Validity gates; returns a list of problems."""
    bad = []
    for gs, rec in recs.items():
        L = rec["learner"]
        if st in ("v1", "v2", "v2np") and not (L.get("intrinsic") or {}).get("nonzero", 0):
            bad.append((algo, st, gs, "no intrinsic bonus"))
        if st == "v2np" and not str(L.get("prior_checkpoint", "")).startswith("none"):
            bad.append((algo, st, gs, "random prior not in use"))
        if algo == "dqn" and st in ("v0", "v1", "a2", "v2") and str(L.get("prior_checkpoint", "")).startswith("none"):
            bad.append((algo, st, gs, "unexpected random prior"))
    return bad


def _line(c, cross=False):
    return (f"| {c['name']} | {c['b_mean']:.6f} | {c['a_mean']:.6f} | {c['paired_delta']:+.6f} | "
            f"[{c['bootstrap_95'][0]:+.6f}, {c['bootstrap_95'][1]:+.6f}] | {c['verdict']} | "
            f"{c['levels'][0]} -> {c['levels'][1]} | {c['progress_runs'][0]}/{c['progress_runs'][2]} -> "
            f"{c['progress_runs'][1]}/{c['progress_runs'][2]} | {c['median_first_progress'][0]} -> {c['median_first_progress'][1]} | "
            f"{len(c['regressed_games'])}: {', '.join(c['regressed_games']) or '-'} |")


def main() -> Path:
    rows, problems = {}, []
    for c in CONDITIONS:
        algo, st = c.split("_")
        recs = {(g.game, g.seed): load(P6, g) for g in plan(GAMES, SEEDS, 1500, [c])}
        rows[c] = {gs: {**row(rec), **s5b.extra(rec)} for gs, rec in recs.items()}
        problems += gates(algo, st, recs)
    if problems:
        raise RuntimeError(f"Validity gates failed, nothing written: {problems[:10]}")
    lines = ["# Phase 6 results: final evaluation on the reserved games", "",
             f"Grid: {len(GAMES)} reserved games x {len(SEEDS)} seeds x {len(CONDITIONS)} conditions = "
             f"{len(GAMES) * len(SEEDS) * len(CONDITIONS)} runs, 1500 scorer-counted actions, env seed 0. "
             "Protocol fixed in docs/PHASE6_FREEZE_RULE.md before the run.",
             "Scores are official local SDK scores, not Kaggle results; reserved means reserved from further tuning, not unseen (docs/EXPOSURE.md).", "",
             "## Mean official score per version", "",
             "| Condition | Mean score | Levels | Games with a level | Level-1 runs | Median first progress |", "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        r = rows[c]
        firsts = [x["first"] for x in r.values() if x["first"] is not None]
        glev = {g for (g, _), x in r.items() if x["levels"] > 0}
        lines.append(f"| {c} | {statistics.mean(x['score'] for x in r.values()):.6f} | {sum(x['levels'] for x in r.values())} | "
                     f"{len(glev)}/{len(GAMES)} | {sum(x['progress'] for x in r.values())}/{len(r)} | "
                     f"{statistics.median(firsts) if firsts else '-'} |")
    out, cmp_rows = {}, []
    for algo, pairs in COMPARISONS.items():
        for a, b in pairs:
            c = s5b.compare(f"{algo} {a.upper()} vs {b.upper()}", rows[f"{algo}_{a}"], rows[f"{algo}_{b}"])
            out.setdefault(algo, []).append(c)
            cmp_rows.append(_line(c))
    x = s5b.compare("DQN V2np vs PPO V2", rows["dqn_v2np"], rows["ppo_v2"])
    out["cross"] = [x]
    cmp_rows.append(_line(x))
    lines += ["", "## Paired comparisons (game-cluster bootstrap over 8 reserved games; wide intervals expected)", "",
              "| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress ref -> treat | Games regressed |",
              "|---|---:|---:|---:|---|---|---|---|---|---|", *cmp_rows, "",
              "## Mean score per game", "", "| Game | " + " | ".join(CONDITIONS) + " |", "|---|" + "---:|" * len(CONDITIONS)]
    for g in GAMES:
        lines.append(f"| {g} | " + " | ".join(f"{statistics.mean(rows[c][(g, s)]['score'] for s in SEEDS):.4f}" for c in CONDITIONS) + " |")
    lines += ["", "Validity gates passed: all records present, scorer count equals SDK total, activation checks (bonus, prior mode).", "",
              "## Limits", "",
              "Eight reserved public games, one environment seed, five learner seeds; all games were exposed to earlier development. "
              "An interval crossing zero is inconclusive. No Kaggle or unseen-game evidence."]
    (P6 / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    path = ROOT / "docs/PHASE6_RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
