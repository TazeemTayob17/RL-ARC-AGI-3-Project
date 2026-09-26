"""New matched five-game comparison; not a reconstruction of missing historical records."""
import json
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_phase3 import plan

GAMES = ["ft09", "g50t", "ls20", "ar25", "sp80"]
OUT = ROOT / "results/phase3_v0_v1_comparison"


def summarize():
    rows = []
    for version, directory in [("V0", OUT), ("V1", ROOT / "results/phase3_selected_baseline")]:
        for cfg in plan(GAMES, [0, 1, 2], 1500, 32, ["dqn"], version, 1):
            path = directory / cfg.run_id() / "record.json"
            if not path.exists():
                raise FileNotFoundError(f"Comparison incomplete: {path}")
            rec = json.loads(path.read_text())
            assert rec["identity"]["config_hash"] == cfg.hash(), path
            assert rec["status"] in ("budget", "won"), path
            count = rec["counters"]
            assert count["scorer_actions"] == count["sdk_total_actions"] <= 1500, path
            grads = rec["learner"].get("gradient_steps", 0)
            assert grads > 0, f"No learning updates: {path}"
            rows.append({"version": version, "game": cfg.game, "seed": cfg.seed,
                         "score": rec["official_score"]["score"],
                         "levels": rec["official_score"]["levels_completed"],
                         "gradient_steps": grads, "scorer_actions": count["scorer_actions"],
                         "record": str(path.relative_to(ROOT))})
    totals = {}
    for v in ("V0", "V1"):
        rs = [row for row in rows if row["version"] == v]
        totals[v] = {"mean_game_score": statistics.mean(x["score"] for x in rs),
                     "total_levels_across_15_runs": sum(x["levels"] for x in rs),
                     "mean_gradient_steps": statistics.mean(x["gradient_steps"] for x in rs)}
    lines = ["# New five-game DQN comparison", "",
             "This is an independent, reproducible comparison. The original five-game scores have now been confirmed from the teammate archive (see TEAMMATE_COMPARISON_VERIFICATION.md). This new protocol was declared before that archive arrived and differs in games, seeds and budget.", "",
             "Games: ft09, g50t, ls20, ar25, sp80. Seeds: 0, 1, 2. Environment seed: 0. Budget: 1500 scorer-counted commands. One CPU thread. V1 records are reused from the matched Phase 3 grid.", "",
             "| Implementation | Mean official local game score | Total levels across 15 runs | Mean gradient steps |",
             "|---|---:|---:|---:|"]
    for v, values in totals.items():
        lines.append(f"| {v} | {values['mean_game_score']:.6f} | {values['total_levels_across_15_runs']} | {values['mean_gradient_steps']:.1f} |")
    lines += ["", "## Per-game mean scores over three seeds", "",
              "| Game | Original V0 | Selected V1 | V1 minus V0 |", "|---|---:|---:|---:|"]
    for game in GAMES:
        means = {v: statistics.mean(x["score"] for x in rows if x["version"] == v and x["game"] == game) for v in ("V0", "V1")}
        lines.append(f"| {game} | {means['V0']:.6f} | {means['V1']:.6f} | {means['V1'] - means['V0']:+.6f} |")
    delta = totals["V1"]["mean_game_score"] - totals["V0"]["mean_game_score"]
    lines += ["", f"Observed mean score difference: {delta:+.6f}. Both systems made training updates in every run.", "",
              "This compares two complete systems: representation, pretraining, exploration, reward convention, value head and training schedules differ. It does not isolate the effect of one improvement. The original V0 warm-up remains 1000 actions; this budget gives it an opportunity to train, but does not equalize gradient counts or establish convergence.", "",
              "All scores are official local SDK scores, not Kaggle/private scores. These public games were used in development. Runtime rankings are omitted because concurrent machine load differs. The archived 0.0235/0.4904 scores belong to a different, 200-configured-action, seed-5209 protocol; they are not targets for this new protocol."]
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"protocol": {"games": GAMES, "seeds": [0,1,2], "budget": 1500}, "totals": totals, "runs": rows}, indent=2))
    path = ROOT / "docs/DQN_V0_V1_COMPARISON.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    summarize()
