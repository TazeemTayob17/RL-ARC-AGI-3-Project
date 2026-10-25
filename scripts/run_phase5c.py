"""Phase 5c grid on development games: Improvement 4 = frontier-directed exploration option
(docs/further-improvements.md, I4; src/frontier.py).

  V4   = V3 + I4        V3 is baseline + I1 + I3; the option is offered to the learner, which decides when to take it
  A4   = baseline + I4  no I1 and no I3 (fixed 4,096-pixel click space); isolates the option from the earlier stages
  V4f  = V3 + forced I4 the option is taken whenever it is available: a training-free graph-explorer reference policy

All run for the selected DQN (code V1) and PPO (code V0) on the same 17 dev games x 3 seeds x 1,500 scorer-counted
actions as Phases 3-5b. "Option never" is the Phase 5b V3 record (same games, seeds, budget). I4 constants are fixed in
src/frontier.py (MAX_PATH) and src/dqn_v1/learner.py (OPTION_EXPLORE_PROB) and are not tuned. One worker process per game.

Example: python scripts/run_phase5c.py --games ar25,ft09 --seeds 0 --workers 2 --execute
"""
import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RunConfig

BETA = 0.01                                   # I1, unchanged from Phases 4 and 5b
CONDITIONS = ("dqn_v4", "ppo_v4", "dqn_a4", "ppo_a4", "dqn_v4f", "ppo_v4f")
OUT = "results/phase5c_improvement4"


def plan(games: list[str], seeds: list[int], budget: int, conditions: list[str], threads: int = 1) -> list[RunConfig]:
    split = json.loads((ROOT / "configs" / "splits.json").read_text())
    if not games or any(g not in split["dev"] for g in games):
        raise ValueError("Every selected game must be a development game in configs/splits.json")
    if not seeds or budget <= 0 or not conditions or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Invalid seeds, budget or condition")
    arch = {"id": "onehotcnn_stack2", "channels": 32, "context_dim": 128}
    out = []
    for game in games:
        for seed in seeds:
            for c in conditions:
                algo, stage = c.split("_")
                params = {"frontier_mode": "forced" if stage == "v4f" else "learned", "stage": {"v4f": "V4f"}.get(stage, stage.upper())}
                if stage != "a4":                                     # V4 and V4f sit on top of V3 = I1 + I3
                    params.update({"object_clicks": True, "effect_memory": True, "intrinsic_beta": BETA})
                out.append(RunConfig(algo=algo, game=game, seed=seed, budget=budget, arch=dict(arch), threads=threads,
                                     version="V1" if algo == "dqn" else "V0", params=params, tag=f"phase5c-{stage}"))
    return out


def _run_games_parallel(args, games):
    def one(game):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--games", game, "--seeds", args.seeds,
               "--budget", str(args.budget), "--conditions", args.conditions, "--out", args.out, "--execute", "--workers", "1"]
        return game, subprocess.run(cmd, cwd=ROOT).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, games))
    bad = [g for g, rc in results if rc != 0]
    if bad:
        raise SystemExit(f"workers failed for: {bad}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default=None, help="comma-separated dev games (default: all dev games)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=OUT)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    split = json.loads((ROOT / "configs" / "splits.json").read_text())
    games = (a.games or ",".join(split["dev"])).split(",")
    try:
        configs = plan(games, [int(s) for s in a.seeds.split(",")], a.budget, a.conditions.split(","))
    except ValueError as e:
        p.error(str(e))
    print(f"{len(configs)} runs; budget {a.budget}; output {a.out}", flush=True)
    if not a.execute:
        for cfg in configs:
            print(cfg.run_id(), flush=True)
        return
    if a.workers > 1 and len(games) > 1:
        return _run_games_parallel(a, games)

    logging.disable(logging.INFO)            # SDK INFO includes a fresh anonymous API key
    from src.run import run
    for i, cfg in enumerate(configs, 1):
        record_path = Path(a.out) / cfg.run_id() / "record.json"
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old["identity"]["config_hash"] != cfg.hash() or old["status"] not in ("budget", "won"):
                raise RuntimeError(f"Existing run needs inspection: {record_path}")
            print(f"[{i}/{len(configs)}] exists {cfg.run_id()}", flush=True)
            continue
        print(f"[{i}/{len(configs)}] running {cfg.run_id()}", flush=True)
        rec = run(cfg, out_dir=a.out)
        f = rec["learner"].get("frontier") or {}
        print(f"  {rec['status']} score={rec['official_score']['score'] if rec['official_score'] else None} "
              f"wall={rec['cost']['wall_s']}s options={f.get('options_started')} steps_in_options={f.get('option_steps')}", flush=True)
        if rec["status"] == "error":
            raise RuntimeError(rec["error"])


if __name__ == "__main__":
    main()
