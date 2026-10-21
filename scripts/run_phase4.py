"""Phase 4 grids on development games.

  --mode i1      count-based novelty bonus (I1) for the selected DQN (code V1) and PPO (code V0); matched against the
                 Phase 3 records of the same games/seeds/budget (results/phase3_selected_baseline).
  --mode budget  budget-sensitivity diagnostic (NOT an improvement): baseline vs learning-off at a larger budget.

Configs match Phase 3 exactly except for the I1 parameters, so the Phase 3 records are the matched baseline.
One worker process per game (seeds sequential inside), as Phase 3 did, to avoid concurrent SDK cache loads.

Example: python scripts/run_phase4.py --mode i1 --games ar25,ft09 --seeds 0 --workers 2 --execute
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

BETA = 0.01                                  # predeclared before any run; not tuned
I1_CONDITIONS = ("dqn_i1", "ppo_i1")
BUDGET_CONDITIONS = ("dqn", "dqn_no_updates", "ppo", "ppo_no_updates")
BUDGET_GAMES = "ft09,g50t,ls20,ar25,sp80"    # predeclared 5-game subset (same games as the Phase 3 longer comparison)
BUDGET_DEFAULT = 5000
OUT = {"i1": "results/phase4_i1", "budget": "results/phase4_budget"}


def plan(mode: str, games: list[str], seeds: list[int], budget: int, conditions: list[str],
         beta: float = BETA, threads: int = 1) -> list[RunConfig]:
    split = json.loads((ROOT / "configs" / "splits.json").read_text())
    if not games or any(g not in split["dev"] for g in games):
        raise ValueError("Every selected game must be a development game in configs/splits.json")
    allowed = I1_CONDITIONS if mode == "i1" else BUDGET_CONDITIONS
    if not seeds or budget <= 0 or not conditions or any(c not in allowed for c in conditions):
        raise ValueError("Invalid seeds, budget or condition for this mode")
    arch = {"id": "onehotcnn_stack2", "channels": 32, "context_dim": 128}
    out = []
    for game in games:
        for seed in seeds:
            for c in conditions:
                algo = c.split("_")[0]
                params = {"intrinsic_beta": beta, "stage": "V1"} if mode == "i1" else {}
                out.append(RunConfig(algo=algo, game=game, seed=seed, budget=budget, arch=dict(arch),
                                     threads=threads, version="V1" if algo == "dqn" else "V0",
                                     updates_disabled=c.endswith("_no_updates"), params=params,
                                     tag="phase4-i1" if mode == "i1" else "phase4-budget"))
    return out


def _run_games_parallel(args, games):
    def one(game):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode, "--games", game,
               "--seeds", args.seeds, "--budget", str(args.budget), "--conditions", args.conditions,
               "--beta", str(args.beta), "--out", args.out, "--execute", "--workers", "1"]
        return game, subprocess.run(cmd, cwd=ROOT).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, games))
    bad = [g for g, rc in results if rc != 0]
    if bad:
        raise SystemExit(f"workers failed for: {bad}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("i1", "budget"), required=True)
    p.add_argument("--games", default=None, help="comma-separated dev games (default: all dev games / budget subset)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budget", type=int, default=None, help="default 1500 (i1) or 5000 (budget)")
    p.add_argument("--conditions", default=None)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    split = json.loads((ROOT / "configs" / "splits.json").read_text())
    a.games = a.games or (",".join(split["dev"]) if a.mode == "i1" else BUDGET_GAMES)
    a.budget = a.budget or (1500 if a.mode == "i1" else BUDGET_DEFAULT)
    a.conditions = a.conditions or ",".join(I1_CONDITIONS if a.mode == "i1" else BUDGET_CONDITIONS)
    a.out = a.out or OUT[a.mode]
    games = a.games.split(",")
    try:
        configs = plan(a.mode, games, [int(s) for s in a.seeds.split(",")], a.budget, a.conditions.split(","), a.beta)
    except ValueError as e:
        p.error(str(e))
    print(f"{len(configs)} runs; mode {a.mode}; budget {a.budget}; output {a.out}", flush=True)
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
        print(f"  {rec['status']} score={rec['official_score']['score'] if rec['official_score'] else None} "
              f"wall={rec['cost']['wall_s']}s", flush=True)
        if rec["status"] == "error":
            raise RuntimeError(rec["error"])


if __name__ == "__main__":
    main()
