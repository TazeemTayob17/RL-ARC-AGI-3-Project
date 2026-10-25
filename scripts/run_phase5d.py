"""Phase 5d grid on development games: the DQN without its pretrained prior (docs/PHASE5D_NOPRIOR_CRITERIA.md).

  V3np = V3 (baseline + I1 + I3) with a random frozen encoder instead of the pretrained prior
  V0np = baseline with a random frozen encoder instead of the pretrained prior

DQN only (PPO has no prior). Same 17 dev games x 3 seeds x 1,500 scorer-counted actions as Phases 3-5c. Matched references:
Phase 5b V3 records and Phase 3 baseline records (pretrained prior). One worker process per game.

Example: python scripts/run_phase5d.py --games ar25,ft09 --seeds 0 --workers 2 --execute
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
CONDITIONS = ("dqn_v3np", "dqn_v0np")
OUT = "results/phase5d_noprior"


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
                params = {"prior_mode": "random", "stage": {"v3np": "V3np", "v0np": "V0np"}[stage]}
                if stage == "v3np":
                    params.update({"object_clicks": True, "effect_memory": True, "intrinsic_beta": BETA})
                out.append(RunConfig(algo=algo, game=game, seed=seed, budget=budget, arch=dict(arch), threads=threads,
                                     version="V1" if algo == "dqn" else "V0", params=params, tag=f"phase5d-{stage}"))
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
        print(f"  {rec['status']} score={rec['official_score']['score'] if rec['official_score'] else None} "
              f"wall={rec['cost']['wall_s']}s", flush=True)
        if rec["status"] == "error":
            raise RuntimeError(rec["error"])


if __name__ == "__main__":
    main()
