"""Run or preview matched, provisional Phase 3 conditions on development games.

Example: python scripts/run_phase3.py --games sp80 --seeds 0 --budget 1500 --execute
The team must freeze the architecture and budget before treating results as final V0 evidence.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RunConfig

CONDITIONS = ("dqn", "ppo", "dqn_no_updates", "ppo_no_updates", "random_type", "random_flat")


def plan(games: list[str], seeds: list[int], budget: int, channels: int, conditions: list[str], dqn_version: str = "V0", threads: int = 0) -> list[RunConfig]:
    split = json.loads((ROOT / "configs" / "splits.json").read_text())
    if not games or any(game not in split["dev"] for game in games):
        raise ValueError("Every selected game must be a development game in configs/splits.json")
    if not seeds or any(seed < 0 for seed in seeds) or budget <= 0 or channels <= 0 or not conditions or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Invalid seeds, budget, channels, or condition")
    arch = {"id": "onehotcnn_stack2", "channels": channels, "context_dim": 128}
    return [RunConfig(algo=c.removesuffix("_no_updates"), game=game, seed=seed, budget=budget,
                      arch=arch, threads=threads, version=dqn_version if c.startswith("dqn") else "V0",
                      updates_disabled=c.endswith("_no_updates"),
                      tag="phase3-selected-dqn" if dqn_version == "V1" else "phase3-provisional")
            for game in games for seed in seeds for c in conditions]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", required=True, help="Comma-separated development game IDs")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--budget", required=True, type=int, help="Scorer-counted commands per run")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads; 0 uses the library default")
    p.add_argument("--dqn-version", choices=("V0", "V1"), default="V0",
                   help="V1 is the teammate-selected assignment baseline; V0 retains the earlier scaffold")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--out", default="results/phase3_provisional")
    p.add_argument("--execute", action="store_true", help="Run; without this flag only preview")
    a = p.parse_args()
    try:
        configs = plan(a.games.split(","), [int(s) for s in a.seeds.split(",")], a.budget,
                       a.channels, a.conditions.split(","), a.dqn_version, a.threads)
    except ValueError as e:
        p.error(str(e))
    print(f"{len(configs)} runs; DQN implementation {a.dqn_version}; output: {a.out}", flush=True)
    if not a.execute:
        for cfg in configs:
            print(cfg.run_id(), flush=True)
        return

    logging.disable(logging.INFO)  # SDK INFO includes a fresh anonymous API key.
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
        print(f"  {rec['status']} score={rec['official_score']} wall={rec['cost']['wall_s']}s", flush=True)
        if rec["status"] == "error":
            raise RuntimeError(rec["error"])


if __name__ == "__main__":
    main()
