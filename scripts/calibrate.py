"""Step 10: budget calibration (plan 6 'Budget calibration'). Measures end-to-end cost, not environment throughput.

  python scripts/calibrate.py --games ft09,g50t,ar25 --channels 32 --budget 1500 --out results/calibration
  python scripts/calibrate.py --inventory results/calibration      # build docs/CALIBRATION.md from saved measurements

Per (algo, game, channels) it records: seconds per action for env and inference, seconds per gradient step, peak RSS,
gradient-step / update counts. The inventory then extrapolates total time for any budget B:
    time(B) = B * (t_env + t_act) + n_grad_steps(B) * t_grad
with n_grad_steps(B) = (B - warmup)/train_every for DQN and B/rollout * epochs * minibatches for PPO.
Numbers are for the machine that ran them (recorded in each record); re-run on the cluster before fixing B.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ALGO_DEFAULTS, RunConfig
from src.run import run

ap = argparse.ArgumentParser()
ap.add_argument("--games", default="ft09,g50t,ar25")
ap.add_argument("--algos", default="dqn,ppo")
ap.add_argument("--channels", default="32")
ap.add_argument("--budget", type=int, default=1500)
ap.add_argument("--warmup", type=int, default=200, help="calibration-only warm-up so updates start early")
ap.add_argument("--out", default="results/calibration")
ap.add_argument("--inventory", default=None, help="dir of saved records -> print inventory tables")
a = ap.parse_args()


def measure():
    for ch in map(int, a.channels.split(",")):
        for game in a.games.split(","):
            for algo in a.algos.split(","):
                arch = {"id": "onehotcnn_stack2", "channels": ch, "context_dim": 128}
                params = {"warmup": a.warmup, "eps_decay_actions": a.budget // 2} if algo == "dqn" else {}
                rec = run(RunConfig(algo=algo, game=game, budget=a.budget, arch=arch, params=params, tag=f"calib-c{ch}"),
                          out_dir=a.out)
                c, lr = rec["cost"], rec["learner"]
                steps = max(rec["diagnostics"]["steps"], 1)
                print(f"{algo:<4} {game:<5} ch={ch:<3} wall={c['wall_s']:>7.1f}s env={c['env_s']/steps*1e3:5.1f}ms/act "
                      f"act={c['act_s']/steps*1e3:5.1f}ms/act grad_steps={lr['gradient_steps']:>4} "
                      f"peakRSS={c['peak_rss_mb']:.0f}MB  status={rec['status']}", flush=True)


def load(d):
    rows = []
    for f in glob.glob(f"{d}/*/record.json"):
        r = json.load(open(f))
        c, lr, cfg = r["cost"], r["learner"], r["identity"]["config"]
        steps = max(r["diagnostics"]["steps"], 1)
        g = max(lr["gradient_steps"], 1)
        rows.append(dict(algo=cfg["algo"], game=cfg["game"], ch=cfg["arch"]["channels"], t_env=c["env_s"] / steps,
                         t_act=c["act_s"] / steps, t_grad=c["observe_update_s"] / g if lr["gradient_steps"] else None,
                         peak=c["peak_rss_mb"], replay_mb=lr.get("replay_mb"), host=c["host"], threads=c["threads"]))
    return rows


def n_grad(algo, B):
    p = ALGO_DEFAULTS[algo]
    return max(0, (B - p["warmup"]) / p["train_every"]) if algo == "dqn" else B / p["rollout"] * p["epochs"] * p["minibatches"]


def inventory(d):
    rows = load(d)
    keys = sorted({(r["algo"], r["ch"]) for r in rows})
    print("| algo | channels | games measured | env ms/act | inference ms/act | s / gradient step | peak RSS MB |")
    print("|---|---|---|---|---|---|---|")
    unit = {}
    for al, ch in keys:
        rs = [r for r in rows if r["algo"] == al and r["ch"] == ch]
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        tg = sum(r["t_grad"] for r in rs if r["t_grad"]) / max(1, sum(1 for r in rs if r["t_grad"]))
        unit[(al, ch)] = (m("t_env") + m("t_act"), tg)
        print(f"| {al} | {ch} | {len(rs)} | {m('t_env')*1e3:.1f} | {m('t_act')*1e3:.1f} | {tg:.2f} | {max(r['peak'] for r in rs):.0f} |")
    print(f"\nHost: {rows[0]['host']}, torch threads: {rows[0]['threads']}\n")
    print("Hours for ONE run at budget B (plan-default learner settings, extrapolated):\n")
    Bs = (10_000, 25_000, 50_000)
    print("| algo | channels | " + " | ".join(f"B={b:,}" for b in Bs) + " |")
    print("|---|---|" + "---|" * len(Bs))
    for (al, ch), (t_act, tg) in unit.items():
        print(f"| {al} | {ch} | " + " | ".join(f"{(b * t_act + n_grad(al, b) * tg) / 3600:.2f} h" for b in Bs) + " |")
    print("\nInventory, plan version matrix (per algorithm: V0, V1, V2, A2 + updates-disabled V0 = 5 learner runs per"
          " game-seed; random policies negligible). Serial hours on this machine; divide by parallel workers:\n")
    print("| stage | games x seeds | B | DQN hours | PPO hours | total hours |")
    print("|---|---|---|---|---|---|")
    for ch in sorted({c for _, c in unit}):
        if ("dqn", ch) not in unit or ("ppo", ch) not in unit:
            continue
        for label, games, seeds in (("development (Phases 3-5)", 17, 3), ("reserved final (Phase 6)", 8, 3)):
            for b in Bs:
                h = {al: games * seeds * 5 * (b * unit[(al, ch)][0] + n_grad(al, b) * unit[(al, ch)][1]) / 3600
                     for al in ("dqn", "ppo")}
                print(f"| {label}, channels={ch} | {games}x{seeds} | {b:,} | {h['dqn']:.0f} | {h['ppo']:.0f} | {sum(h.values()):.0f} |")


if a.inventory:
    inventory(a.inventory)
else:
    measure()
