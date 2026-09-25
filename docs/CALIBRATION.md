# Budget calibration (Phase 2, step 10) - PRELIMINARY, laptop, shared CPU

**Read this first.** These numbers were measured on the development laptop (8 logical cores, torch using 4 threads,
CPU only, 19.9 GB RAM) while **another project's DQN training job (`ResearchProject`, not ours) was running and every
run started at 100% system CPU** (`cost.system_cpu_pct_before_run` in each record). They are therefore
*pessimistic upper bounds* for this machine, and they say nothing about the cluster. The plan requires budgets to be
sized on the hardware actually used: **re-run `scripts/calibrate.py` on the cluster (idle node) before fixing B.**
The per-unit costs are what to reuse; totals are extrapolations by formula, not runs.

- Measured: 1500 actions per run, DQN warm-up shortened to 200 so updates start early, otherwise plan-default learner
  settings (DQN: batch 32, train every 4 actions; PPO: rollout 128, 4 epochs, 4 minibatches). Games: `ft09` (click),
  `g50t` (keyboard), `ar25` (mixed) at 32 channels; `ft09` only at 16 and 8 channels. Records: `results/calibration/`.
- Model: `time(B) = B * (t_env + t_act) + n_grad_steps(B) * t_grad`; `n_grad_steps` = (B - 1000)/4 for DQN,
  B/128 * 4 * 4 for PPO. `t_grad` here includes the replay/buffer bookkeeping, so it is a slight over-estimate.
- DQN replay at the plan's 50,000 capacity with all four frames of each transition stored: **783 MB** (measured);
  process peak RSS at 1500 actions was 490-570 MB. Indexed frame sharing would cut replay to about a quarter if RAM binds.
- Not measured: GPU (this machine has none), multi-process contention across runs, seeds beyond one per cell.

## What dominates
The CNN's four full-resolution 3x3 convolutions. One DQN update needs three forward passes and a backward pass over a
batch of 32 frames; at the default V0 width (32 channels) that is about 0.9 s here. Environment cost is small
(3-4 ms/action). Cheaper *exact* changes already applied: click head computed as `conv1x1(local) + linear(context)`
instead of a convolution over a broadcast tensor, and channels-last memory format (about 1.3x and 1.2x).
Width, DQN update cadence (`train_every`), batch size, and B are configuration levers; none has been changed from the
plan defaults except where noted.

| algo | channels | games measured | env ms/act | inference ms/act | s / gradient step | peak RSS MB |
|---|---|---|---|---|---|---|
| dqn | 8 | 1 | 3.6 | 7.6 | 0.37 | 499 |
| dqn | 16 | 1 | 4.2 | 9.7 | 0.52 | 492 |
| dqn | 32 | 3 | 3.1 | 11.8 | 0.90 | 566 |
| ppo | 8 | 1 | 4.2 | 11.2 | 0.20 | 499 |
| ppo | 16 | 1 | 4.1 | 13.0 | 0.30 | 499 |
| ppo | 32 | 3 | 3.8 | 15.8 | 0.55 | 566 |

Host: Windows-11-10.0.26200-SP0, torch threads: default

Hours for ONE run at budget B (plan-default learner settings, extrapolated):

| algo | channels | B=10,000 | B=25,000 | B=50,000 |
|---|---|---|---|---|
| dqn | 8 | 0.26 h | 0.69 h | 1.40 h |
| dqn | 16 | 0.36 h | 0.96 h | 1.95 h |
| dqn | 32 | 0.60 h | 1.61 h | 3.27 h |
| ppo | 8 | 0.11 h | 0.28 h | 0.56 h |
| ppo | 16 | 0.15 h | 0.38 h | 0.75 h |
| ppo | 32 | 0.25 h | 0.61 h | 1.23 h |

Inventory, plan version matrix (per algorithm: V0, V1, V2, A2 + updates-disabled V0 = 5 learner runs per game-seed; random policies negligible). Serial hours on this machine; divide by parallel workers:

| stage | games x seeds | B | DQN hours | PPO hours | total hours |
|---|---|---|---|---|---|
| development (Phases 3-5), channels=8 | 17x3 | 10,000 | 66 | 29 | 95 |
| development (Phases 3-5), channels=8 | 17x3 | 25,000 | 176 | 71 | 247 |
| development (Phases 3-5), channels=8 | 17x3 | 50,000 | 358 | 143 | 501 |
| reserved final (Phase 6), channels=8 | 8x3 | 10,000 | 31 | 13 | 45 |
| reserved final (Phase 6), channels=8 | 8x3 | 25,000 | 83 | 34 | 116 |
| reserved final (Phase 6), channels=8 | 8x3 | 50,000 | 169 | 67 | 236 |
| development (Phases 3-5), channels=16 | 17x3 | 10,000 | 92 | 38 | 131 |
| development (Phases 3-5), channels=16 | 17x3 | 25,000 | 245 | 96 | 341 |
| development (Phases 3-5), channels=16 | 17x3 | 50,000 | 498 | 192 | 690 |
| reserved final (Phase 6), channels=16 | 8x3 | 10,000 | 43 | 18 | 62 |
| reserved final (Phase 6), channels=16 | 8x3 | 25,000 | 115 | 45 | 160 |
| reserved final (Phase 6), channels=16 | 8x3 | 50,000 | 235 | 90 | 325 |
| development (Phases 3-5), channels=32 | 17x3 | 10,000 | 154 | 63 | 217 |
| development (Phases 3-5), channels=32 | 17x3 | 25,000 | 410 | 156 | 566 |
| development (Phases 3-5), channels=32 | 17x3 | 50,000 | 835 | 313 | 1148 |
| reserved final (Phase 6), channels=32 | 8x3 | 10,000 | 73 | 29 | 102 |
| reserved final (Phase 6), channels=32 | 8x3 | 25,000 | 193 | 74 | 266 |
| reserved final (Phase 6), channels=32 | 8x3 | 50,000 | 393 | 147 | 540 |

## Reading the inventory
- "5 learner runs per game-seed" = V0, V1, V2, A2 plus the updates-disabled V0 control, per algorithm. Random policies
  cost almost nothing. Tuning, scoring/submissions and contingency (plan's formula) are **not** included.
- Serial hours on one shared laptop; the real wall-clock is these divided by the number of runs the cluster can execute
  in parallel (runs are independent processes by design).
- If the total is over what the cluster allows, the plan's order of cuts applies: optional candidates, then seeds
  beyond three, then repeated runs - never the six required versions.
- Nothing here chooses B, the width, or the game count. Those decisions wait for cluster measurements and the
  architecture-selection result (V0's width is a provisional scaffold value).
