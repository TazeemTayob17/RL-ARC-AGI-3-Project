# Exposure table (PROJECT_PLAN 3.3)

All 25 public games were already used in architecture development (plan 3.3), so **no public game is untouched**.
"Reserved" below means reserved from *further* tuning only. Split: `configs/splits.json`, hash `8819bd3b28d9`,
status **confirmed by the group**. Rule: stratified by action-set type, seeded,
and a game whose completion trajectory was inspected (ft09) is never reserved.

## Cluster-research exposure (added in Phase 4)

Sources: `BASELINE_EXPERIMENT_HISTORY.txt` (verified in the repo) and the team research handoff (cluster experiments E001–E074; reported, not re-run). None of these artifacts is used by our agents except the frozen DQN prior checkpoints.

| Exposure | Games affected | Effect on the reserved games (bp35, cn04, lp85, s5i5, sb26, sk48, tu93, wa30) |
|---|---|---|
| Selected-DQN architecture, exploration and ablation development | all 25 | All reserved games influenced architecture and exploration choices |
| Frozen DQN prior checkpoints: each fold checkpoint trained on the other four public folds | all 25 (each game is held out of one checkpoint, in the other four) | Every reserved game is in the training data of four of the five checkpoints; the deployment checkpoint (fold 0) was trained on folds 1–4, which include cn04, lp85, s5i5, wa30 |
| E074 human-demonstration models (BC/DDQN, offline) trained on `sb26, s5i5, lf52, tn36, g50t` | 5 | sb26 and s5i5 (reserved) were used for training; **not used by any of our agents** |
| Wider-project reported results and design iteration (vc33, cd82, sc25, lp85, wa30 and others) | many | lp85 and wa30 (reserved) appear in reported controller results; they informed the choice of improvement candidates (event-repeat evidence includes lp85) |
| Anti-cheat rule | all | No agent reads game files or human data; guard test `tests/test_no_cheating.py` |

"Reserved" therefore still means only reserved from *further tuning*; no public game is untouched.

Columns filled from this repo's recon and Phase 1 probes (`scripts/probe/`, `docs/evidence/`). The architecture-development and historical training exposure are now supported by
`BASELINE_EXPERIMENT_HISTORY.txt`: all 25 games influenced architecture, exploration, and DQN development.
Exact cluster exposure dates and raw experiment records are still pending. The dates in the table describe
the earlier repository probes only, not the cluster experiments. The matched Phase 3 baseline grid completed on all 17 development games; its three-seed training/control runs are recorded in results/phase3_selected_baseline.

| game | split | metadata inspected | trajectories inspected | random / probe runs | architecture development | training/tuning | dates |
|---|---|---|---|---|---|---|---|
| ar25 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| bp35 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| cd82 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| cn04 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| dc22 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| ft09 | dev | reset survey | seed-7 random trajectory replayed to level 1 (reset probe); game-over, speed, determinism probes | random agent (3 games), game-over, speed, determinism, reset, scorer probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| g50t | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| ka59 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| lf52 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| lp85 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| ls20 | dev | reset survey; frame inspection | - | random agent (3 games), determinism, speed, game-over, scorer probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| m0r0 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| r11l | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| re86 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| s5i5 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| sb26 | reserved | reset survey | - | game-over probe; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| sc25 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| sk48 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| sp80 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| su15 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| tn36 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| tr87 | dev | reset survey | - | speed, game-over probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| tu93 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| vc33 | dev | reset survey; ACTION6 probes | - | random agent (3 games), determinism, speed, game-over probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
| wa30 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | cluster architecture search | cluster DQN/exploration development | 2026-09-21 to 24 |
