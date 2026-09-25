# Exposure table (PROJECT_PLAN 3.3)

All 25 public games were already used in architecture development (plan 3.3), so **no public game is untouched**.
"Reserved" below means reserved from *further* tuning only. Split: `configs/splits.json`, hash `8819bd3b28d9`,
status **proposed, awaiting team confirmation** (plan question 3). Rule: stratified by action-set type, seeded,
and a game whose completion trajectory was inspected (ft09) is never reserved.

Columns filled from this repo's recon and Phase 1 probes (`scripts/probe/`, `docs/evidence/`). The
"architecture development" column must be filled by whoever ran the architecture-selection experiment; it is
not known from this repo. "Training/tuning" is none so far (no learner exists yet).

| game | split | metadata inspected | trajectories inspected | random / probe runs | architecture development | training/tuning | dates |
|---|---|---|---|---|---|---|---|
| ar25 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| bp35 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| cd82 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| cn04 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| dc22 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| ft09 | dev | reset survey | seed-7 random trajectory replayed to level 1 (reset probe); game-over, speed, determinism probes | random agent (3 games), game-over, speed, determinism, reset, scorer probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| g50t | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| ka59 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| lf52 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| lp85 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| ls20 | dev | reset survey; frame inspection | - | random agent (3 games), determinism, speed, game-over, scorer probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| m0r0 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| r11l | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| re86 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| s5i5 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| sb26 | reserved | reset survey | - | game-over probe; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| sc25 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| sk48 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| sp80 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| su15 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| tn36 | dev | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| tr87 | dev | reset survey | - | speed, game-over probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| tu93 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| vc33 | dev | reset survey; ACTION6 probes | - | random agent (3 games), determinism, speed, game-over probes; Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
| wa30 | reserved | reset survey (available_actions, win_levels) | - | Phase 1 local random reference: 1000 actions, type-uniform + flat-uniform, seed 0 | to be supplied | none | 2026-09-21 to 24 |
