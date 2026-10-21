# Phase 4 results: Improvement 1 (count-based novelty bonus)

Grid: 17 dev games × 3 seeds × 2 algorithms = 102 runs, 1500 scorer-counted actions, env seed 0.
Treatment: bonus beta/sqrt(n) on exact observed grids (beta = 0.01, predeclared, not tuned), zero on action/undo loops.
Matched baseline: Phase 3 records of the same games and seeds. Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.

| Algorithm | Baseline mean score | I1 mean score | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels base → I1 | Progress runs base → I1 | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|
| dqn | 0.337304 | 0.346419 | +0.009115 | [+0.000000, +0.023143] | crosses zero (inconclusive) | 9 → 12 | 9/51 → 12/51 | 0: - |
| ppo | 0.066544 | 0.067810 | +0.001266 | [-0.000203, +0.003345] | crosses zero (inconclusive) | 12 → 12 | 12/51 → 11/51 | 1: ar25 |

## I2 selection rule (fixed before Phase 4; docs/PROJECT_PLAN.md 7.2)

- Fraction of I1 runs completing Level 1: DQN 0.235, PPO 0.216 (threshold 1/3).
- Rule 1: fewer than one third reach Level 1 for both algorithms → choose **A** (event-terminated repeat option).

## Mechanism activation

- dqn: mean nonzero-bonus steps per run 1305; mean total bonus 10.12.
- ppo: mean nonzero-bonus steps per run 1375; mean total bonus 11.16.

## Limits

Short development runs on exposed public games, one environment seed, three learner seeds. An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence.
