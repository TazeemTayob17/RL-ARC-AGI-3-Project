# Phase 5d results: DQN without the pretrained prior

Grid: 17 dev games x 3 seeds x 2 conditions = 102 runs, 1500 scorer-counted actions, env seed 0. Random frozen encoder (fixed seed) instead of the pretrained prior; not tuned.
Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.

| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| V3np vs V3 (primary) | 0.345492 | 0.369949 | +0.024457 | [-0.041950, +0.126357] | crosses zero (inconclusive) | 18 -> 13 | 16/51 -> 13/51 | 85.0 -> 57 | 0.719 -> 0.721 | 4: sc25, sp80, tr87, vc33 |
| V0np vs V0 | 0.337304 | 0.336895 | -0.000409 | [-0.005094, +0.003866] | crosses zero (inconclusive) | 9 -> 10 | 9/51 -> 10/51 | 38 -> 52.5 | 0.657 -> 0.665 | 2: sp80, vc33 |
| V3np vs V0np | 0.336895 | 0.369949 | +0.033054 | [-0.053145, +0.144350] | crosses zero (inconclusive) | 10 -> 13 | 10/51 -> 13/51 | 52.5 -> 57 | 0.665 -> 0.721 | 4: m0r0, r11l, sp80, tn36 |
| V3np vs PPO V3 | 0.261662 | 0.369949 | +0.108287 | [-0.316709, +0.565753] | crosses zero (inconclusive) | 14 -> 13 | 14/51 -> 13/51 | 220.0 -> 57 | 0.816 -> 0.721 | 4: cd82, ft09, m0r0, sp80 |

Activation: runs not using the random prior: 0.

## Limits

Short development runs on exposed public games, one environment seed, three learner seeds. An interval crossing zero is inconclusive.
