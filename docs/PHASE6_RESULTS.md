# Phase 6 results: final evaluation on the reserved games

Grid: 8 reserved games x 5 seeds x 9 conditions = 360 runs, 1500 scorer-counted actions, env seed 0. Protocol fixed in docs/PHASE6_FREEZE_RULE.md before the run.
Scores are official local SDK scores, not Kaggle results; reserved means reserved from further tuning, not unseen (docs/EXPOSURE.md).

## Mean official score per version

| Condition | Mean score | Levels | Games with a level | Level-1 runs | Median first progress |
|---|---:|---:|---:|---:|---:|
| dqn_v0 | 0.005483 | 4 | 1/8 | 4/40 | 230.0 |
| dqn_v1 | 0.005483 | 4 | 1/8 | 4/40 | 230.0 |
| dqn_v2 | 0.002656 | 5 | 1/8 | 5/40 | 556 |
| dqn_a2 | 0.002689 | 5 | 1/8 | 5/40 | 395 |
| dqn_v2np | 0.002033 | 4 | 1/8 | 4/40 | 483.5 |
| ppo_v0 | 0.002541 | 4 | 1/8 | 4/40 | 646.0 |
| ppo_v1 | 0.002548 | 4 | 1/8 | 4/40 | 640.5 |
| ppo_v2 | 0.001389 | 7 | 3/8 | 7/40 | 474 |
| ppo_a2 | 0.000698 | 6 | 2/8 | 6/40 | 562.5 |

## Paired comparisons (game-cluster bootstrap over 8 reserved games; wide intervals expected)

| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress ref -> treat | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|---|
| dqn V1 vs V0 | 0.005483 | 0.005483 | +0.000000 | [+0.000000, +0.000000] | crosses zero (inconclusive) | 4 -> 4 | 4/40 -> 4/40 | 230.0 -> 230.0 | 0: - |
| dqn V2 vs V0 | 0.005483 | 0.002656 | -0.002827 | [-0.008482, +0.000000] | crosses zero (inconclusive) | 4 -> 5 | 4/40 -> 5/40 | 230.0 -> 556 | 1: lp85 |
| dqn V2 vs V1 | 0.005483 | 0.002656 | -0.002827 | [-0.008482, +0.000000] | crosses zero (inconclusive) | 4 -> 5 | 4/40 -> 5/40 | 230.0 -> 556 | 1: lp85 |
| dqn V2 vs A2 | 0.002689 | 0.002656 | -0.000032 | [-0.000097, +0.000000] | crosses zero (inconclusive) | 5 -> 5 | 5/40 -> 5/40 | 395 -> 556 | 1: lp85 |
| dqn V2NP vs V2 | 0.002656 | 0.002033 | -0.000623 | [-0.001869, +0.000000] | crosses zero (inconclusive) | 5 -> 4 | 5/40 -> 4/40 | 556 -> 483.5 | 1: lp85 |
| dqn V2NP vs V0 | 0.005483 | 0.002033 | -0.003450 | [-0.010351, +0.000000] | crosses zero (inconclusive) | 4 -> 4 | 4/40 -> 4/40 | 230.0 -> 483.5 | 1: lp85 |
| ppo V1 vs V0 | 0.002541 | 0.002548 | +0.000007 | [+0.000000, +0.000022] | crosses zero (inconclusive) | 4 -> 4 | 4/40 -> 4/40 | 646.0 -> 640.5 | 0: - |
| ppo V2 vs V0 | 0.002541 | 0.001389 | -0.001152 | [-0.005705, +0.001897] | crosses zero (inconclusive) | 4 -> 7 | 4/40 -> 7/40 | 646.0 -> 474 | 1: lp85 |
| ppo V2 vs V1 | 0.002548 | 0.001389 | -0.001159 | [-0.005727, +0.001897] | crosses zero (inconclusive) | 4 -> 7 | 4/40 -> 7/40 | 640.5 -> 474 | 1: lp85 |
| ppo V2 vs A2 | 0.000698 | 0.001389 | +0.000691 | [-0.000007, +0.001724] | crosses zero (inconclusive) | 6 -> 7 | 6/40 -> 7/40 | 562.5 -> 474 | 1: lp85 |
| DQN V2np vs PPO V2 | 0.001389 | 0.002033 | +0.000644 | [-0.001897, +0.004181] | crosses zero (inconclusive) | 7 -> 4 | 7/40 -> 4/40 | 474 -> 483.5 | 2: cn04, sk48 |

## Mean score per game

| Game | dqn_v0 | dqn_v1 | dqn_v2 | dqn_a2 | dqn_v2np | ppo_v0 | ppo_v1 | ppo_v2 | ppo_a2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bp35 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cn04 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0041 | 0.0000 |
| lp85 | 0.0439 | 0.0439 | 0.0212 | 0.0215 | 0.0163 | 0.0203 | 0.0204 | 0.0042 | 0.0042 |
| s5i5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sb26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sk48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0028 | 0.0014 |
| tu93 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wa30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Validity gates passed: all records present, scorer count equals SDK total, activation checks (bonus, prior mode).

## Limits

Eight reserved public games, one environment seed, five learner seeds; all games were exposed to earlier development. An interval crossing zero is inconclusive. No Kaggle or unseen-game evidence.
