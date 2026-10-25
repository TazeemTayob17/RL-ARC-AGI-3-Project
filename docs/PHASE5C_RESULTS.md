# Phase 5c results: Improvement 4 (frontier-directed exploration option)

Grid: 17 dev games x 3 seeds x 6 conditions = 306 runs, 1500 scorer-counted actions, env seed 0. I4 constants fixed in src/frontier.py and src/dqn_v1/learner.py, not tuned.
Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.

| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| dqn V4 vs V3 | 0.345492 | 0.235327 | -0.110165 | [-0.263518, -0.006863] | below zero | 18 -> 18 | 16/51 -> 15/51 | 85.0 -> 148 | 0.719 -> 0.714 | 6: r11l, sc25, sp80, tn36, tr87, vc33 |
| dqn A4 vs V0 | 0.337304 | 0.302595 | -0.034709 | [-0.205556, +0.103821] | crosses zero (inconclusive) | 9 -> 9 | 9/51 -> 9/51 | 38 -> 39 | 0.657 -> 0.693 | 3: sp80, tn36, vc33 |
| dqn V4 vs A4 | 0.302595 | 0.235327 | -0.067268 | [-0.312356, +0.083253] | crosses zero (inconclusive) | 9 -> 18 | 9/51 -> 15/51 | 39 -> 148 | 0.693 -> 0.714 | 1: r11l |
| dqn V4 vs V4f | 0.137664 | 0.235327 | +0.097663 | [-0.173914, +0.452111] | crosses zero (inconclusive) | 8 -> 18 | 8/51 -> 15/51 | 193.0 -> 148 | 0.734 -> 0.714 | 1: r11l |
| dqn V4f vs V3 | 0.345492 | 0.137664 | -0.207828 | [-0.548858, -0.017123] | below zero | 18 -> 8 | 16/51 -> 8/51 | 85.0 -> 193.0 | 0.719 -> 0.734 | 7: lf52, r11l, sc25, sp80, tn36, tr87, vc33 |
| dqn V4 vs V1 | 0.346419 | 0.235327 | -0.111092 | [-0.296954, +0.022281] | crosses zero (inconclusive) | 12 -> 18 | 12/51 -> 15/51 | 74.0 -> 148 | 0.692 -> 0.714 | 4: cd82, m0r0, r11l, tn36 |
| ppo V4 vs V3 | 0.261662 | 0.274404 | +0.012742 | [-0.259511, +0.285358] | crosses zero (inconclusive) | 14 -> 14 | 14/51 -> 14/51 | 220.0 -> 188.0 | 0.816 -> 0.810 | 2: cd82, m0r0 |
| ppo A4 vs V0 | 0.066544 | 0.210071 | +0.143527 | [-0.010658, +0.393122] | crosses zero (inconclusive) | 12 -> 9 | 12/51 -> 9/51 | 272.5 -> 109 | 0.784 -> 0.781 | 3: ar25, cd82, ft09 |
| ppo V4 vs A4 | 0.210071 | 0.274404 | +0.064333 | [-0.016404, +0.203310] | crosses zero (inconclusive) | 9 -> 14 | 9/51 -> 14/51 | 109 -> 188.0 | 0.781 -> 0.810 | 3: ls20, m0r0, sp80 |
| ppo V4 vs V4f | 0.137664 | 0.274404 | +0.136740 | [+0.001514, +0.394793] | above zero | 8 -> 14 | 8/51 -> 14/51 | 193.0 -> 188.0 | 0.734 -> 0.810 | 0: - |
| ppo V4f vs V3 | 0.261662 | 0.137664 | -0.123998 | [-0.474114, +0.176508] | crosses zero (inconclusive) | 14 -> 8 | 14/51 -> 8/51 | 220.0 -> 193.0 | 0.816 -> 0.734 | 4: cd82, ft09, m0r0, sp80 |
| ppo V4 vs V1 | 0.067810 | 0.274404 | +0.206594 | [-0.013406, +0.526825] | crosses zero (inconclusive) | 12 -> 14 | 11/51 -> 14/51 | 225 -> 188.0 | 0.788 -> 0.810 | 2: cd82, m0r0 |

## Mechanism activation

- dqn V4: option offered at 100.0% of 75541 decisions; chosen 20309 times (runs with at least one: 51/51); 20365 of 75597 steps inside options; mean option length 1.00; walks reaching the expected node 56/58; new states per 100 steps inside / outside options 57.0 / 48.5; deviations 2.
- dqn A4: option offered at 100.0% of 75394 decisions; chosen 33361 times (runs with at least one: 51/51); 33522 of 75555 steps inside options; mean option length 1.00; walks reaching the expected node 157/246; new states per 100 steps inside / outside options 55.1 / 51.0; deviations 89.
- dqn V4F: option offered at 100.0% of 75273 decisions; chosen 75273 times (runs with at least one: 51/51); 75526 of 75526 steps inside options; mean option length 1.00; walks reaching the expected node 252/360; new states per 100 steps inside / outside options 52.5 / 0.0; deviations 107.
- ppo V4: option offered at 100.0% of 75437 decisions; chosen 17580 times (runs with at least one: 51/51); 17605 of 75462 steps inside options; mean option length 1.00; walks reaching the expected node 25/35; new states per 100 steps inside / outside options 56.0 / 64.7; deviations 10.
- ppo A4: option offered at 100.0% of 75438 decisions; chosen 17522 times (runs with at least one: 51/51); 17552 of 75468 steps inside options; mean option length 1.00; walks reaching the expected node 30/43; new states per 100 steps inside / outside options 51.7 / 56.9; deviations 13.
- ppo V4F: option offered at 100.0% of 75273 decisions; chosen 75273 times (runs with at least one: 51/51); 75526 of 75526 steps inside options; mean option length 1.00; walks reaching the expected node 252/360; new states per 100 steps inside / outside options 52.5 / 0.0; deviations 107.

## Limits

Short development runs on exposed public games, one environment seed, three learner seeds. I4 constants (maximum walk length, exploration share of the option for DQN, uniform choice of the untried action) were fixed before any run. An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence.
