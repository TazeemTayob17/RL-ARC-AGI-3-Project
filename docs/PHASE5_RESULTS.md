# Phase 5 results: Improvement 2 (event-terminated repeat option)

Grid: 17 dev games x 3 seeds x 4 conditions = 204 runs, 1500 scorer-counted actions, env seed 0. Option cap 8, engagement probability 0.5 (predeclared, not tuned).
Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.

| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|
| dqn V2 vs V1 | 0.346419 | 0.011390 | -0.335029 | [-0.821224, +0.003236] | crosses zero (inconclusive) | 12 -> 8 | 12/51 -> 7/51 | 5: cd82, m0r0, r11l, sp80, tn36 |
| dqn A2 vs V0 | 0.337304 | 0.008743 | -0.328561 | [-0.810058, +0.009644] | crosses zero (inconclusive) | 9 -> 7 | 9/51 -> 6/51 | 3: r11l, sp80, tn36 |
| dqn V2 vs A2 | 0.008743 | 0.011390 | +0.002646 | [-0.000622, +0.008214] | crosses zero (inconclusive) | 7 -> 8 | 6/51 -> 7/51 | 1: m0r0 |
| ppo V2 vs V1 | 0.067810 | 0.055459 | -0.012352 | [-0.123982, +0.136137] | crosses zero (inconclusive) | 12 -> 5 | 11/51 -> 5/51 | 5: cd82, ft09, m0r0, r11l, sp80 |
| ppo A2 vs V0 | 0.066544 | 0.057905 | -0.008639 | [-0.120717, +0.142568] | crosses zero (inconclusive) | 12 -> 6 | 12/51 -> 5/51 | 5: cd82, ft09, m0r0, r11l, sp80 |
| ppo V2 vs A2 | 0.057905 | 0.055459 | -0.002447 | [-0.007896, +0.000979] | crosses zero (inconclusive) | 6 -> 5 | 5/51 -> 5/51 | 2: ar25, g50t |

## Mechanism activation

- dqn V2: option fired in 51/51 runs; mean 217 options longer than one per run; mean steps per decision 2.64; mean decisions per run 684; stop reasons {'cap': 3926, 'no_effect': 18197, 'cycle': 2, 'level_up': 8, 'terminal': 1043, 'illegal': 0, 'budget': 49, 'not_engaged': 11684}.
- dqn A2: option fired in 51/51 runs; mean 206 options longer than one per run; mean steps per decision 2.61; mean decisions per run 711; stop reasons {'cap': 3880, 'no_effect': 20337, 'cycle': 5, 'level_up': 7, 'terminal': 930, 'illegal': 0, 'budget': 51, 'not_engaged': 11072}.
- ppo V2: option fired in 51/51 runs; mean 208 options longer than one per run; mean steps per decision 2.85; mean decisions per run 615; stop reasons {'cap': 4052, 'no_effect': 15026, 'cycle': 25, 'level_up': 5, 'terminal': 1018, 'illegal': 0, 'budget': 51, 'not_engaged': 11209}.
- ppo A2: option fired in 51/51 runs; mean 208 options longer than one per run; mean steps per decision 2.84; mean decisions per run 617; stop reasons {'cap': 4039, 'no_effect': 15140, 'cycle': 25, 'level_up': 6, 'terminal': 1016, 'illegal': 0, 'budget': 51, 'not_engaged': 11213}.

A run whose option never fires is not evidence about A (failure rule in the 7.3 card).

## Limits

Short development runs on exposed public games, one environment seed, three learner seeds. The option is a behaviour-level mechanism with fixed cap and engagement probability, not a learned choice; any benefit is a benefit of that rule. Part of the evidence that motivated A came from the reserved game lp85 (disclosed in docs/EXPOSURE.md). An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence.
