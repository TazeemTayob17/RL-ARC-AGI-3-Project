# Phase 5b results: Improvement 3 (object-level clicks with effect memory)

Grid: 17 dev games x 3 seeds x 6 conditions = 306 runs, 1500 scorer-counted actions, env seed 0. I3 constants fixed in src/objects.py, not tuned.
Scores are official local SDK scores, not Kaggle results; all public games have prior development exposure.

| Comparison | B (reference) mean | A (treatment) mean | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels ref -> treat | Progress runs ref -> treat | Median first progress action ref -> treat | Frame-change rate ref -> treat | Games regressed |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| dqn V3 vs V1 | 0.346419 | 0.345492 | -0.000927 | [-0.067875, +0.053109] | crosses zero (inconclusive) | 12 -> 18 | 12/51 -> 16/51 | 74.0 -> 85.0 | 0.692 -> 0.719 | 4: cd82, m0r0, r11l, tn36 |
| dqn A3 vs V0 | 0.337304 | 0.323878 | -0.013426 | [-0.072165, +0.032221] | crosses zero (inconclusive) | 9 -> 12 | 9/51 -> 12/51 | 38 -> 54.5 | 0.657 -> 0.692 | 3: r11l, sp80, tn36 |
| dqn V3 vs A3 | 0.323878 | 0.345492 | +0.021613 | [+0.002684, +0.046930] | above zero | 12 -> 18 | 12/51 -> 16/51 | 54.5 -> 85.0 | 0.692 -> 0.719 | 1: cd82 |
| dqn V3 vs V3nm | 0.325707 | 0.345492 | +0.019785 | [-0.003896, +0.048679] | crosses zero (inconclusive) | 15 -> 18 | 14/51 -> 16/51 | 85.0 -> 85.0 | 0.709 -> 0.719 | 2: cd82, m0r0 |
| dqn V3nm vs V1 | 0.346419 | 0.325707 | -0.020712 | [-0.082630, +0.032430] | crosses zero (inconclusive) | 12 -> 15 | 12/51 -> 14/51 | 74.0 -> 85.0 | 0.692 -> 0.709 | 5: cd82, m0r0, r11l, sp80, tn36 |
| ppo V3 vs V1 | 0.067810 | 0.261662 | +0.193852 | [-0.004613, +0.452738] | crosses zero (inconclusive) | 12 -> 14 | 11/51 -> 14/51 | 225 -> 220.0 | 0.788 -> 0.816 | 1: m0r0 |
| ppo A3 vs V0 | 0.066544 | 0.262343 | +0.195799 | [-0.003045, +0.456389] | crosses zero (inconclusive) | 12 -> 13 | 12/51 -> 13/51 | 272.5 -> 175 | 0.784 -> 0.816 | 2: ar25, m0r0 |
| ppo V3 vs A3 | 0.262343 | 0.261662 | -0.000681 | [-0.002437, +0.000342] | crosses zero (inconclusive) | 13 -> 14 | 13/51 -> 14/51 | 175 -> 220.0 | 0.816 -> 0.816 | 2: ft09, m0r0 |
| ppo V3 vs V3nm | 0.264303 | 0.261662 | -0.002641 | [-0.007654, +0.000083] | crosses zero (inconclusive) | 14 -> 14 | 13/51 -> 14/51 | 175 -> 220.0 | 0.796 -> 0.816 | 2: cd82, ft09 |
| ppo V3nm vs V1 | 0.067810 | 0.264303 | +0.196493 | [+0.000016, +0.456101] | above zero | 12 -> 14 | 11/51 -> 13/51 | 225 -> 175 | 0.788 -> 0.796 | 1: m0r0 |

## Mechanism activation

- dqn V3: runs offering click candidates 39/51; dropped as dead 10.3% of 2506711 offered; outcomes recorded 36311; retries 12858.
- dqn A3: runs offering click candidates 39/51; dropped as dead 10.3% of 2486393 offered; outcomes recorded 37748; retries 12915.
- dqn V3NM: runs offering click candidates 39/51; dropped as dead 0.0% of 2180412 offered; outcomes recorded 0; retries 0.
- ppo V3: runs offering click candidates 39/51; dropped as dead 13.6% of 2931730 offered; outcomes recorded 27778; retries 20039.
- ppo A3: runs offering click candidates 39/51; dropped as dead 12.2% of 2918859 offered; outcomes recorded 26123; retries 17914.
- ppo V3NM: runs offering click candidates 39/51; dropped as dead 0.0% of 2931383 offered; outcomes recorded 0; retries 0.

A run that offered no click candidates (keyboard-only game) is not evidence about I3.

## Limits

Short development runs on exposed public games, one environment seed, three learner seeds. I3 constants (tile size, candidate cap, retry probability, dead-signature threshold) were fixed before any run. An interval crossing zero is reported as inconclusive. No Kaggle or unseen-game evidence.
