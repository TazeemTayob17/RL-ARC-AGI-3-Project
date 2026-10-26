# Phase 6 verdict: final evaluation on the reserved games

Written after the grid, against `docs/PHASE6_FREEZE_RULE.md`. Numbers are in `docs/PHASE6_RESULTS.md`. Local SDK scores on the 8 reserved public games, not Kaggle results. All 25 public games were exposed to earlier development (`docs/EXPOSURE.md`), and part of the repeat-option evidence came from lp85.

## Validity
- 360 of 360 records present; the scorer count equals the SDK total in every run; every run ended on budget.
- Validity gates passed (I1 bonus nonzero in V1, V2 and V2np; random prior in use in V2np; pretrained prior in the other DQN versions).
- Activation: in all 40 DQN and all 40 PPO paired runs, V1 actions diverge from V0 (median first divergence at action 113 for DQN and 185 for PPO), and V2 diverges from A2 in all 80. The versions are really different agents, and the runs are evidence about them.

## What the numbers say
| Version | Mean score | Levels | Games with a level | Level-1 runs |
|---|---:|---:|---:|---:|
| DQN V0 | 0.0055 | 4 | 1/8 | 4/40 |
| DQN V1 | 0.0055 | 4 | 1/8 | 4/40 |
| DQN V2 | 0.0027 | 5 | 1/8 | 5/40 |
| DQN A2 | 0.0027 | 5 | 1/8 | 5/40 |
| DQN V2np | 0.0020 | 4 | 1/8 | 4/40 |
| PPO V0 | 0.0025 | 4 | 1/8 | 4/40 |
| PPO V1 | 0.0025 | 4 | 1/8 | 4/40 |
| PPO V2 | 0.0014 | 7 | 3/8 | 7/40 |
| PPO A2 | 0.0007 | 6 | 2/8 | 6/40 |

1. **Every paired comparison is inconclusive** (all intervals include zero), and the intervals are wide: eight game clusters, and only one game scores for most versions.
2. **Scores are tiny and concentrated.** Seven of the eight reserved games score zero for every DQN version. All DQN score comes from lp85. PPO V2 and A2 also score on cn04 and sk48.
3. **The development gains did not transfer in score.** The nominal means fall from V0/V1 to V2 for both algorithms (DQN 0.0055 to 0.0027, PPO 0.0025 to 0.0014), driven by lp85, where V0/V1 do better.
4. **The only positive signal is breadth of progress for PPO.** PPO V2 reached a level in 3 of 8 games (7/40 runs) against 1 of 8 (4/40) for V0 and V1. This is a count difference of three runs and is not significant.
5. **The Level-1 rate is much lower than on development games** (10 to 17.5% of runs, against about 31% on dev for the best conditions). The development results are an optimistic picture of performance on the reserved games.

## Prior-free DQN (the nomination rule)
The freeze rule nominates DQN V2np unless the whole interval of V2np vs V2 lies below zero. V2np vs V2 is -0.0006 with an interval of [-0.0019, 0.0000]: the upper bound sits exactly at zero (seven of the eight game differences are zero), so the rule's condition is not met, but the result is borderline and the evidence for preferring V2np is weak. The nomination stays as predeclared, with this caveat. V2np has 4/40 Level-1 runs against 5/40 for V2. The decision between them is returned to the team.

## Conclusions
1. The two improvements chosen on development evidence (I1, I3) do not show a measurable benefit on the reserved games. I1 changes behaviour but not the score; I3 reaches more games with a level for PPO and fewer points on lp85.
2. This is the honest final-evaluation result and goes in the report as it stands: no tuning on these games happened, and none will.
3. The most likely reading is that agents with this budget solve very few games, and the dev-game gains were partly specific to those games.
4. No claim of unseen-game performance is made; only the Kaggle evaluation can supply that.

## Limits
Eight reserved public games, one environment seed, five learner seeds, tiny scores concentrated in one game. Intervals crossing zero stay inconclusive.
