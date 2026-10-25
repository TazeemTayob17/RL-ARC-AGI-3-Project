# Phase 5d verdict: DQN without the pretrained prior

Written after the grid, against `docs/PHASE5D_NOPRIOR_CRITERIA.md`. Numbers are in `docs/PHASE5D_RESULTS.md`. Local SDK scores on exposed public games, not Kaggle results.

## Validity
102 of 102 records present, scorer count equals SDK total in every run, `ACTIVATION OK`: every run used the random frozen encoder and diverged from its matched pretrained run.

## Results
| Comparison | Paired difference | 95% interval | Verdict |
|---|---:|---|---|
| V3np vs V3 (primary) | +0.024 | [-0.042, +0.126] | inconclusive |
| V0np vs V0 | -0.0004 | [-0.005, +0.004] | inconclusive (effectively identical) |
| V3np vs V0np | +0.033 | [-0.053, +0.144] | inconclusive |
| V3np vs PPO V3 | +0.108 | [-0.317, +0.566] | inconclusive |

Secondary: levels 18 (V3) to 13 (V3np), Level-1 runs 16/51 to 13/51, median first-progress action 85 to 57; baseline levels 9 to 10, Level-1 runs 9/51 to 10/51.

## Conclusions
1. **The pretrained prior does not measurably help.** Removing it leaves the mean score unchanged (V0np equals V0 to three decimals; V3np is nominally higher). The only signal against the prior-free variant is fewer levels and Level-1 runs at the V3 stage (13/51 vs 16/51); that is within noise of a three-seed, two-game-dominated comparison and is reported, not explained away.
2. **The DQN-versus-PPO gap is not explained by the prior.** The prior-free DQN V3 still scores above PPO V3 nominally (+0.108), with an interval that crosses zero. The confound (W9) is therefore reduced: the DQN's lead does not come from the pretrained encoder. The remaining differences are the Q-learner and its structured exploration.
3. **Deployment and exposure.** Predeclared reading: if V3np is not below V3, the prior-free DQN is preferred for deployment. It is not below V3 on the primary comparison, so the prior-free DQN is the preferred option for reserved-game evaluation and Kaggle: it needs no checkpoint, no game-to-checkpoint mapping, and no pretraining on public-game data (which includes reserved games).
4. Caveat for the final choice: the Level-1 count favours the pretrained prior by 3 runs. The choice between them is a freeze decision made on development evidence in the selection step, with this caveat stated.

## Limits
Short development runs on exposed public games, one environment seed, three learner seeds; score dominated by two games. Intervals crossing zero stay inconclusive.
