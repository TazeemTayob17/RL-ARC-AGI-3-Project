# Phase 5c verdict: Improvement 4 (frontier-directed exploration option)

Written after the grid, against the criteria fixed in `docs/PHASE5C_I4_CRITERIA.md`. Numbers are in `docs/PHASE5C_RESULTS.md`. Scores are local SDK scores on exposed public games, not Kaggle results.

## Validity

- 306 of 306 records present; scorer count equals SDK total in every run (checked by the summary script).
- Activation check: `ACTIVATION OK`. The option was offered in every run, chosen in 51/51 runs of every learned condition, and actions diverge from the matched reference in every run. So the results are evidence about I4.
- Mechanism finding: the option almost always has length 1 (mean 1.00). Walks to a distant frontier were started only 58 times in DQN V4 and 35 times in PPO V4, out of about 20,000 and 17,000 option uses. I4 therefore acted as "try an untried action here", not as navigation through the graph. This was a stated risk (criteria, risk 3).
- Where walks did start, the graph was accurate for V4 (DQN 56/58, PPO 25/35) and less so in A4/V4f (DQN A4 157/246, V4f 252/360), with 89 to 107 deviations per grid.

## Primary comparisons

| Comparison | Paired difference | 95% interval | Verdict |
|---|---:|---|---|
| DQN V4 vs V3 | -0.110 | [-0.264, -0.007] | below zero (harm) |
| PPO V4 vs V3 | +0.013 | [-0.260, +0.285] | inconclusive |
| DQN A4 vs V0 | -0.035 | [-0.206, +0.104] | inconclusive |
| PPO A4 vs V0 | +0.144 | [-0.011, +0.393] | inconclusive (close to above zero) |

## Secondary and attribution

- Levels and Level-1 runs: DQN V4 18 levels, 15/51 progress runs (V3: 18, 16/51). PPO V4 14 levels, 14/51 (V3: 14, 14/51). DQN A4 9, 9/51 (V0: 9, 9/51). PPO A4 9 levels, 9/51 (V0: 12 levels, 12/51).
- DQN V4 regressed on 6 games (r11l, sc25, sp80, tn36, tr87, vc33) and first progress moved later (action 85 to 148). The score loss is the main primary signal and matches risk 4 (tn36 and r11l displaced).
- The training-free forced explorer (V4f) is worse than V3 for DQN (-0.208, below zero) and inconclusive for PPO. Always taking the option is not a good policy on its own. PPO V4 beats V4f (+0.137, above zero), so learning when to use the option helps PPO. The DQN V4 vs V4f interval crosses zero.
- Forced mode never fell back to the learner in practice, so DQN V4f and PPO V4f are the same policy and have the same score.
- PPO A4 raised the mean score over V0 but reduced Level-1 runs (12 to 9). The score gain is therefore not a progress gain. It is within the score concentration on two games and its interval crosses zero.

## Conclusions

1. I4 is evaluated. Result: no improvement. One predeclared comparison shows harm (DQN V4 vs V3), and the other three are inconclusive.
2. I4 is not promoted into the final agent for DQN. For PPO the evidence does not support promotion either (V4 vs V3 +0.013, interval wide).
3. **I5 gate: not met.** The best Level-1 rate across I3 and I4 conditions is DQN V3 at 16/51 = 31.4%, below one third. I4 conditions reached 15/51 (DQN V4) and 14/51 (PPO V4). I5 is skipped and the reason recorded here.
4. Likely reason (hypothesis, not tested here): with about 100 candidate actions per node, almost every seen state still has untried actions, so the graph rarely gets to plan a walk. A cheaper candidate set would be needed before the graph can matter. That would be a new, separately predeclared intervention.

## Limits

Short dev runs on exposed public games, one environment seed, three learner seeds, score dominated by two games. Intervals crossing zero stay inconclusive. No Kaggle or unseen-game evidence.
