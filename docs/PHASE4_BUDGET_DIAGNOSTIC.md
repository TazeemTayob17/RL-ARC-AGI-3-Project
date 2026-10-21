# Phase 4 budget-sensitivity diagnostic (not an improvement)

Baseline vs learning-off at 5000 scorer-counted actions on 5 predeclared dev games × 3 seeds × both algorithms.
Question: does learning start to matter at a larger budget? Scores are official local SDK scores; public games are exposed.

| Algorithm | Mean score (learning) | Mean score (off) | Paired difference | Game-cluster bootstrap 95% | Verdict | Levels learn / off | Progress runs learn / off | Mean gradients |
|---|---:|---:|---:|---|---|---|---|---:|
| dqn | 0.006826 | 0.000000 | +0.006826 | [+0.000009, +0.020363] | above zero | 3 / 0 | 3/15 / 0/15 | 617 |
| ppo | 0.094059 | 0.094282 | -0.000223 | [-0.007158, +0.004477] | crosses zero (inconclusive) | 7 / 7 | 7/15 / 7/15 | 623 |

If learning helps here, later matched comparisons are rerun at this budget; otherwise the 1500-action protocol stands.
