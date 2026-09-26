# Phase 3 selected baseline results

Complete development grid: 17 games × 3 seeds × 6 conditions = 306 runs.
Budget: 1500 scorer-counted commands; env seed 0; one CPU thread.
Scores are official local SDK scores, not Kaggle results. All public games have prior development exposure.
DQN implementation V1 is the adopted assignment baseline; PPO uses the existing provisional V0 architecture.

| Condition | Mean game score | Total levels across runs | Progress runs | Mean gradients | Mean seconds |
|---|---:|---:|---:|---:|---:|
| dqn | 0.337304 | 9 | 9/51 | 183.9 | 166.8 |
| ppo | 0.066544 | 12 | 12/51 | 191.9 | 183.9 |
| dqn_no_updates | 0.339643 | 9 | 9/51 | 0.0 | 77.3 |
| ppo_no_updates | 0.065673 | 11 | 11/51 | 0.0 | 21.3 |
| random_type | 0.021906 | 10 | 10/51 | 0.0 | 5.0 |
| random_flat | 0.001460 | 5 | 5/51 | 0.0 | 5.1 |

## Learning versus matching controls

- dqn: mean score difference -0.002339; approximate game-cluster bootstrap 95% interval [-0.013018, +0.005976].
- ppo: mean score difference +0.000871; approximate game-cluster bootstrap 95% interval [-0.005369, +0.007470].

## Diagnostic hypotheses before final selection

- dqn: 42/51 runs found no progress. Candidate I1: test stagnation-triggered recovery/revisit control in addition to the existing structured explorer. Test against this frozen baseline; this is a hypothesis, not a demonstrated improvement.
- ppo: 39/51 runs found no progress. Candidate I1: test a matched salience-guided click-exploration treatment. Test against this frozen baseline; this is a hypothesis, not a demonstrated improvement.

## Limits

The teammate five-game figures are confirmed by archived scorecards (docs/TEAMMATE_COMPARISON_VERIFICATION.md); they are one-seed, short-budget whole-system evidence, not an isolated online-learning result. This bounded evaluation does not establish convergence or unseen-game generalization. The group later accepted the baselines, split and budget and reported tutor approval; no matched RL tournament of architecture finalists is claimed. Competition deployment/testing remains later work. Existing DQN features cannot be credited again as later improvements.

## Final group decision

The group delegated improvement selection. I1 selected for both algorithms: count-based intrinsic training reward. See PHASE3_HANDOFF.md. Its implementation and pilot are Phase 4 work, excluded from this branch.
