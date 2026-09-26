# New five-game DQN comparison

This is an independent, reproducible comparison. The original five-game scores have now been confirmed from the teammate archive (see TEAMMATE_COMPARISON_VERIFICATION.md). This new protocol was declared before that archive arrived and differs in games, seeds and budget.

Games: ft09, g50t, ls20, ar25, sp80. Seeds: 0, 1, 2. Environment seed: 0. Budget: 1500 scorer-counted commands. One CPU thread. V1 records are reused from the matched Phase 3 grid.

| Implementation | Mean official local game score | Total levels across 15 runs | Mean gradient steps |
|---|---:|---:|---:|
| V0 | 0.656670 | 3 | 121.4 |
| V1 | 0.006773 | 1 | 183.7 |

## Per-game mean scores over three seeds

| Game | Original V0 | Selected V1 | V1 minus V0 |
|---|---:|---:|---:|
| ft09 | 0.000000 | 0.000000 | +0.000000 |
| g50t | 0.000000 | 0.000000 | +0.000000 |
| ls20 | 0.000000 | 0.000000 | +0.000000 |
| ar25 | 0.000000 | 0.000000 | +0.000000 |
| sp80 | 3.283350 | 0.033866 | -3.249484 |

Observed mean score difference: -0.649897. Both systems made training updates in every run.

This compares two complete systems: representation, pretraining, exploration, reward convention, value head and training schedules differ. It does not isolate the effect of one improvement. The original V0 warm-up remains 1000 actions; this budget gives it an opportunity to train, but does not equalize gradient counts or establish convergence.

All scores are official local SDK scores, not Kaggle/private scores. These public games were used in development. Runtime rankings are omitted because concurrent machine load differs. The archived 0.0235/0.4904 scores belong to a different, 200-configured-action, seed-5209 protocol; they are not targets for this new protocol.
