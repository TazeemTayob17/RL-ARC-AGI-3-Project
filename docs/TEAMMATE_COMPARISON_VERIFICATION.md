# Archived five-game DQN comparison verification

The raw JSON scorecards and README confirm the reported figures. This is an archive consistency audit, not an independent rerun.

Games: sp80, r11l, cd82, ls20, tn36. Both agents: seed 5209, configured MAX_ACTIONS 200. The framework recorded 201 scorer actions per game, 1005 total each. No game was fully completed.

| Implementation | SDK score | Levels completed | Total scored actions |
|---|---:|---:|---:|
| V0 | 0.02351798 | 1 | 1005 |
| V1 | 0.49039188 | 3 | 1005 |

## Per-game evidence

| Game | V0 score | V1 score | V0 levels | V1 levels |
|---|---:|---:|---:|---:|
| sp80 | 0.00000000 | 0.00000000 | 0 | 0 |
| r11l | 0.11758989 | 1.59609550 | 1 | 1 |
| cd82 | 0.00000000 | 0.50435075 | 0 | 1 |
| ls20 | 0.00000000 | 0.00000000 | 0 | 0 |
| tn36 | 0.00000000 | 0.35151315 | 0 | 1 |

## Checks and interpretation

Aggregate JSON values match the nested scorecard. Per-game scores average to the overall score. Actions and levels sum correctly. Both agents have identical versioned game IDs and human baseline action arrays. Original Slurm job ID in README: 60185.

V1 completed additional levels in cd82 and tn36, and completed the first r11l level in 38 actions versus 140 for V0. sp80 and ls20 had no completed levels for either agent.

The supplied implementation gives original V0 a 1000-action warm-up. At 201 scored commands per game, it would make no training updates with those defaults. Selected V1 uses a 16-action warm-up and pretrained features. The archive lacks resolved configs, code hashes, action traces and gradient counters, so actual update counts cannot be established from these files alone. This supports a short-budget whole-system comparison, not a claim that online learning alone caused the gain. One seed does not establish a reliable multi-seed advantage. These are local SDK scores, not unseen Kaggle results.

## Follow-up evidence

The completed 1500-action, three-seed comparison supplies separate evidence because both agents have an opportunity to train and saved records include gradient counters. Its five-game selection was declared before this archive arrived and is retained; it is a separate experiment. The full 306-run Phase 3 grid also completed successfully.
