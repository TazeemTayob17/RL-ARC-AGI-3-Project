# Phase 3 integration and verification

## Integration
Imported the teammate's DQN patch and five frozen public-fold checkpoints. Original DQN, PPO, adapter and bridge implementations remain as supplied. Selected DQN is the assignment starting baseline despite its implementation label V1. No Phase 4 reward changes are included.

## Experiments and validation
All 306 matched grid records and 15 additional original-DQN records completed. Scorer counters were checked against SDK totals, config hashes validated, normal learners performed gradients and disabled controls performed none. Focused integration checks passed. The original full-suite distant-cue capacity check failed earlier and is not claimed repaired.

The initial concurrent launch encountered an SDK cache-load error. The final grid grouped one game per worker, with its three seeds sequentially; all final groups succeeded. No failed result is presented as successful. Source manifests and raw records are preserved. Machine-load timings are descriptive, not fair algorithm runtime rankings.

## Historical evidence
The teammate later supplied V0_V1_5GAME_RESULTS.tar.gz. Extracted scorecards and README are preserved under results/teammate_five_game_comparison. Aggregate scores, actions, levels, game IDs and human baseline arrays passed consistency checks. This was archive verification, not an exact rerun. See TEAMMATE_COMPARISON_VERIFICATION.md.

## Interpretation and decisions
Learning-control differences do not establish an online-update benefit. Most runs found no progress. The longer original/selected DQN comparison is driven entirely by sp80 and uses a different protocol from the teammate's test.

The group accepted the selected DQN/current PPO, split and budget and reported tutor approval. It delegated first-improvement selection; a count-based exploration reward is the next experiment. This branch includes the selection decision, not Phase 4 implementation or results.

See PHASE3_HANDOFF.md for the consolidated group handoff. All public games have development exposure; selected DQN unknown-game deployment and competition packaging remain later work.
