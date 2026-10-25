# Phase 5d: DQN without the pretrained prior (criteria written before any run)

**Question.** The selected DQN carries a frozen encoder and action-effect head pretrained offline on public-game data (cross-validation folds that include reserved games); PPO has no such prior. How much of the DQN's behaviour and score comes from that prior, and is a prior-free DQN a usable deployment option? This also bears on two plan issues: the DQN-versus-PPO confound (W9) and the deployment checkpoint, which for unknown games falls back to one fold checkpoint that has seen public-game data.

**Intervention (ablation, not an improvement).** `prior_mode: "random"` replaces the pretrained checkpoint with the same frozen architecture at random weights (one fixed seed, `RANDOM_PRIOR_SEED` in `src/dqn_v1/prior.py`, identical for every game and run). Its action-effect logits are zeroed, so the structured explorer's logit term gives no preference (its visit-count and salience terms are unchanged). Everything else (Q-network, replay, I1, I3, exploration) is unchanged. No checkpoint, game id or outside data is read in this mode. The learner never sees a game id; the seed offset stays controller-side as in Phases 3 to 5c.

**Conditions (DQN only, 17 dev games x 3 seeds x 1,500 scorer-counted actions):**
- **V3np** = V3 (baseline + I1 + I3) with the random encoder. Matched reference: Phase 5b V3 records.
- **V0np** = baseline with the random encoder. Matched reference: Phase 3 DQN baseline records.

**Predeclared comparisons** (paired by game and seed; paired mean official local score difference with game-cluster bootstrap 95% interval; interval crossing zero = inconclusive):
1. **Primary: V3np vs V3.** What the pretrained prior adds at the V3 stage.
2. V0np vs V0. What the prior adds to the plain baseline.
3. V3np vs V0np. Whether I1 + I3 help without the prior.
4. V3np vs PPO V3 (Phase 5b). Whether a prior-free DQN differs from PPO on the same I1 + I3 stage (the confound).
5. Secondary, reported whatever the primary says: levels, Level-1 runs, games with at least one level, median first-progress action, frame-change rate, games regressed.

**Reading.** If V3np is not below V3 (interval crossing zero or above), the prior is not needed for the dev results and the prior-free DQN is preferred for deployment (no exposure to public-game pretraining data, no checkpoint choice). If V3np is below zero, the prior is doing real work and its exposure must be disclosed next to any reserved-game result. The result is reported either way. Constants are not tuned; this grid is not repeated.

**Activation precondition.** For every run, diagnostics must show the random prior in use (`prior_checkpoint` starts with "none") and the actions must diverge from the matched pretrained run. A run that fails this is not evidence.

Local SDK scores on exposed public games, not Kaggle results.
