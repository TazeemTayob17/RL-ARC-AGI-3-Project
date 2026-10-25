# Phase 5b: I3 success and failure criteria (written before any grid run)

Intervention: object-level click actions with effect memory (`src/objects.py`, `src/dqn_v1/object_actions.py`, `src/ppo.py`), for DQN (code V1) and PPO (code V0). Constants fixed once, not tuned: tile size 16 above 512 cells, candidate cap 96, retry probability 0.05, dead signature after 3 tries.

Grid: 17 development games x 3 seeds x 1,500 scorer-counted actions; V3 (baseline + I1 + I3), A3 (baseline + I3), V3nm (V3 without effect memory), per algorithm. Matched references: Phase 4 I1 records (V1) and Phase 3 baseline records (V0).

Predeclared reading, per algorithm, primary comparison V3 vs V1 and A3 vs V0:
- **Primary:** paired mean official local score difference with a game-cluster bootstrap 95% interval. Interval above zero = improvement; below zero = harm; crossing zero = inconclusive (still a valid evaluated stage).
- **Secondary (reported whatever the primary says):** levels completed; Level-1 (progress) runs; median action of first progress; frame-change rate (effective-action rate); games regressed; wall time.
- **Attribution:** V3 vs V3nm isolates the effect memory; V3nm vs V1 isolates object actions; V3 vs A3 isolates I1 on top of I3.
- **Activation (precondition):** `scripts/check_phase5b_activation.py` must report object candidates offered, the memory recording outcomes, and actions diverging from the reference in click games. A run that offers no click candidates is not evidence about I3.
- **Gate for I5:** at least one third of runs reaching Level 1 in the best I3 condition; otherwise I5 is skipped and the reason recorded.
- **Reserved games are not used.** All scores are local SDK scores on exposed public games, not Kaggle results.
