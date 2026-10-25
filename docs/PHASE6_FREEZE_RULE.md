# Phase 6: freeze rule and final-evaluation criteria (written before any reserved-game run)

## What is frozen, and why
Final versions, using development evidence only (Phases 3 to 5d). Nothing here is tuned on the reserved games.

| Assignment label | DQN (code `dqn`/`V1`) | PPO (code `ppo`/`V0`) | Dev evidence |
|---|---|---|---|
| V0 | selected DQN baseline | PPO baseline | Phase 3 |
| V1 = V0 + I1 | + count-based novelty (beta 0.01) | same | Phase 4: inconclusive for both |
| V2 = V0 + I1 + I3 | + object-level clicks with effect memory | same | Phase 5b: PPO +0.194 over V1 (interval just crosses zero), DQN levels 12 to 18 and Level-1 runs 12/51 to 16/51, DQN V3 above A3 |
| A2 = V0 + I3 | I3 without I1 | same | ablation: V2 minus I1 |
| V2np (DQN only, deployment variant) | V2 with a random frozen encoder, no pretrained prior | n/a | Phase 5d: not below V2 on the primary comparison |

- **Second improvement.** The assignment's I2 slot is filled by I3. The event-terminated repeat option (the plan's I2) and the frontier option (I4) were evaluated on development games and not promoted (docs/PHASE5_RESULTS.md, docs/PHASE5C_VERDICT.md). I5 was skipped because its gate was not met (docs/PHASE5C_VERDICT.md). They stay in the report as evaluated stages with their results; they are not rerun on reserved games.
- **Choosing I3 is a judgement on development evidence, not a significant win.** The dev intervals for V2 vs V1 cross zero for both algorithms. The report must say so.
- **Nominated Kaggle agents:** PPO V2; DQN V2np. V2np is nominated because it needs no pretrained checkpoint (no pretraining on public-game data that includes reserved games, no game-to-checkpoint mapping) and was not below V2 on development games. Its Level-1 count was 3 runs lower (13/51 vs 16/51); that is stated. If the reserved evaluation shows V2np below V2 with the whole interval below zero, the nomination is not changed silently: it is reported and the decision returned to the team.

## Final evaluation protocol
- **Games:** the 8 reserved games in `configs/splits.json` (bp35, cn04, lp85, s5i5, sb26, sk48, tu93, wa30). "Reserved" means reserved from further tuning; all 25 public games were exposed to earlier development (docs/EXPOSURE.md), and part of the repeat-option evidence came from lp85. Results are not unseen-game evidence; only Kaggle's semi-private and private evaluation is.
- **Seeds:** 0 to 4 (five). **Budget:** 1,500 scorer-counted actions. **Environment seed:** 0. Same code, parameters and constants as the development runs. No retuning and no repeat grid.
- **Conditions per game and seed:** DQN V0, V1, V2, A2, V2np; PPO V0, V1, V2, A2. Total 8 x 5 x 9 = 360 runs.
- **Primary comparisons (per algorithm):** V2 vs V0, V1 vs V0, V2 vs V1, V2 vs A2; DQN also V2np vs V2. Paired mean official local score difference with a game-cluster bootstrap 95% interval (8 clusters, so intervals are wide and stated as such). An interval crossing zero is inconclusive.
- **Secondary (reported whatever the primary says):** levels completed, games with at least one level, Level-1 runs, median first-progress action, frame-change rate, games regressed, wall time. Because scores are dominated by few games, per-game scores are tabulated.
- **Cross-algorithm table:** DQN vs PPO at matched stages, with the stated differences (DQN uses a frozen pretrained encoder at V0 to V2; V2np removes it).
- **Validity gates before reading any result:** all 360 records present; scorer count equals SDK total in every run; the agent guard test passes; runtime guard shows zero violations; activation check (I1 bonus nonzero in V1/V2/V2np, object candidates offered in V2/A2/V2np, random prior in use in V2np). A run failing a gate is not evidence and halts the evaluation for review.
- **Halt rule:** any sign of an agent reading game internals, game identity, or files outside its observation stops the grid and is reported before anything else.

Local SDK scores, not Kaggle results.
