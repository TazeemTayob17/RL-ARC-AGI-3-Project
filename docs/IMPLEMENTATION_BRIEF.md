# Implementation brief (Phase 4 handoff)

Derived from `docs/PROJECT_PLAN.md`, Phase 1–3 evidence (`docs/DECISIONS.md`, `docs/PHASE3_HANDOFF.md`, `docs/evidence/`) and the wider-project research context. The decision table is the source of truth; nothing below repeats it. Phases 1–3 are done; this brief covers Phase 4 onward.

## 1. Build now

| decision | current choice | why | revisit when |
|---|---|---|---|
| Layout | `src/{adapter,model,dqn,ppo,run}.py`, `src/dqn_v1/`, `configs/`, `tests/`, `results/`, `kaggle/`; split a file only when readability requires | Existing | A second real alternative forces a split |
| Deep-learning stack | PyTorch, own small cited DQN/PPO; no Gymnasium | No third-party RL library | A library is adopted |
| Action space | 4102 = 6 simple (ACTION1-5, 7) + 4096 clicks; index `0..5` simple, `6 + y*64 + x` click | Verified in Phase 1 | Never (settled) |
| Masks | Rebuilt every step from `available_actions`; empty frame reuses last non-empty mask; `RESET` never learner-selectable | Plan 5.3 | A dev game shows it changing |
| Resets | Adapter issues exactly one RESET after GAME_OVER, never two in a row | Two consecutive resets = full reset (ft09) | Never, unless an experiment declares it |
| Reward | `r = max(0, progress_next - high_water_before)`; progress and high-water stored per transition | Correct under either reset behaviour | Probe on more dev games contradicts ft09 |
| Intrinsic reward | Separate `intrinsic` field, never mixed into `reward`; scale and reset rule recorded | Plan 4.3 | I1 results |
| Budget counter | Research budget counts scorer-rule actions; raw env commands also logged | Matches official accounting | Kaggle counts differently |
| Terminals | GAME_OVER and WIN terminal; cutoffs non-terminal | Plan 4.4 | Never |
| Assignment baselines | DQN = code `dqn`/`V1` (selected DQN); PPO = code `ppo`/`V0`; assignment stage in a separate `stage` field | Code labels collide with assignment labels | Never |
| Learner game-blindness | Run controller resolves checkpoint and seed and passes them in; learners never receive the game ID | Anti-cheat rule | Never |
| Deployment checkpoint | One declared checkpoint for unknown games: trained only on dev games (needs teammate training code), fallback one predeclared fold checkpoint, disclosed | V1 currently refuses unknown games | Training code supplied |
| I1 | Count of exact observed-grid hashes per run and level; bonus β/√n at collection time; zero on action/undo loops; β = 0 reproduces baseline | Group choice; plan 7.2 | Phase 4 results |
| I2 | A (event-terminated repeat) or B (cross-level effect beliefs) by the predeclared rule; C deferred | Plan 7.2 shortlist | After I1 grid |
| Scoring | Official SDK scorecard per run, saved with provenance | Phase 1 | Kaggle baselines differ |
| Parallelism | One process per run; scheduler-agnostic launcher | Cluster details unknown | Cluster details known |

## 2. Boundaries

```python
# adapter.py: the only code that touches the SDK
class ArcSession:
    def start_run(game, seed, budget) -> Observation
    def step(action_index: int) -> Transition
# learner contract (also wrapped by kaggle/learner_agent.py)
class Learner:
    def act(obs: Observation) -> int
    def observe(t: Transition) -> None
```

- Adapter knows nothing about models or algorithms. Learners know nothing about the SDK **or the game identity**.
- Intrinsic reward and options/beliefs live inside learners, behind `observe()`, so they work in both the local run path and the Kaggle bridge path.
- `run.py` owns budget, stopping, checkpoints, logging, scoring, checkpoint/seed resolution; it never touches model internals.

## 3. Experiment identity

Every run record and checkpoint stores: schema version, run ID, algorithm, code version, **assignment `stage`**, architecture ID and full config, algorithm config, **intrinsic-reward config**, observation/history contract, action representation, auxiliary/memory config, game, seed, budget, split hash, code commit, package versions, **checkpoint path and SHA-256**, and a stable config hash. Phase 3 config hashes must not change: new settings appear only in configs that use them. Per run: runtime, peak memory, both action counters, scorecard JSON, progress curve, **module-activation counts**.

## 4. Order of work

### Phase 4 — Improvement 1
1. **Anti-cheat guard test.** Pass: scans agent code for `environment_files`, game-source access, baseline-count access and game names; passes on clean code, fails on a planted violation.
2. **Remove game ID from the DQN agent.** Pass: checkpoint and seed resolved in the run controller; Phase 3 action traces reproduce exactly; an unknown game id runs with the declared deployment checkpoint.
3. **Kaggle baseline submission.** Pass: valid submission evaluated; ID, score, timestamp recorded.
4. **Novelty counter + I1 in DQN and PPO.** Pass: β = 0 reproduces baseline; no bonus on undo loops; counts reset between runs.
5. **I1 activation smoke** (2 games, 1 seed). Pass: bonus nonzero, reaches TD targets/advantages, Q-values or logits move after updates.
6. **I1 grid**: 17 dev games × 3 seeds × 1,500 actions per algorithm. Pass: compared with Phase 3 records by paired score difference and game-cluster bootstrap 95% interval; inconclusive is reported as such.
7. **Budget-sensitivity diagnostic**: baseline vs learning-off at a larger budget on 5 predeclared dev games × 3 seeds × both algorithms. Pass: result recorded; later matched comparisons rerun at that budget if learning helps.
8. **Bug-versus-research-failure labels.** Pass: every negative result labelled (defect · inactive · budget · calibration · genuine).
9. **Exposure update and I2 choice.** Pass: `docs/EXPOSURE.md` includes cluster-research exposure; I2 chosen by the predeclared rule (fewer than one third of I1 runs reach Level 1 → A; effective actions rediscovered after level changes → B; unclear → A).

### Phase 5 — Improvement 2
10. Implement A or B per its experiment card with synthetic tests. Pass: mechanism fires, V2 and A2 runs on the same grid, ablations (beliefs reset/shuffled for B) lose the gain if it is real.

### Later
11. Phase 6 reserved-game evaluation, Phase 7 submissions, Phase 8 report and package — see the plan's §9.

**Phase 4 outcome.** Steps 1, 2, 4–9 are done. I1 grid: inconclusive for both algorithms (`docs/PHASE4_RESULTS.md`). Budget diagnostic: PPO no effect; DQN weak positive at 5,000 actions (`docs/PHASE4_BUDGET_DIAGNOSTIC.md`); 1,500 actions stays the protocol. I2 = A by the predeclared rule. Step 3 (Kaggle baseline submission) needs the user's account.

## 5. Waiting on evidence

- Deployment checkpoint: teammate's encoder training code.
- Budget for later stages: decided, 1,500 actions (DQN 5,000-action signal is weak).
- I2: decided, A.
- Cluster timing: Wits cluster details (GPU, CPU, Python, scheduler).
- Kaggle limits and baselines: a scored Kaggle submission.
- Wider-project code (native-Q, B0–B2, event-repeat): not available; nothing is assumed.

**Next decisions**
1. Review Phase 4 changes (nothing is committed).
2. Make the Kaggle baseline submission and record ID, score, timestamp.
3. Ask the teammate for the encoder training code (deployment checkpoint).
4. Approve Phase 5: implement I2 = A (event-terminated repeat option).
5. Request approval before changing I2 away from the rule's outcome.
