# Implementation brief (Phase 2 handoff)

Derived from `docs/PROJECT_PLAN.md` and Phase 1 evidence (`docs/DECISIONS.md`, `docs/evidence/`). The decision table is the source of truth; nothing below repeats it.

## 1. Build now

| decision | current choice | why | revisit when |
|---|---|---|---|
| Layout | Plan 5.1 files: `src/{adapter,model,dqn,ppo,run}.py`, `configs/`, `tests/`, `results/`, `kaggle/`; split a file only when readability requires | Plan default | A second real alternative forces a split |
| Deep-learning stack | PyTorch, own small cited DQN/PPO; **no Gymnasium** | No third-party RL library is used, so the adapter is not required; fewer layers | A library such as SB3 is adopted |
| torch pin | CPU build now; pin GPU/CUDA build once cluster details arrive | Cluster unknown | Cluster details known |
| Action space | 4102 = 6 simple (ACTION1-5, 7) + 4096 clicks; index `0..5` simple, `6 + y*64 + x` click; `x` = column, `y` = row, origin top-left | Verified in Phase 1 | Never (settled) |
| Masks | Rebuilt every step from the latest frame's `available_actions`; on an empty frame reuse the last non-empty mask; `RESET` is never learner-selectable | Plan 5.3; `available_actions` not verified constant | A dev game shows it changing |
| Resets | Adapter issues exactly one RESET after GAME_OVER, never two in a row; `start_run()` separate from level reset | Phase 1: two consecutive resets = full reset, one = restart level | Never issue a full reset unless an experiment declares it |
| Reward | `r = max(0, progress_next - high_water_before)`, then update high-water; store progress and high-water before/after in each transition | Correct under either reset behaviour, so nothing is guessed | Reset probe on more dev games contradicts ft09 |
| Budget counter | Research budget B counts scorer-rule actions (steps + non-first resets; steps in GAME_OVER never sent); log raw env commands too. Direct SDK use: first reset free. Framework loop: every command counted, no initial RESET is sent, budget enforced by `MAX_ACTIONS` | Matches official accounting; verified against the scorecard in both paths (tests) | Kaggle counts differently |
| Terminals | GAME_OVER and WIN terminal; rollout/time cutoffs non-terminal; never bootstrap from a post-reset frame | Plan 4.4 | Never |
| Observation contract | `Observation`: current frame `int8[64,64]`, previous frame, mask, previous action, progress, high-water (no validity flag: the adapter never sends a step during GAME_OVER, so no empty frame is ever reused). History length is **not** in the contract; V0 uses 2 | V0 shapes must not leak | Architecture search reports |
| Representation | Plain classes for encoder / temporal / action representation / heads; chosen by config at run start; auxiliary head and memory are absent-by-default (no-op) | Plan 5.2; modularity only where an experiment needs it | A second real encoder exists |
| V0 model | Categorical-colour CNN, 2-frame history, pooled global context broadcast into the click head | Plan V0 scaffold, not a research conclusion | Architecture validation completes |
| DQN | Double DQN, type-balanced epsilon, ordinary replay arrays, plan 5.4 settings as config | Plan | Calibration |
| PPO | One live env, sequential rollouts, factorised type/location policy, plan 5.5 settings as config | Plan | Calibration |
| Scoring | Official SDK scorecard per run, saved with provenance; never substitute completion counts | Phase 1: local scorer works, all 25 games have baselines | Kaggle baselines differ |
| Parallelism | One process per run, no in-process vector envs; scheduler-agnostic launcher | Cluster unknown; keeps semantics backend-independent | Cluster details known |

## 2. Boundaries

```python
# adapter.py: the only code that touches the SDK
class ArcSession:
    def start_run(game, seed, budget) -> Observation
    def step(action_index: int) -> Transition   # handles reset-on-GAME_OVER, budget, reward, masks
# learner contract (also what kaggle/my_agent.py wraps)
class Learner:
    def act(obs: Observation) -> int
    def observe(t: Transition) -> None          # called before the next act()
# model.py
class Representation:   # encoder + temporal; returns features + declared views
    def forward(obs_batch) -> RepresentationOutput
```

- Adapter knows nothing about CNNs, history length or algorithms. Learners know nothing about the SDK.
- DQN/PPO consume `RepresentationOutput` plus their own head; Bellman and PPO/GAE code do not change when the encoder does.
- `run.py` owns budget, stopping, checkpoints, logging and scoring; it never touches model internals.

## 3. Experiment identity

Every run record and checkpoint stores: schema version, run ID, algorithm + version label (V0/V1/V2/A2), architecture ID and full architecture config, algorithm config, observation/history contract, action representation, auxiliary/memory config, game, seed, budget, split hash (`configs/splits.json`), code commit, package versions (`results/env_pins.json`), and a stable config hash. Plus per run: runtime, peak memory, both action counters, scorecard JSON, progress curve.

## 4. Order of work

1. **Adapter + action mapping + masks.** Pass: index round-trip and real click placement through the SDK; masks correct on click-only, keyboard-only and mixed games; no illegal action selected.
2. **Run controller + both random policies.** Pass: reproduces Phase 1's random results; budget stops exactly; no reset after exhaustion.
3. **Reward, progress, boundaries.** Pass: diagnostic ft09 replay gives exactly one reward; no back-to-back resets; history cleared at reset while counters persist.
4. **Config + identity + checkpoint reload.** Pass: reloading reconstructs the exact architecture, not just weights.
5. **V0 model + interfaces.** Pass: a dummy alternative encoder swaps in through config with no other file changed.
6. **DQN.** Pass: online/target masking, terminal skip, finite loss with masked actions and terminals.
7. **PPO.** Pass: joint log-prob agrees with the direct computation before any update; GAE stops at boundaries.
8. **Synthetic tasks** (navigation, click, distant-cue). Pass: both algorithms improve; an updates-disabled copy does not.
9. **Kaggle lifecycle.** Pass: through the framework's `Agent.main()` loop, transitions are built from consecutive `frames` and weights persist and update.
10. **Calibration** (needs confirmed split and hardware). Pass: run inventory with measured time per 10,000 actions, peak RAM, update counts.

## 5. Waiting on evidence

- Final architecture: architecture-search results, then matched reward-based validation of the finalists.
- Reset behaviour beyond ft09: one more probe on other dev games; changes nothing unless it contradicts.
- Kaggle runtime, action limit, reset rules: read from Kaggle; sets the competition-regime config.
- DQN/PPO eligibility and local-score evidence: tutor.
- Kaggle baselines equal to the SDK's: a scored Kaggle run.
- Budget B, hyperparameters, exploration schedule: step 10.
- I1/I2: Phase 3 diagnostics.

**Next decisions**
1. Confirm `configs/splits.json` before step 10.
2. Pin torch once cluster hardware is known.
3. Run the reset probe on 2-3 more dev games.
4. Choose B from step 10's measurements.
5. Pick I1 from Phase 3 diagnostics.
