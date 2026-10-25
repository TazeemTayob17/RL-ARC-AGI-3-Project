# Phase 5c: I4 success and failure criteria (written before any grid run)

Intervention: a frontier-directed exploration option (`src/frontier.py`), added to the choice set of both learners: the selected DQN (code V1) and PPO (code V0).

**Mechanism.** The agent keeps a per-run graph of states it has observed (a node is an exact grid at a level; an edge is a primitive action that led from one node to another; the candidate actions of a node are its legal simple actions plus one click per object, as in I3). The option "go to frontier" walks the shortest known path (at most 40 edges) to the nearest node that still has an untried action, then takes one untried action there, chosen uniformly by the wrapper's own RNG. The option ends on done, deviation (a step did not reach the expected node), level-up, terminal, budget or illegal action. Every primitive step is a real environment step charged to the budget. The learner sees the whole option as one semi-Markov transition (first observation, discounted reward sum, last next-observation, k steps), exactly as for the I2 repeat option. DQN scores the option as one extra row of its Q-network choice set, learned like any other action. PPO gets one extra action type with its own logit and policy-gradient credit.

**Constants fixed once, not tuned:** maximum walk length 40 edges; DQN exploration draws take the option with probability 0.25 when it is legal; untried action chosen uniformly; node key = exact grid plus level; candidate actions as in I3 (cap 96 objects).

**Anti-cheat.** The graph is built only from the Observation and Transition fields the learner is shown, is created at run start and is never saved, shared or reloaded. No game id, no `environment_files/`, no outside data, no per-game constants.

Grid: 17 development games x 3 seeds x 1,500 scorer-counted actions, per algorithm:
- **V4** = V3 + I4 (V3 = baseline + I1 + I3). The option is offered to the learner.
- **A4** = baseline + I4 only (no I1, no I3; fixed 4,096-pixel click space).
- **V4f** = V3 + forced option: the option is taken whenever available. A training-free graph-explorer reference policy, not a candidate agent. The learner acts only when the graph is exhausted and is not shown the option's transitions.
- "Option never" is the Phase 5b V3 record (same games, seeds, budget). With `frontier_mode: off` the wrapped learner is bit-for-bit the unwrapped one (tested), so the V3 records are a valid control.

Matched references: Phase 5b V3 records (for V4 and V4f), Phase 3 baseline records (for A4), Phase 4 I1 records (for the cumulative comparison).

Predeclared reading, per algorithm, primary comparison **V4 vs V3** and **A4 vs V0**:
- **Primary:** paired mean official local score difference with a game-cluster bootstrap 95% interval. Interval above zero = improvement; below zero = harm; crossing zero = inconclusive (still a valid evaluated stage).
- **Secondary (reported whatever the primary says):** levels completed; Level-1 (progress) runs; median action of first progress; frame-change rate; games regressed; wall time. Because the score is dominated by two games, the Level-1 run count and the number of games with at least one level are read next to the score.
- **Attribution:** V4 vs V4f isolates whether learning when to use the option matters (learned vs always); V4f vs V3 shows what a training-free graph explorer does on its own; V4 vs A4 shows what I1 + I3 add on top of the option; V4 vs V1 is the cumulative effect of I3 + I4 over the I1 stage.
- **Mechanism metrics (reported, not success criteria):** how often the option is offered and chosen; mean option length; share of walks that reached the expected node (graph accuracy); new states per 100 steps inside and outside options; stop reasons; nodes with untried actions left.
- **Activation (precondition):** `scripts/check_phase5c_activation.py` must show, for every run, that the frontier wrapper was active, that the option was offered, and that for V4/A4 the option was chosen in the large majority of runs and the actions diverge from the matched reference. A run in which the option was never offered is not evidence about I4.
- **Gate for I5:** at least one third of runs reaching Level 1 in the best condition of I3 or I4; otherwise I5 is skipped and the reason recorded. (After I3 the best condition reached 16/51 = 31.4%, which is below the gate.)
- **Reserved games are not used.** All scores are local SDK scores on exposed public games, not Kaggle results.

Known risks, stated before running: (1) hidden state or non-deterministic transitions make walks deviate, so the option will often end early on some games; (2) games whose win needs a specific order of actions are not helped by trying every action once; (3) with about 100 candidate actions per node, exhaustive coverage of even one node costs a large share of a 1,500-action budget; (4) tn36 and r11l, which dominate the mean score, may regress if the option displaces the actions that find their first level.
