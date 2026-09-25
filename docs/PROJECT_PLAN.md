# ARC-AGI-3 RL Assignment — Implementation Planning Prompt

**Course:** COMS4061A/COMS7071A, 2026  
**Updated:** 22 September 2026  
**Course deadline:** 27 October 2026, 23:59 SAST  
**Internal submission target:** 24 October 2026, for both Kaggle tracks  
**Purpose:** Source-of-truth planning prompt for turning the assignment requirements and current evidence into a concise implementation brief.

> **Best-use note for the planning pass:** run this in a fresh Opus 5 context with only this document and directly relevant source files. If the interface offers an effort setting, use a high reasoning effort. Do not prepend the previous brainstorming conversation; this document is intended to be the clean source of truth for the planning task.

This document preserves the original plan's two-algorithm investigation and evidence-led improvement stages. It corrects submission gaps and simplifies the initial implementation. It is a plan, not a claim that experiments or fixes have already been completed.

### Implementation-planning instruction

When this document is given to a planning model, treat it as the team's current research specification. Preserve existing decisions unless there is a concrete correctness, scientific-validity, or implementation conflict. Do **not** invent experiment results or promote provisional choices to final decisions.

The goal is not to produce the most sophisticated-looking design or the most code. The goal is to reduce irreversible work: build the parts we already understand, keep uncertain research choices replaceable, and make every reported claim traceable to evidence. Progress means a smaller, better-tested decision surface and faster future experiments, not simply more implementation.

The final neural architecture is intentionally **not frozen yet**. A separate controlled architecture-selection experiment is currently comparing observation encoders, temporal/state representations, action representations/heads, auxiliary objectives, and memory/state-retrieval representations across games, folds, and seeds. Its results will be supplied later.

Therefore:

- plan the software so representation components can be replaced independently without rewriting the environment adapter, DQN/PPO algorithms, experiment controller, logging, scoring, or evaluation pipeline;
- keep the current small categorical CNN and current action-head design as a **provisional V0/smoke-test baseline**, not as the selected final architecture;
- preserve the complete legal action space and dynamic masks regardless of the internal representation;
- avoid speculative architecture additions merely because they are fashionable or plausible;
- make architecture selection configuration-driven and record enough information in every run/checkpoint to reconstruct the exact instantiated model;
- when architecture-search results arrive, support matched reward-based validation of several credible finalists before freezing the RL base architecture;
- prefer the minimum modularity required for these real experiments rather than building a general-purpose RL framework.

The planning step should produce an implementation plan, interfaces, tests, acceptance checks, and implementation order. It should **not** choose a final neural architecture or begin evidence-led I1/I2 decisions before the relevant evidence exists.

### How to read this plan

Not every choice in this document has the same status. Use these plain-English labels:

- **Settled** — required by the assignment, verified against the interface, or supported by completed evidence. Keep it unless a real correctness problem or new evidence forces a change.
- **Working baseline** — good enough to let the team build and test now, but deliberately easy to replace. It is not a research conclusion.
- **Waiting for evidence** — intentionally undecided. Do not guess just to make the plan look complete.
- **Possible later experiment** — an idea worth considering only if diagnostics give us a reason to test it.

For this project, the requirement to compare one course-covered and one non-course learning algorithm, legal action-space preservation, assignment requirements, and interface behaviour already verified against the pinned environment are **Settled**.

The specific DQN/PPO pairing is a **Working baseline** until course eligibility is confirmed. If the tutor rejects either choice, replace only the ineligible algorithm while preserving the surrounding experiment design.

Reset, progress, and accounting behaviour that still depends on interface checks is **Waiting for evidence**. The current small CNN, short-history representation, and action-head design are a **Working baseline**. The final representation architecture is **Waiting for evidence**. Recurrent PPO, multi-step DQN, intrinsic reward, RND, and similar ideas are **Possible later experiments** until diagnostics justify them.

**Plain-English rule:** if a choice changes what the experiment means, wait for evidence. If it is ordinary reversible engineering plumbing, choose the simplest sensible option and keep moving.

## 1. Evidence, scope, and requirements

This plan was derived from the assignment brief and earlier project planning. Some earlier notes, probe logs, code, or experiment records may not be present in the planning context. Do not invent their contents or block ordinary implementation because they are missing. Treat a claim as verified only when the relevant evidence is actually available or reproduced. Machine-specific paths, installed packages, throughput, and hardware observations are local facts, not assumptions about every team machine.

### Required outcomes

1. Implement and evaluate one course algorithm and one non-course algorithm: provisionally DQN and PPO. Confirm eligibility with a tutor if uncertain.
2. Evaluate at least three versions of each: baseline, first substantive improvement, then a second substantive improvement added to the first.
3. Choose each improvement from evidence produced by the previous stage. A hyperparameter adjustment alone does not count as one of the required improvements.
4. Compare final agents, including ablations, adaptation/generalisation evidence, interaction efficiency, compute, variability, and success/failure cases. Because all public games have already been exposed during architecture development, public reserved subsets must be described as reserved from further tuning rather than untouched; true unseen evidence comes from competition evaluation.
5. Report the official score for each of the six required versions, identifying the evaluation set and scoring version. Keep shaped reward and diagnostic metrics separate.
6. Use at least three independent seeds for stochastic experiments. The brief allows a justified resource limitation, but plan for three and reduce optional scope before reducing replication.
7. Submit working code and the Paper Track write-up by the Moodle deadline. Include the Paper Track link and saved text, code submission evidence, matching notebook, report, and contribution statement in the ZIP.

The investigation is the main assessment objective. A valid zero-score submission remains eligible for the class leaderboard, but zero scores do not remove the need for functioning learners and sound analysis.

### Rules retained

- Use only observations and metadata permitted by the evaluation interface. Never inspect hidden game state or game implementation source to derive a solution.
- No game-specific action scripts or hidden-solution reuse. Use shared algorithm configurations rather than hand-tuning individual games.
- Disclose all development exposure. Do not claim untouched public generalisation after using public games during architecture development.
- Match interaction budgets and document compute budgets, training cost, resets, stopping rules, and resource differences.
- Preserve official ARC scoring for reported results; shaped/internal rewards remain training signals, not the official score.
- Kaggle submission notebooks must fit the competition execution constraints: CPU/GPU runtime at most 9 hours and internet disabled.
- Publicly available external data and pretrained models may be used when competition rules and licenses permit; privately shared competition code/data outside the team is not allowed.
- Record unsuccessful experiments and distinguish reused code, pretrained models, and prior work from the team's own contribution.
- Declare AI assistance and ensure the team can explain its submitted code and claims.

## 2. What changes from the original plan

| Issue | Updated decision | Implementation and verification |
|---|---|---|
| Paper Track scheduled after Moodle | Submit both tracks by 24 October internally | Add both submissions and their evidence to the final checklist; verify successful submission, not just notebook execution |
| Official scores not assured for all versions | Six-version scoring matrix is mandatory | Establish official local scoring or schedule Kaggle evaluations before the large experiment grid |
| Reward depends on unobserved historical progress | Verify reset semantics; expose progress context if using high-water reward | Store current progress and high-water mark with each transition; test replay consistency |
| PPO assumes eight copies of an unseen game | Use one live environment in the baseline | Collect sequential rollouts through the same agent lifecycle used at submission |
| Flat sampling overwhelmingly favours clicks | Balance exploration by action type | DQN samples type then coordinates; PPO uses a type policy and conditional click policy |
| Click predictions in the provisional V0 baseline see only a local patch | Add whole-board context to the V0 click features as a baseline correction | Broadcast a pooled board feature into the provisional spatial head and test a distant-cue task; do not treat pooled global context as a final representation requirement |
| Held-out split follows broad reconnaissance | Audit exposure and freeze split immediately | Restrict later probes to development games and qualify generalisation claims |
| Coverage described as proof of learning | Add an updates-disabled control | Compare behaviour with and without parameter updates under matched conditions |
| Replay/storage complexity introduced too early | Begin with ordinary replay/storage while keeping execution backend-independent | Validate correctness locally, then parallelise independent runs wherever resources allow without changing seeds, budgets, configs, logging, or result interpretation |
| Statistical aggregation underspecified | Show individual runs and define resampling units | Separate seed variability on fixed games from variation across games |
| Neural architecture specified before architecture-selection evidence is complete | Treat the current CNN/history/action-head design as provisional V0 and keep representation modules swappable | Use configuration-driven encoder, temporal, action-head, auxiliary-head, and optional memory interfaces; later validate strong architecture-search finalists under matched RL reward before freezing |

These baseline design corrections are applied before baseline results are frozen. They do not replace the two later evidence-led improvements required for each algorithm.

## 3. First decisions: interface, scoring, and exposure

### 3.1 Recover evidence and pin the environment

Obtain the referenced notes and probe logs from the original author. Record SDK version, game versions, starter commit, Python version, OS, and hardware. Preserve results that genuinely exist; do not recreate documentation saying an unrun check passed.

Use permitted SDK outputs to test on development games:

- Observation dimensions, colour range, frame-list behaviour, and intermediate frames.
- Click coordinate orientation and submission format.
- Available action sets and whether they change during a game.
- `GAME_OVER`, `WIN`, reset behaviour, and progress counters.
- Determinism for the tested version and action sequences. Do not extrapolate a few deterministic probes into a guarantee about every game or future version.
- Official action accounting, including initialisation, resets, no-ops, and unsuccessful attempts.

Current REST documentation distinguishes a current-level restart from a full-game restart, and documents two consecutive resets as a full reset. Verify the pinned local SDK and competition gateway rather than assuming that all reset methods are interchangeable. Record `full_reset` if the interface exposes it.

**Reset probe:** On a development game with a recorded legitimate completion trajectory, reach a later level, take an action, reset once, and log progress and frames. Separately test a second immediate reset and a reset after `GAME_OVER`. Keep the replay sequence exclusively as a diagnostic, not a hard-coded agent policy.

### 3.2 Establish scoring before large experiments

Verify the official scorer available in the pinned SDK/starter and whether it supplies the required human baselines. Use the official implementation when possible rather than rewriting a scoring formula from prose.

For every evaluated version, save score output, scoring version, game set, action accounting, and local/Kaggle provenance. Validate that unsuccessful attempts and reset costs are included as required by that scorer.

If official local scores cannot be obtained, plan Kaggle scoring for all six versions and ask the tutor how to report any local scoring limitation. Record an unavailable score as unavailable; never substitute a completion count and call it the official score. Do not assume Kaggle provides per-seed variability; report local replicated results separately from the official submission score.

Check current entry/merger deadlines, runtime, accelerators, dependency availability, action limits, submission limits, and reset/information-sharing rules directly on Kaggle. Those external details remain pending verification; they do not change the Moodle deadline.

### 3.3 Keep evaluation claims honest

All 25 public games have already been used during architecture development, so there is **no untouched public held-out set**. Do not describe any public subset as unseen or untouched.

Maintain an exposure table: game, metadata inspected, trajectories inspected, random probes, architecture development, training/tuning, and date. From this point onward, a public subset may be reserved from **further** tuning for matched development evaluation, but report it exactly that way and disclose the prior exposure.

Use fixed, predeclared development/evaluation subsets and seeds for each experiment stage. Freeze the candidate set, evaluation protocol, and selection rule before reward-based architecture validation so the team cannot keep changing the comparison after seeing results.

True unseen evidence must come later from the competition's semi-private/private evaluation or another genuinely unexposed source. A substantive fix after viewing any reserved evaluation result must be disclosed and the affected comparisons rerun; do not preserve an "untouched" label after exposure.

## 4. Learning formulation and adapter

### 4.1 Research question

How effectively can the selected course-covered and non-course learning algorithms — currently planned as DQN and PPO, pending eligibility confirmation — learn useful behaviour within an unfamiliar game under a fixed interaction and runtime budget, and which evidence-led changes improve that adaptation?

Model each game as partially observable. The provisional V0 baseline uses a short two-frame history as an approximation to useful temporal context, not as a claim that the underlying state is fully observed or that two frames are sufficient. The canonical observation/history contract must support configurable finite history and/or stateful temporal modules without changing environment semantics or core RL logic. Online learning is the chosen experimental setting. Do not assert that a frozen pretrained policy cannot transfer; transferable pretraining is a possible later experiment.

### 4.2 Agent lifecycle

Use one environment session per run. A run is one algorithm/version, game, seed, and budget. Separate `start_run()` from episode/level reset so resetting the environment cannot accidentally reset the total budget or learned parameters.

| Boundary | Keep | Reset |
|---|---|---|
| Between actions | Network, optimiser, buffer, run counters, exploration memory | Nothing |
| Between levels | Network, optimiser, buffer, run counters | Only history fields explicitly invalidated by the interface; log this choice |
| After GAME_OVER / environment reset | Network, optimiser, buffer, total actions, high-water mark | Current observation history and episode-local state |
| Between games | Only explicitly declared pretrained weights, if that experiment uses them | Optimiser, buffer, exploration memory, counters, other learned weights |
| Between seeds | Same fixed initial pretrained checkpoint if applicable | All run-specific state and randomness |

Use a common small lifecycle such as `act(observation, info)` and `observe(transition)`, or equivalent methods compatible with the starter. The submission adapter must deliver the previous action's outcome to the learner before requesting the next action. Smoke-test that weights actually persist and update through the competition callback lifecycle.

Keep this agent lifecycle generic enough that a later competition-oriented policy can reuse the same adapter, action legality, logging, scoring, and run controller. DQN and PPO are the required learning implementations, not assumptions that every future competition agent must be DQN/PPO. Do **not** implement a planner, world model, language-model agent, or hybrid policy now merely to preserve this option; the existing small agent contract is the extension seam.

### 4.3 Reward and progress

Log raw progress and the best progress reached independently of the reward.

- If permitted resets preserve completed levels and progress is monotone, use positive progress increments as the sparse baseline reward.
- If full resets can reduce progress and the team retains a first-ever-progress reward, define `r = max(0, progress_next - best_progress_before)` and then update the high-water mark.
- For the latter, include both current progress and high-water mark in learner inputs before and after each action, stored with the transition. This removes the specific hidden reward context; it does not make the whole game fully observable.
- Do not recompute old replay rewards using today's high-water mark.
- Keep any intrinsic reward in a separate field with an explicit scale, schedule, and state-reset rule.

This reward remains a surrogate for official scoring. If full resets require replaying early levels, report that first-ever-progress reward supplies no renewed reward for relearning those levels. Consider alternatives only as declared, evaluated changes.

### 4.4 Boundaries and accounting

The run controller stops at `WIN`, budget exhaustion, or the agreed runtime limit. Do not issue another reset after exhaustion.

Maintain both total environment commands and the scorer's action counter, if available. Use the same research command budget across conditions and the official counting rule for official scores; explain discrepancies.

For baseline updates, treat `GAME_OVER` and `WIN` as terminal. Treat a sampling rollout boundary as nonterminal. An external action/time cutoff may bootstrap from the last valid observation when estimating continuing value; document that convention and never bootstrap from an autoreset observation. Do not propagate GAE across episode resets. If finite-budget value is explicitly modelled later, provide remaining budget and change cutoff semantics consistently.

On reset, initialise the previous frame from the new frame and the previous action with a sentinel. Reuse a last frame for an empty terminal response only with an explicit validity flag. Unexpected empty responses during active play should be logged and handled as interface errors, not silently presented as valid unchanged states.

## 5. Minimal baseline implementation

### 5.1 Project structure

Start with these files; split them only when readability requires it:

```text
src/adapter.py       # Observation, reward, action mapping, boundaries
src/model.py         # Start with representation interfaces + provisional V0; split into small model modules only when a second real alternative exists
src/dqn.py           # DQN training and action selection
src/ppo.py           # PPO training and action selection
src/run.py           # Run controller, random policies, result logging
scripts/figures.py   # Figures and tables from saved results
configs/            # Split and experiment configurations
tests/              # Small interface and learner checks
kaggle/             # Submission adapter/notebook packaging
results/            # Per-run records
docs/               # Evidence, decisions, references, submission checklist
```

Use installed dependencies and standard-library JSON/CSV logging where adequate. Adapt a small, cited reference implementation rather than creating a general RL framework. Keep experiment semantics independent of the execution backend: use simple local execution for correctness checks, while allowing independent runs to be scheduled in parallel on available compute without changing configurations, seeds, budgets, logging, or result interpretation.

### 5.2 Observation and visual model

**Architecture status:** the visual model described below is the provisional V0/smoke-test architecture, not the selected final architecture. A concurrent architecture-selection study is evaluating alternative encoders, temporal representations, action representations, auxiliary objectives, and memory representations. Do not interpret the baseline below as experimentally preferred until those results and subsequent reward-based validation are available.

Input for the provisional baseline: current frame, previous frame, their difference mask, previous action type/coordinates, and relevant permitted progress context. Encode colours categorically rather than treating colour IDs as ordered intensities.

For V0, use a small convolutional encoder that preserves spatial resolution. Pool its feature map to form a whole-board vector. Concatenate permitted scalar/action context with this vector, then broadcast the resulting context across the spatial grid and concatenate it with local features before the click head. Use local-plus-global features to predict click values/logits and pooled context for simple-action/value heads.

Why this remains a useful baseline: four 3x3 stride-1 convolutions alone provide only a 9x9 receptive field per click, so global context lets distant cues influence click predictions. Pooling can still discard spatial relationships. The distant-cue task remains a correctness/diagnostic check, but richer architecture choices should be promoted from controlled architecture evidence rather than ad hoc intuition alone.

Implement only the modularity needed for the planned experiments. The exact Python names may differ, but preserve clear contracts equivalent to:

```text
RepresentationOutput:
    encoded features required by the configured learner
    + optional structured/spatial views
    + declared metadata describing available feature views

ObservationEncoder:
    canonical observation -> encoded representation

TemporalModule:
    encoded representation + declared history -> state representation

ActionRepresentation:
    maps the canonical ARC legal-action space into the structure required
    by the configured learner while preserving a reversible mapping to
    every legal environment action

AlgorithmHead:
    DQN -> legal-action values
    PPO -> legal-action distribution parameters

AuxiliaryHead:
    optional training-only predictions such as action effect/state change

MemoryModule:
    optional read/update interface for persistent interaction memory
```

`AuxiliaryHead` and `MemoryModule` are extension points, not instructions to build unused systems now. A no-op/absent configuration is sufficient until an experiment actually needs them.

DQN and PPO may require algorithm-specific output heads, but they should consume a common representation interface wherever scientifically sensible. The environment adapter must not know whether the active representation uses a CNN, residual encoder, object-based encoder, GRU, attention module, or another tested design.

Architecture selection must be configuration-driven. Adding an evidence-backed architecture should normally require a component implementation, configuration entry, and component-specific tests rather than edits to the environment lifecycle, scoring, experiment logging, or unrelated RL code.

A representation change is **modular enough** when adding a new encoder, temporal module, action representation, or memory mechanism normally requires only:

1. the new component implementation;
2. its configuration;
3. component-specific tests.

It should not require changes to ARC environment semantics, reward computation, lifecycle/reset handling, scorer logic, DQN Bellman-update logic, PPO objective/GAE logic, experiment accounting, or the result schema unless the research question itself explicitly changes those layers. If a genuinely different architecture requires additional shared representation metadata or feature views, extend the relevant data contract additively and backward-compatibly where practical rather than forcing the new design through assumptions inherited from V0.

Modularity must stay cheap in the action/training hot paths. Avoid repeated tensor conversions, duplicated observation preprocessing, unnecessary feature copies, reflection-heavy dispatch, or configuration parsing on every step. Resolve configuration when a run starts and keep the per-action path direct. Abstraction is useful only if it lowers experiment-change cost without materially increasing runtime or memory cost.

Every run and checkpoint must record enough information to reconstruct the exact architecture and experiment identity: schema version, experiment/run ID, architecture ID, full architecture configuration, algorithm configuration, observation/history contract, action representation, auxiliary objectives, memory configuration, seed, code commit/version, and a stable configuration hash.

When the architecture-selection experiment completes, select a small, predeclared shortlist of credible and structurally different finalists using the search results and a stated shortlist rule. Before running reward-based validation, freeze the finalists, games, seeds, interaction/runtime budgets, comparison metrics, and selection rule. Compare the finalists under those matched conditions. Freeze the RL base architecture only after this reward-based validation; proxy metrics such as action-effect prediction are diagnostic evidence, not proof of RL usefulness. If the validation is inconclusive, keep multiple finalists rather than manufacturing a winner.

**Canonical examples**

```text
Bad coupling:
DQNAgent directly constructs OneHotCNN and assumes exactly two frames.

Preferred:
The experiment configuration selects a representation specification.
DQN consumes the resulting state representation through a stable contract.
Changing onehotcnn+stack2 to objecttx+gru4 does not require changing the
Bellman update, environment adapter, scorer, or experiment logger.
```

```text
Bad research inference:
The action-effect predictor has the highest validation AUPRC, therefore it
is the final ARC architecture.

Preferred:
The proxy result makes that architecture a strong finalist. Compare it with
other credible finalists under matched RL reward before freezing the base
architecture.
```

```text
Routine ambiguity:
Two ordinary serialization layouts are both valid and have no effect on
research semantics.

Preferred:
Choose the simpler reversible layout, record the choice, and continue.

Research-critical ambiguity:
It is not yet verified whether progress survives a full reset.

Preferred:
Leave reset-dependent reward semantics undecided until the interface probe resolves
them, because assuming either behaviour would change the meaning of reward.
```

### 5.3 Actions and exploration

Retain the internal mapping of six simple actions plus 4,096 click locations where supported. Build masks from the latest permitted `available_actions`, without assuming they are constant. Do not remove useful pixels using game-specific heuristics.

The internal action representation/head is also provisional and must be swappable behind the common action interface. Alternative vector, spatial, coarse/fine, or other tested representations may structure the same legal action space differently, but no representation may silently prune legal actions. The DQN/PPO action semantics, masks, accounting, and experiment code must remain stable when the action-head implementation changes.

**DQN exploration:** On an epsilon-random step, choose uniformly among available action types. If ACTION6 is chosen, sample a valid coordinate uniformly. On a greedy step, choose the highest legal Q-value over the full action space. Record this as type-balanced epsilon exploration.

**Provisional V0 PPO policy:** Predict logits over available action types and, conditionally on ACTION6, logits over 4,096 locations. This is the baseline factorisation, not a claim that it is the final selected action representation. For clicks:

```text
log p(action) = log p(ACTION6) + log p(location | ACTION6)
```

For simple actions, use their type log-probability only. The joint entropy is type entropy plus `p(ACTION6) * conditional_click_entropy`. Exclude masked types and handle unavailable clicks without computing invalid log-probabilities. Store the action and its old joint log-probability for the PPO ratio.

For the reference policies report both flat-uniform and type-uniform random. Flat-uniform chooses clicks about 99.85% of the time when all 4,102 actions are legal; this is a mixed-action-game bias.

For controlled comparisons, DQN and PPO should use the same base representation configuration by default and share the same environment access and canonical observation/action contracts. This is an experimental-comparability default, not a software constraint: if later evidence shows a representation interacts materially with one algorithm, evaluate that as an explicit, separately labelled experiment rather than forcing artificial architectural equality. Their algorithm-specific output heads may differ as required by DQN versus PPO.

### 5.4 DQN

Use replay, a target network, Huber loss, and epsilon-greedy exploration. If using the Double DQN target, name the baseline **Double DQN** and cite it; confirm course eligibility. Do not count it again as a later improvement.

These are provisional experiment settings, not software-architecture constraints: learning rate 1e-4, gamma 0.99, batch 32, replay capacity up to 50,000, warm-up up to 1,000 actions, one training event per four actions, target updates every 1,000 actions. Keep them configuration-driven and calibrate them before freezing the baseline so warm-up and exploration schedules fit the actual budget.

Store raw observations, context, actions, rewards, boundaries, and next-state masks. Mask online-network next-action selection in Double DQN and evaluate the selected action using the target network. Skip bootstrapping terminal transitions, avoiding `0 * -inf` numerical failures.

Use ordinary replay arrays as the provisional V0 storage implementation. Measure actual memory for the configured observation/history contract rather than assuming the original 400 MB estimate covers everything. If memory pressure justifies indexed frame sharing or another storage scheme, change the storage implementation behind the replay contract and verify overwritten-reference and reset-boundary behaviour; do not let the initial replay layout constrain later representation choices.

### 5.5 PPO

Use one live environment with sequential rollouts, clipped PPO, GAE, and a value head. The following are provisional experiment settings, not software-architecture constraints: 128 actions per rollout, four epochs, four minibatches, clip 0.2, learning rate 2.5e-4, gamma 0.99, lambda 0.95, entropy coefficient 0.01. Keep them configuration-driven and calibrate before freezing.

Store raw observations/context, actions, masks, old log-probabilities, values, rewards, and boundary flags. Reuse the collection-time mask during updates. Process partial final rollouts safely. Record both rollout-update events and gradient steps; they are different quantities.

Do not add cloned hidden-game sessions unless the interface explicitly supports them and the research protocol defines their reward, budget, stopping, and scoring semantics. Any multi-session result must be labelled separately from single-session adaptation.

## 6. Verification and calibration

### Interface checks

- Action index/coordinate round-trip and actual click placement through the SDK.
- Dynamic masks, including no-click games; no illegal action selected.
- Reset once, reset twice, GAME_OVER, WIN, and budget exhaustion.
- History cleared at reset, with run counters and learned weights preserved.
- Positive progress reward with stored context before/after a full reset.
- DQN target masking and PPO joint log-probability agreement before any update.
- Finite losses with masked actions and terminal transitions.
- Architecture substitution: replace the provisional encoder/temporal/action-head implementation through configuration without changing environment, scoring, or core DQN/PPO lifecycle code.
- Checkpoint reconstruction: reload a saved run and verify that the exact architecture/component configuration is reconstructed, not merely the weights of the provisional CNN.
- Full-action preservation: every tested action-head representation must still expose all currently legal actions after masking; no architecture-specific heuristic may silently remove them.

### Learning checks before ARC experiments

Use a small synthetic navigation task and a click task, followed by a distant-cue click task where local target appearance is insufficient. These are implementation/correctness checks, not architecture-selection benchmarks. Verify both algorithms can improve with training and that the configured representation/action path is exercised correctly. A random policy or an updates-disabled version must not be misreported as learning.

On ARC, record first-reward time, rewarded-transition count, action distribution, coverage, loss/value statistics, and episode outcomes. Coverage is a diagnostic, not proof of useful learning. Add a matched updates-disabled control on the development calibration set, keeping initialisation and exploration schedule consistent.

At gamma 0.99, a reward 1,000 actions away is discounted by roughly 0.000043. Compare observed completion lengths against the credit-assignment horizon. A larger interaction budget is not automatically a remedy for delayed rewards.

### Budget calibration

Measure end-to-end time per 10,000 actions, training/inference time, peak RAM/VRAM, and number of updates on representative development games. Do not size experiments from environment-only throughput.

Choose the research action budget B, runtime cap, game counts, and fixed tuning budget from these measurements. Use the same hardware and comparable resource limits where feasible; explicitly justify differences. Save a complete run inventory:

```text
total time estimate = sum(measured time for each scheduled condition/game/seed)
                      + tuning + scoring/submissions + contingency
```

Include ablations and controls in that estimate. If over budget, remove optional candidates, extra seeds beyond three, and unnecessary repeated runs before reducing required versions or core evidence.

## 7. Experiments and improvement decisions

### 7.1 Required matrix

For each algorithm, use these labels:

| Version | Purpose |
|---|---|
| V0 | Baseline configuration; frozen only after architecture-selection and calibration gates pass |
| V1 = V0 + I1 | First evidence-led substantive improvement |
| V2 = V0 + I1 + I2 | Second evidence-led substantive improvement |
| A2 = V0 + I2 | Ablation isolating the second improvement |

V0, V1, and V2 form the six mandatory versions across the two algorithms. Before architecture selection is complete, any V0 implementation is a **pipeline-validation scaffold**, not the frozen assignment baseline. Freeze V0 only after: (1) the required interface/lifecycle tests pass, (2) the architecture shortlist has completed its predeclared reward-based comparison or is explicitly retained as inconclusive, and (3) the baseline algorithm configuration and experiment budget are fixed. V1 already provides the V2-minus-I2 ablation; A2 provides V2-minus-I1. Reuse compatible existing results rather than rerunning them without a reason.

Evaluate each selected stage on matched development games and seeds. At the final freeze, evaluate the six versions and relevant ablations on a predeclared public subset reserved from further tuning, while disclosing its prior architecture-development exposure. At minimum preserve matched comparisons of baselines, final agents, and the ablations needed for the main claims. Treat Kaggle semi-private/private evaluation as the only true unseen competition evidence.

Run the research regime and a competition-compatible regime. If the competition regime changes warm-up or update cadence to fit shorter limits, freeze that configuration on development games and report it separately. Never silently substitute a different algorithm configuration under the same result label.

### 7.2 Choose changes from observed failures

At each stage record: **limitation -> hypothesis -> intervention -> evidence -> conclusion**. Choose one substantive change at a time for each algorithm. Two algorithms may use the same idea, but its effect must be evaluated separately.

Use the same compact loop whenever new evidence arrives:

```text
observed failure or opportunity
-> one plausible explanation
-> smallest test that can distinguish it
-> success/failure criterion fixed before the run
-> keep / drop / retain as inconclusive
-> next highest-value test
```

The purpose is compounding learning, not accumulating features. Prefer a small experiment that kills a bad idea quickly over a large implementation that makes several assumptions at once. When evidence is inconclusive, preserve uncertainty rather than adding complexity to force progress.

Apply the same evidence standard to architecture decisions. The provisional V0 CNN/history/action-head design is an engineering baseline, not an empirical conclusion. Architecture-search proxy metrics may narrow the candidate set, but they must not by themselves establish the final RL architecture. Where the search leaves several credible finalists, compare them under matched reward-based RL conditions before freezing the base architecture. Do not count an architecture change as an evidence-led I1/I2 improvement unless it is introduced and evaluated as such under the assignment protocol.

| Observed limitation | Candidate change | Implementation details and failure checks |
|---|---|---|
| Frequent revisits, few discoveries | Count-based intrinsic reward | Count hashes of permitted observations, scoped by level/run; define update timing and bonus scale. For DQN, explicitly choose stored historical bonuses or recomputation at sampling and disclose nonstationarity. Check whether novelty predicts progress |
| Many ineffective clicks | Learned action-effect prediction | Train on observed action/outcome pairs; use predictions to bias exploration while retaining a nonzero exploration floor. Compare progress as well as prediction accuracy; useful actions can appear visually unchanged |
| History-dependent failures | Recurrent PPO | Feed previous action/reward and observation context into a small GRU. Train contiguous sequences and reset hidden state at declared boundaries; isolated random-transition batches are insufficient |
| Rewards reached but propagated too slowly | Multi-step DQN returns | Store n-step returns with correct discount and terminal handling; compare under unchanged interaction budget |
| Slow initial adaptation | Development-only pretraining | Pretrain on development games, freeze checkpoint selection before held-out evaluation, and report pretraining cost separately |

The candidate table above is illustrative rather than exhaustive. Its presence should not shape the base software architecture beyond the clean extension points already justified by the current experiments. RND and return-to-frontier planning remain optional higher-effort candidates. Do not schedule them before diagnostics justify the work. Hyperparameter changes can support a chosen intervention but cannot substitute for a required substantive improvement.

Keep the project focused on the assignment investigation and the Kaggle participation required by it. The same environment, logging, scoring, and submission infrastructure should support the nominated Kaggle code submission, but do not introduce a separate speculative competition system in this planning pass. Any later post-assignment competition work is outside this implementation brief and must not shape today's architecture beyond ordinary clean interfaces.

Keep frame-change rate, unique-state rate, and novelty as distinct measurements. Do not redefine "frame changed" to mean "new state". Report action/undo cycles and repeated-state transitions so visual movement is not mistaken for progress.

## 8. Reporting and statistical analysis

Save per-run configuration, commit, versions, seed, hardware, runtime, peak memory, interaction counters, scoring output, progress curve, and relevant diagnostics. Preserve selected success and failure trajectories. Freeze a result manifest for the report.

Report:

- Official score with provenance; raw completion and win outcomes alongside it.
- Highest completed-level count, not ambiguously "level reached"; optionally label the next level entered separately.
- Actions spent on unsuccessful runs, and time/actions to completion with non-completions marked as censored at the budget. Never average only successful trajectories without labelling that selection.
- Learning/adaptation curves against actions, with individual seed results where feasible.
- Compute and offline-training cost separately from evaluation/adaptation cost.
- Updates-disabled control results and improvement-specific diagnostics.

For a fixed game set, use stratified resampling of seeds within games for aggregate uncertainty, retaining paired comparisons where appropriate. If discussing generalisation across games, also resample games and state that this addresses a different uncertainty source. Do not treat timepoints from one trajectory as independent runs.

With only three seeds, intervals are descriptive and can be unstable. Report raw outcomes, including all-zero samples; a zero-width empirical interval does not prove the method can never succeed. Show per-game means and success counts. IQM is optional and may hide rare successes when most outcomes are zero; it must not be the only summary.

## 9. Schedule and phase acceptance

Dates are targets. If a check fails, fix it before spending the next stage's experiment budget. Record team decisions in one concise log rather than maintaining many overlapping documents.

| Dates | Phase | Acceptance evidence |
|---|---|---|
| 22-24 Sep | **Phase 1** — Interface, exposure audit, scoring, submission pipeline | Split recorded; reset probes; scorer route; both teams registered; random agent successfully competition-evaluated; current limits documented |
| 24 Sep-1 Oct | **Phase 2** — Minimal implementation, architecture integration, and calibration | Both learners solve synthetic checks; submission callbacks preserve learning; masks/rewards/boundaries verified; provisional V0 is replaceable through configuration; architecture-search finalists can be inserted without changing unrelated infrastructure; complete compute estimate |
| 1-6 Oct | **Phase 3** — Baselines | Matched V0 results, random references, updates-disabled control, official scoring evidence; I1 chosen from diagnostics |
| 6-12 Oct | **Phase 4** — Improvement 1 | V1 compared with V0 separately for both algorithms; I2 chosen from those results |
| 12-18 Oct | **Phase 5** — Improvement 2 | V2 results; development-only final selection; configurations and ablations frozen |
| 18-23 Oct | **Phase 6** — Final evaluation and evidence | Reserved-from-further-tuning public evaluation, ablations, all six required score records, representative trajectories, final tables; unseen claims reserved for Kaggle evaluation |
| By 24 Oct | **Phase 7** — Both Kaggle submissions | Successfully evaluated nominated code and submitted Paper Track write-up; links, identifiers, timestamps, saved copies |
| 24-27 Oct | **Phase 8** — Report and Moodle package verification | Eight-page compliant report, matching notebook, evidence, contribution statement, credential-free ZIP |

Start the report outline during baseline work and draft results after every stage. Reserve about half the main-content space for results, ablations, and analysis. Essential evidence must fit within eight pages; supplementary material cannot replace it.

## 10. Submission checklist

- [ ] Same group membership and consistent team name in both Kaggle tracks; at most four members.
- [ ] Current external entry, merger, and submission deadlines checked on Kaggle.
- [ ] All six required versions have official score evidence or a clearly documented, tutor-resolved limitation.
- [ ] Nominated class-leaderboard code submission successfully evaluated by the Moodle deadline.
- [ ] Code identifier, score, timestamp, and matching notebook preserved.
- [ ] Paper Track write-up submitted by Moodle, under 1,500 words, describing the linked code; link and saved text included.
- [ ] Report uses the RLC 2026 template with cover page disabled, required author details, and at most eight main-content pages.
- [ ] Hyperparameters, budgets, exposure, seeds, methods, limitations, ablations, and resource use included.
- [ ] Reused sources and AI assistance declared; member contribution statement included.
- [ ] No API keys or credentials in submitted files.
- [ ] ZIP named `Kaggle-team-name.zip`; one member uploads and verifies receipt before 27 October, 23:59 SAST.

## 11. Questions to resolve with the tutor or competition documentation

1. Do the chosen DQN/Double DQN and PPO baselines satisfy course/non-course eligibility?
2. If official local scoring is unavailable, what evidence is acceptable for each required version beyond the nominated Kaggle entry?
3. Is the disclosed exposure history and proposed held-out split acceptable?
4. What exactly persists through the competition agent lifecycle, and what online updates are permitted within its runtime limits?

Proceed with independent implementation and verification while these questions are resolved. Do not let an unanswered optional design question halt the whole project.

## 12. Sources and verification status

- `RL_Assignment.pdf`, supplied course brief, especially Requirements, Particulars and Restrictions, Evaluation, and Submission. This controls course deliverables and the Moodle deadline.
- `PROJECT_PLAN.md`, supplied original plan. Referenced external experiment evidence still needs to be supplied or reproduced.
- [Official ARC scoring methodology](https://docs.arcprize.org/methodology), consulted during the review on 22 September 2026. Pin the scoring implementation/version used in experiments.
- [Official reset API](https://docs.arcprize.org/api-reference/commands/start-or-reset-game-instance), consulted during the review. Verify equivalence with the pinned SDK and competition interface.
- [Official Kaggle starter documentation](https://docs.arcprize.org/arc-prize-2026), consulted during the review. A completed notebook run and a successfully evaluated competition submission are distinct steps.
- [PPO: Schulman et al.](https://arxiv.org/abs/1707.06347), algorithm reference.
- [Statistical evaluation: Agarwal et al.](https://arxiv.org/abs/2108.13264), uncertainty and aggregate-reporting reference.
- Add the exact DQN/Double DQN, GAE, code implementation, and chosen improvement references when adopted. Verify current Kaggle rules and limits directly; this document does not certify their unresolved values.

**Immediate next step:** complete the reset/scoring/submission-interface checks and build the small modular RL scaffold while the architecture-selection experiment finishes in parallel. Record any missing evidence that materially affects experiment meaning, but do not block ordinary reversible implementation on unrelated missing notes. Do not freeze the final RL architecture until the predeclared shortlist has undergone matched reward-based validation.

---

<planning_task>
<role>
You are acting as the team's technical research lead: closer to a strong engineering/product lead than the person writing every line of code.

Turn the research plan into a short, decisive implementation brief that another engineer can execute. Your job is to reduce future rewrite cost without hiding uncertainty, keep the critical path obvious, make routine reversible decisions yourself, and protect the scientific meaning of every experiment. More code and more explanation are not automatically more progress.
</role>

<task>
This planning pass has one job: produce the implementation architecture and implementation plan for the assignment requirements and the Kaggle submission path required by the assignment. Do not also implement code, choose the final neural architecture, write the report, perform a literature review, or design a separate speculative competition system.

Use the body of this document as source material and constraints; the `<planning_task>` block controls what you must produce in this pass. Operational details such as report formatting, submission checklists, and later-stage statistical reporting should influence scope only when they affect implementation now; otherwise do not expand them.

If two instructions pull in different directions, resolve the conflict using this order and carry the winning interpretation forward consistently:

1. assignment and competition rules;
2. verified empirical evidence;
3. decisions already settled by the team;
4. working-baseline choices that let implementation proceed;
5. possible later experiments;
6. scheduling or implementation convenience.

Do not preserve two incompatible instructions in the output. Resolve internal contradictions using this priority. Only leave something unresolved when the missing fact is external and genuinely changes the experiment's meaning.

The final neural architecture is still waiting for evidence. The current categorical CNN, short-history representation, and action-head design are a working V0 scaffold. Keep them available as a functioning reference baseline, but do not let their tensor shapes, history length, pooled features, or action-head structure become hidden assumptions in shared interfaces.

Results from the architecture search may narrow the shortlist, but proxy-task performance does not prove which architecture will learn the most reward. The final base architecture should only be locked after credible finalists are compared under matched RL conditions.

Design the repository for the required DQN/PPO investigation and its Kaggle submission path. Keep the agent/environment boundary clean enough that evidence-backed representation or improvement changes do not require rewriting environment semantics, legality checks, scoring, trajectory logging, or evaluation. Do not add generic policy-replacement machinery for hypothetical future agents.
</task>

<decision_policy>
Make routine, reversible engineering decisions yourself when they do not change the research question, experiment semantics, benchmark legality, comparability, or interpretation of results.

If an unresolved choice would materially change any of those things, do not guess. State what evidence is needed to settle it.

For ordinary implementation ambiguity, choose the simplest reversible solution and continue.

Plain-English test:
- If this choice changes what the experiment means, wait for evidence.
- If this choice is just plumbing, choose the simplest sensible option and move on.
</decision_policy>

<deliverable>
Produce a short implementation brief for a technical lead to hand to the engineer doing the implementation.

Use exactly these five sections:

1. **Build now** — only the decisions needed to start. Begin with one compact decision table: `decision | current choice | why | revisit when`.
2. **Boundaries** — the few interfaces that keep architecture, learner, environment, and evaluation separable.
3. **Experiment identity** — the minimum config/checkpoint fields needed to reproduce a run.
4. **Order of work** — an ordered checklist with a pass condition for each step.
5. **Waiting on evidence** — only decisions that genuinely cannot be made yet, with the evidence that will settle each one.

Treat the decision table as the handoff source of truth for the implementation pass. Do not duplicate the same decision elsewhere in different wording.

Communication rules:
- Default format: decision first, then the minimum reason needed to act.
- Target roughly 800-1,100 words; shorter is better if nothing important is lost.
- Write for a technically strong teammate, not a beginner.
- Prefer one compact table plus short bullets over prose.
- Keep each rationale to one sentence unless a longer explanation changes a decision.
- Explain uncertainty by naming the missing evidence, not by writing a long caveat.
- Include code only for tiny interface signatures or configuration shapes that remove ambiguity.
- End with a five-line **Next decisions** block showing the next evidence-driven decision sequence, not a summary.

Resolve internal contradictions before presenting the brief. Do not include a generic risks list unless a specific risk changes an implementation or experiment decision.

Optimise for research iteration speed, low coupling, efficient experiment turnover, low runtime overhead, and code that another teammate can understand quickly.

Prefer direct function/class boundaries and ordinary configuration objects. Do not introduce plugin systems, registries, dependency-injection frameworks, generic DAGs, reflection-based factories, or speculative extension layers unless at least two real current implementations need that mechanism. Keep abstraction proportional to demonstrated variation.

Preserve as much of the existing plan as possible. Do not redesign unrelated parts merely because another design is possible. Use ordinary engineering language throughout.

<quality_gate>
Before returning the brief, check the finished answer against these conditions and fix any failure silently:
- no final architecture was invented;
- no unresolved research choice was disguised as an engineering default;
- V0-specific shapes/history/action structure did not leak into shared interfaces;
- routine reversible engineering decisions were actually made rather than deferred;
- no speculative framework, separate competition agent, or unused subsystem was introduced;
- every planned component can be traced to an assignment requirement, a verified interface need, or a currently justified experiment;
- the critical path is obvious and each implementation step has a pass condition;
- the answer is materially shorter and easier to execute than the source document.

Do not print this checklist or a self-review unless one item cannot be satisfied because an external fact is missing.
</quality_gate>

<tone_preference>
Keep the brief focused and concise. Lead with decisions and actions. Do not turn uncertainty into a long explanation.
</tone_preference>
</deliverable>
</planning_task>
