# Further improvements: research, code audit and plan

Status: **proposal only.** Nothing here is implemented, and `docs/PROJECT_PLAN.md` is unchanged. Every item needs the team's go-ahead. New agent improvements would go in a **Phase 5b (additional improvements)**, before the Phase 6 freeze.

The project's top rule applies throughout:
- agents see only the public frame, the action mask and the progress counters;
- no game-ID logic, game files, human-baseline counts or per-game tuning;
- any sign of cheating stops the work and is reported.

---

## 1. What the research says

### 1.1 How the score works (official methodology)

- **Per-level score** = (human baseline actions / agent actions)², capped. Within a game, levels are weighted by their 1-indexed level number, and the total is the mean of the game scores.
- **Our records match this exactly.** r11l reaching Level 1 at action 10 scores 4.762 = 100/21, the full Level-1 share of a 6-level game. At action 33 it scores 2.116, and at action 67 it scores 0.513. Both values imply the same Level-1 baseline. Actions are counted cumulatively from the start of the run, including those spent before a GAME_OVER.
- **Consequence:** the score rewards finding the next level in very few actions. Behaviour after roughly 3–5× the human count is worth almost nothing for that level, and later levels carry more weight. Learning that only pays off after hundreds of actions barely moves the score. This explains much of "learning-on = learning-off" (§2) and should be stated in the report.
- **Not verified:** secondary sources mention an action cutoff at 5× human, and the Kaggle runtime and action caps are unconfirmed (`docs/DECISIONS.md`). Verify all of these on Kaggle before choosing the competition budget.

### 1.2 What has worked on ARC-AGI-3 without LLMs

| System | Core idea | Result | Lesson for us |
|---|---|---|---|
| StochasticGoose (1st, 2025 preview) | CNN trained online to predict *whether an action changes the frame*. A convolutional per-pixel click head, hierarchical type → click sampling, a de-duplicated buffer, and a model reset at each new level | 12.58%, 18 levels. Fell to 0.25% on the full benchmark once efficiency scoring and unseen games arrived | A dense, self-generated signal (frame change) beats the sparse level reward. A click head must be spatial. Brute force fails under efficiency scoring |
| Graph explorer (Rudakov et al., 3rd private) | Training-free. Segments the frame into single-colour components, masks status bars, hashes states, keeps a directed state graph, and walks the shortest path to the nearest state with untested actions. Click candidates are ranked into 5 priority tiers | Median 30/52 levels. In ablations, segmentation alone lifted random play (vc33 0 → 5 levels) | Object-level clicks plus systematic return-to-frontier is the strongest non-LLM recipe. Status-bar masking matters for state hashing |
| Learned priorities on the graph explorer (Stanford CS224R) | Replaced the hand priority tiers with online-learned segment scorers (REINFORCE, FOMAML, SAC, PPO) trained on ±1 "did the click change the state" | FOMAML 7.5× the baseline on vc33. PPO underperformed (too little data) | RL fits best as a *small learned component* on a good exploration scaffold, with a dense signal and adaptation at each level |
| Milestone #1 winners, 2026 | Local LLM / VLM agents. Reki's numpy click heuristic prefers small, rare-coloured shapes and stops clicking object types that never change anything ("dead signature") | Top of the 2026 milestone | Not usable for our two-RL-algorithm requirement, but the dead-click rule is a cheap, general idea. Organisers note local public-game checks were a poor proxy for the leaderboard |

### 1.3 RL literature that maps onto our failures

- **Exploration by memory and return:** Go-Explore (Ecoffet et al. 2021) and count-based or UCB exploration (Strehl & Littman 2008; Bellemare et al. 2016). Episodic novelty (NGU, Badia et al. 2020) also fits.
- **Action-space reduction:** invalid-action masking in policy gradients (Huang & Ontañón 2020) and object-centric action abstraction.
- **Auxiliary self-supervised losses:** UNREAL (Jaderberg et al. 2017).
- **Options and semi-MDPs:** Sutton, Precup & Singh 1999. Our I2 infrastructure already implements this.
- **Fast adaptation at each level:** meta-learning (FOMAML), or simply resetting state-specific memory at each level (StochasticGoose).

---

## 2. Code and results audit: where the agents are weak

All numbers below come from the existing Phase 3–5 records: 17 development games × 3 seeds × 1,500 actions.

| # | Weak point | Evidence | Where |
|---|---|---|---|
| W1 | **The DQN's greedy policy collapses onto one fixed action.** About 78% of DQN actions are greedy (ε reaches 0.1 by action 400). The Q-network scores clicks from normalised (x, y) plus a *global* state vector, so nothing ties a click to what is at that pixel. With almost no reward, argmax returns the same action again and again | Single actions repeated: ft09 two actions ×238/×236; m0r0 ×361; sp80 ×442; tn36 ×419. Late-run distinct actions only 55–117 | `src/dqn_v1/action_bank.py:24-36`, `src/dqn_v1/qnet.py:59-73`, `src/config.py:43-45` |
| W2 | **The DQN's score is almost entirely the salience explorer's first few dozen actions, on two games.** tn36 + r11l give 17.07 of the 17.20 total score points (99%). Every DQN success arrives at actions 10–67, while ε ≈ 1 | DQN completes 9 levels vs **10 for random_type**. It is faster, not broader | `docs/PHASE3_RESULTS.md`; per-game breakdown |
| W3 | **Click exploration has no memory.** The click choice is a softmax over a static salience map. State-action counts are applied only to simple actions, so clicks that did nothing are retried | DQN frame-change rate is ft09 0.05, ar25 0.17, sc25 0.39, su15 0.31. random_type gets 0.82 / 0.74 on ar25 / sc25 | `src/dqn_v1/exploration.py:103-113` |
| W4 | **The prior's click-effect output is computed but never used.** `state_outputs` scores all 4,096 clicks every call, but the explorer only checks that the result is not `None`. The prior's action head also ignores its spatial features | Lost signal and wasted compute | `src/dqn_v1/prior.py:71-84,122`, `src/dqn_v1/exploration.py:105` |
| W5 | **PPO's click policy never sharpens.** The most-repeated click occurs ≤4 times in 1,500 actions, with about 1,200 distinct clicks. Under seed 0, *the same click sequence appears in different games*: the click head is effectively uniform. PPO also gets only ~11 updates per run (rollout 128) | PPO ≈ random_type with learned type preferences. tn36 0/3, vc33 0/3 | `src/ppo.py`, `src/config.py:27` |
| W6 | **The novelty state key is the full frame.** Frames that differ only in a counter or status bar count as new states. Likely evidence: tr87 shows 1,418 "unique states" in 1,500 steps using 4 actions; ls20 882; re86 1,214 | I1's bonus may be almost uniform, which could explain its null result. Diagnostic D1 below confirms or rejects this | `src/intrinsic.py:35-37` |
| W7 | **No systematic coverage in navigation (keyboard) games.** g50t, ls20, re86 and tr87 show essentially no progress in any of the 8 conditions × 3 seeds | A random-walk-like policy over 4 actions with no "go back to where untried options remain" | Learner design |
| W8 | **Level 2 is almost never reached** (4 runs out of ~540 learner runs at 1,500 actions). Nothing adapts when a level changes | Level-weighted scoring makes this the largest unclaimed score | Learner design |
| W9 | **The algorithm comparison is confounded.** The DQN baseline has a hand-designed salience explorer and an offline-pretrained encoder. PPO has neither | "DQN beats PPO" partly means "salience explorer beats uniform clicks" | §4.1 of plan (disclosed, but not controlled) |
| W10 | **Compute waste.** Features are computed twice per step (`act` and the next `observe`). There are 4,096-row Python candidate encodings, a 4,096-click prior scoring, a pure-Python flood fill for salience, and a per-sample Python loop in the train step | DQN takes 167 s per run vs 77 s with updates off. This limits seeds and Kaggle runtime | `src/dqn_v1/learner.py:98,125`, `src/dqn_v1/action_bank.py:38-54`, `src/dqn_v1/exploration.py:70-92`, `src/dqn_v1/agent.py:233-289` |
| W11 | **The anti-cheat guard is static only.** It scans a hard-coded file list (a new learner file is not scanned unless added) and has no runtime check of file or network access | Weaker evidence for the report's leakage section | `src/guard.py:27-31` |
| W12 | **Kaggle path not yet real.** The learner is not packaged into the notebook, runtime and action caps are unverified, and the DQN hidden-game checkpoint is fold 0, trained on public games including reserved ones | No real-agent submission exists | `kaggle/learner_agent.py:3-4`, `src/checkpoint_resolver.py:21,53` |

### 2.1 Which part of this plan addresses each weak point

"Addressed" means *expected to address*. Every agent change is a hypothesis that the matched grid runs will test.

| Weak point | Addressed by | How directly |
|---|---|---|
| W1 DQN greedy collapse | I3 | **Mostly direct.** Object rows carry what is at each location (colour, size, position), so a click is tied to frame content. Known no-op objects are removed from argmax, so a dead click cannot be repeated. It may still fixate on one *effective* object |
| W2 score is 99% two games | R1, R2 | **Not fixable by an agent change.** It is a property of the scoring. Remedy: metrics not dominated by two games (Level-1 rates, actions-to-first-level curves, scores without tn36 / r11l). I3/I4 may spread success, but that is a hope, not a fix |
| W3 no click memory | I3 (effect memory) | **Direct** |
| W4 prior click output unused | R6 | **Not fixed.** It becomes irrelevant under object clicks, and R6 stops computing it. Nothing recovers that signal |
| W5 PPO uniform clicks; ~11 updates per run | I3 | **Half.** Uniform clicks: direct (≈10–60 object choices, dead ones masked). Few updates: **not addressed**; rollout length is a hyperparameter and could change only as a disclosed supporting setting |
| W6 novelty key includes counters | §3.4 masking + D1 | **Direct, if D1 confirms the cause.** If not, this weak point is withdrawn |
| W7 no keyboard-game coverage | I4 | **Direct** (its main target) |
| W8 Level 2 almost never | I4 partly, I5 | **Partial.** I4 helps only if it first raises Level-1 completion. I5 targets Level 2 directly but runs only if the Level-1 rate reaches one third |
| W9 DQN vs PPO confound | I3 + R3 | **Partial.** I3 gives both algorithms the same object choices. The DQN's pretrained encoder remains unless "I3 without the prior" (decision 3) is tested |
| W10 compute waste | R6 | **Direct**, as behaviour-preserving engineering (not an improvement) |
| W11 static-only guard | R7 | **Direct** |
| W12 Kaggle path not real | §5 | **Direct**, as engineering work |

Left partly open: PPO update count (W5), and the encoder half of the confound (W9). The report should state both.

**Why I2 (repeat option) hurt, in one line:** on click games the dominant failure is *breadth* (W1, W3), and repeating one click deepens it. tn36 went from 3/3 runs reaching Level 1 to 0/3.

---

## 3. Proposed improvements (ranked)

Each card follows the assignment's required chain: limitation → hypothesis → intervention → evidence → conclusion. Each idea runs inside **both** algorithms and is evaluated separately per algorithm. Constants are fixed before any run, and nothing is tuned on reserved games.

**Version labels need a decision (D-2):** I2 was not promoted, so the proposal is to build new stages on V1:
- **V3 = V1 + I3**, ablation **A3 = V0 + I3**;
- then **V4 = V3 + I4**, ablation **A4 = V0 + I4**.

### I3 (recommended first): object-centric click actions with effect memory

| | |
|---|---|
| **Limitation** | W1, W3, W5: neither algorithm can learn *which thing to click*. DQN clicks a fixed pixel; PPO clicks uniformly over 4,096 pixels; clicks that do nothing are retried |
| **Hypothesis** | Replacing 4,096 pixel clicks with K ≈ 10–60 object candidates gives a learnable choice set. Remembering which (state, object) pairs had no effect raises the effective-action rate and brings first progress earlier, for both algorithms |
| **Shared mechanism** | 1. Segment the current public frame into single-colour connected components; each component is one click candidate, clicked at a representative pixel. 2. Mask the status bar / counter, detected online and generically (§3.4). 3. Effect memory: a per-run table (masked-state hash, object signature) → {changed, no change, game over}, with a small retry probability so hidden-state effects are not ruled out forever |
| **DQN change** | The Q-network already scores a variable candidate set. Replace the 4,096 (x, y) rows with one row per object, with features (colour one-hot, size, bounding box, centroid, times tried, last outcome). Known no-op candidates are excluded from argmax and from the explorer |
| **PPO change** | Per-object logits: pool the click head's spatial features inside each object mask (a pointer-style head). Known no-ops are masked as invalid actions, with the mask stored in the rollout so log-probs stay consistent |
| **Mechanism checks** | Candidate count per frame; effective-action rate; distinct effective actions; no-op retries; PPO click entropy falls over a run |
| **Tests** | Synthetic click task where only one object responds; segmentation unit tests; status-bar masking does not hide play-area changes; mask → log-prob consistency; zero memory reproduces the I3-without-memory actions |
| **Ablations** | A3 (V0 + I3); I3 without effect memory, which separates "object actions" from "memory" |
| **Success / failure** | Predeclared, as in Phases 4–5: paired score difference with a game-cluster bootstrap. **Plus Level-1 runs and actions-to-first-level** (§4), because the score is dominated by two games |
| **Why it helps Kaggle** | Unseen games are mostly about finding the interactive object quickly. This is the main lever in every strong non-LLM entry |
| **Risks** | Games where the click position *within* an object matters; large background components. Mitigation: keep a "random pixel" candidate and split very large components into tiles |
| **Anti-cheat** | Uses only the current frame. No per-game segmentation rules; constants fixed once |
| **Bonus** | A DQN that no longer needs the frozen prior's click head lets us test "I3 without the prior". If it holds up, it removes the reserved-game exposure from the deployed agent |

### I4 (recommended second): frontier-directed exploration option (Go-Explore-style)

| | |
|---|---|
| **Limitation** | W7, W8: no coverage of navigation games; Level 2 almost never reached |
| **Hypothesis** | A state graph over masked states, plus an option "walk the shortest known path to the nearest state that still has untried actions", lets the learners cover reachable states systematically within 1,500 actions |
| **Intervention** | The graph is built only from observed transitions: masked-hash nodes, action edges, and an untried set per node. One new **option** is added to each learner's action set, reusing the I2 semi-MDP machinery (γ^k, summed reward, every step charged). DQN learns Q(s, go-to-frontier); PPO learns when to choose it. I1's bonus (on masked states) rewards reaching new states |
| **Mechanism checks** | How often the option is chosen; the share of the path that succeeds (graph accuracy); new states per 100 actions; actions to the first new state after a reset |
| **Ablations** | A4 (V0 + I4); **option forced** (a pure graph-explorer reference policy); **option never** (= the base) |
| **Why the forced option matters** | It is a strong, training-free reference policy for the report: "how close is RL to systematic search?" |
| **Risks** | Non-deterministic transitions break paths (fall back after one failed step). Actions that trigger a reset must be marked as tested (the bug reported by Rudakov et al.). Memory growth (cap the number of nodes) |
| **Anti-cheat** | The graph comes from the agent's own observations only, and is reset every run / game |

### I5 (conditional): per-level adaptation (the plan's candidate B)

Only worth running if I3/I4 bring Level-1 completion to at least one third of runs; otherwise it is rarely activated (the plan's own rule).
- **Mechanism:** on a level-up, clear state-specific memory (the graph and the masked-state tables). Keep cross-level *object-type* effect statistics with decay.
- **Ablations:** beliefs reset at every level; shuffled beliefs.
- **Motivation:** StochasticGoose's reset at each level, and the FOMAML result.

### Lower-priority alternatives (only if the team prefers them over I3/I4)

| Idea | Why lower |
|---|---|
| Spatial per-pixel Q head for the DQN (the existing `src/model.py` ClickHead pattern) | Fixes W1 but keeps the 4,096-click search; I3 covers the same failure more cheaply |
| UCB-style count bonus inside DQN action selection | Addresses greedy collapse directly, but overlaps I3's effect memory |
| PPO auxiliary frame-change prediction loss (UNREAL-style) | Gives a dense learning signal, but cluster history reports nulls for mechanics losses; needs a new falsifiable question |
| Larger PPO update cadence | A hyperparameter change: not a substantive improvement under the brief |

### 3.4 Shared component: generic status-bar / counter masking

Within a run, pixels that change on nearly every step *regardless of the action* are a counter or timer. Exclude them from state hashes (I1, I3, I4).
- **Diagnostic D1 first (cheap, ~1 h):** recompute I1's unique-state counts on the stored traces with and without the mask. This tells us whether I1's null result was a state-key problem, which is a strong analysis paragraph for the report either way.
- **Rules:** the mask is learned online from frames only; no hard-coded bar positions per game.

---

## 4. Rigour and report strengthening (worth marks even if scores do not rise)

| # | Item | Why |
|---|---|---|
| R1 | **Metrics that are not dominated by two games:** Level-1 success counts with Wilson intervals; **actions-to-first-level survival curves** (Kaplan–Meier, non-completions censored at the budget, as plan §8 asks); per-game heatmaps; score sensitivity excluding tn36 / r11l | The mean score is 99% two games, so the current conclusions are fragile |
| R2 | **Failure analysis of learning-on = learning-off:** greedy-collapse statistics (W1), Q-value spread, update counts, and the efficiency-scoring argument (§1.1) | A well-evidenced negative result is explicitly credited by the brief |
| R3 | **Confound control (W9):** state that the DQN's advantage comes from its explorer. I3 gives both algorithms the same object prior, which makes the algorithm comparison fair from V3 on | Fair-comparison criterion |
| R4 | **Five seeds for the final matched comparisons**, made affordable by the compute fixes (R6) | Three seeds give unstable intervals |
| R5 | **Replays** of representative successes and failures using the SDK's `save_recording` (one success and one failure per algorithm) | "Representative success and failure cases, supported by trajectories or replays" |
| R6 | **Behaviour-preserving speed-ups:** cache features between `observe` and `act`; vectorise candidate encoding; `scipy.ndimage.label` for segmentation; skip unused prior click scoring. Must reproduce the existing action traces exactly | More seeds, plus Kaggle runtime headroom |
| R7 | **Stronger leakage controls:** the guard auto-discovers every file under `src/` except an allow-listed controller set; plus a runtime `sys.addaudithook` during runs that logs every file open, import and socket, and fails if anything outside `assets/` and the venv is touched | Hard evidence for the "no leakage" section, and enforces the top rule |
| R8 | **Reserved-game evaluation once, after freezing** (Phase 6), with exposure disclosed. Kaggle is the only true unseen evidence | Generalisation claims |

---

## 5. Kaggle strengthening

1. **Verify on the competition page:** runtime limit, per-game action cap, number of hidden games, internet/GPU, and whether any cross-game memory is allowed. Record the findings in `docs/DECISIONS.md`.
2. **Budget:** an unfinished level scores 0 whatever the action count, so use as many actions per game as the verified runtime allows. Measure seconds per action after R6.
3. **Packaging:** ship `src/` plus `assets/` as a Kaggle dataset; CPU-only; offline wheels; a Phase-A dry run on two public games through the framework bridge, checking that the scorecard actions equal the local count.
4. **Selection:** pick the final agent on development games only, then freeze it, then do one Kaggle submission (the limit is 1 per day) and nominate it. If I3 without the prior holds up, deploy that (no checkpoint exposure); otherwise use the predeclared fold-0 fallback, disclosed.

---

## 6. Do not do (cheating / overfitting traps)

- Use human-baseline counts, level-solution knowledge, `environment_files/`, or anything recovered from scores inside an agent. (Recovering baselines from scores is fine *only* for analysis text, and is better avoided.)
- Make per-game constants, game-ID branches, or segmentation / priority rules tuned to particular games.
- Tune any constant on reserved games, or rerun reserved evaluation after looking.
- Carry memory across games on Kaggle unless the rules explicitly allow it, and disclose it if so.
- Use LLM / VLM APIs (barred by the competition, and outside the RL requirement).
- Present a hand-coded explorer as the "learning algorithm". The learners must make the decisions, and the forced-option and no-learning ablations must show what learning adds.

---

## 7. Recommended order and cost

Run times are from the Phase 5 precedent: 204 runs took about 4 h on 8 workers.

| Step | Work | Compute |
|---|---|---|
| 1 | D1 masked-novelty diagnostic on stored traces; R6 speed-ups (traces must reproduce) | ~1 h, no new grid |
| 2 | R7 guard + runtime audit hook | Tests only |
| 3 | I3 build + synthetic tests + a 2-game smoke (dev games only) | Small |
| 4 | I3 grid: V3 + A3 + "no memory" ablation, both algorithms | ~5–6 h |
| 5 | I4 build + tests + smoke, then grid: V4 + A4 + forced-option reference | ~5–6 h |
| 6 | I5 only if the Level-1 rate is ≥ 1/3 | ~4 h |
| 7 | Development-only final selection and freeze → Phase 6 reserved evaluation (5 seeds) → Kaggle | Per Phases 6–7 |
| 8 | R1–R5 analysis and figures feed the report | Analysis |

## 8. Decisions needed from the team

1. Approve a Phase 5b and which improvements: **recommended I3, then I4; I5 conditional.**
2. Label scheme: **V3 = V1 + I3** (drop the unpromoted I2) or V3 = V2 + I3.
3. Whether to test an I3 DQN **without** the frozen prior, as a deployment option.
4. Five seeds for final comparisons (recommended) or three.

## Sources

- ARC-AGI-3 scoring methodology — https://docs.arcprize.org/methodology
- ARC Prize 2026 Kaggle kit — https://docs.arcprize.org/arc-prize-2026
- ARC-AGI-3 technical report — https://arxiv.org/abs/2603.24621
- ARC-AGI-3 Preview competition results — https://arcprize.org/competitions/arc-agi-3-preview-agents
- StochasticGoose code — https://github.com/DriesSmit/ARC3-solution
- Rudakov, Shock, Cowley, *Graph-Based Exploration for ARC-AGI-3* — https://arxiv.org/abs/2512.24156
- Bookman, Mendeleev, Feinstein, *Learning Priority Functions for Graph-Based Exploration in ARC-AGI-3* (Stanford CS224R)
- ARC Prize 2026 Milestone #1 winners — https://arcprize.org/blog/arc-prize-2026-milestone-1
- Ecoffet et al. 2021 (Go-Explore); Strehl & Littman 2008 (MBIE-EB); Bellemare et al. 2016 (pseudo-counts); Badia et al. 2020 (NGU); Huang & Ontañón 2020 (invalid action masking); Jaderberg et al. 2017 (UNREAL); Sutton, Precup & Singh 1999 (options)
