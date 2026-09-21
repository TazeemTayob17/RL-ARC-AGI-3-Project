# ARC-AGI-3 Environment Findings

Hands-on verification of the environment, run on this machine on **2026-09-21**.
Companion to `RESEARCH_NOTES.md` (the desk-research claims) and `ENV_RECON_PLAN.md`.

Every fact below carries an evidence tag:

| Tag | Meaning |
|---|---|
| `[RUN]` | Observed in an actual run on this machine. Evidence file named. |
| `[SRC]` | Read directly from installed package source (authoritative, but not exercised). |
| `[DOCS]` | From official repo README / PyPI metadata. |
| `[UNVERIFIED]` | Could not be checked here. |

Raw outputs are in `docs/evidence/`. Probe scripts are in `scripts/probe/`.

---

## 0. Setup

- Python **3.12.14**, installed via `uv python install 3.12` (uv 0.12.17). `[RUN]`
- Venv lives at **`C:\Users\Lenovo\dev\arc-venv`**, *not* in the repo — see
  section 9 for why.
- Exact versions in `requirements.txt` (31 pinned packages). Key ones: `[RUN]`

  | package | version |
  |---|---|
  | arc-agi | 0.9.9 |
  | arcengine | 0.9.3 |
  | numpy | 2.5.3 |
  | pydantic | 2.13.5 |
  | matplotlib | 3.11.2 |
  | pillow | 12.3.0 |

- `arc_agi` exposes **no `__version__` attribute** — use `importlib.metadata.version("arc-agi")`. `[RUN]`
- **No `ARC_API_KEY` was ever set.** The SDK silently fetches an anonymous key on first
  use (`Got anonymous API key: ...`) and everything below worked without registration. `[RUN]`

Run everything as:

```bash
C:/Users/Lenovo/dev/arc-venv/Scripts/python.exe scripts/probe/<probe>.py
```

---

## 1. Observation format

Confirmed by run (`evidence/inspect_frame.txt`, `evidence/action_space.txt`,
`evidence/level_transition.txt`):

- `env.reset()` and `env.step()` both return a **`FrameDataRaw`**, not `FrameData`. `[RUN]`
  The difference matters: on `FrameDataRaw` the grid is a runtime-only private attr
  (`List[np.ndarray]`) that is **not serialised** by `model_dump_json()`; `FrameData`
  stores it as a plain `list[list[list[int]]]`. `[SRC]`
- Grid is **64 x 64**, dtype **`int8`**. `[RUN]`
- `np.array(frame.frame).shape == (n_frames, 64, 64)`. `[RUN]`
- **One step can return multiple frames — confirmed.** `[RUN]` Frames-per-step is *not*
  constant, and varies *within* a single game:

  | game | n_frames at reset | n_frames mid-game |
  |---|---|---|
  | ls20 | 1 | 1 |
  | ft09 | 1 | **5** (steps 1911-1914), **2** at the level transition |
  | bp35 | **2** | not measured |
  | lf52 | **2** | not measured |

  Practical consequence: always index `frame.frame[-1]` for "current" state, and never
  assume `len(frame.frame) == 1`.
- **Colour range: research says 0-15; observed 0-12.** `[RUN]` In `ls20` the observed
  unique values were `[0,1,3,4,5,8,9,11,12]`. 0-15 is *plausible* as the declared range
  but **no run produced a value above 12**, and nothing in `arcengine` source pins the
  upper bound. Treat 0-15 as the safe allocation size, not as a verified range.

### `FrameData` / `FrameDataRaw` fields `[SRC]` + `[RUN]`

From `arcengine/enums.py`, confirmed live:

| field | type | notes |
|---|---|---|
| `game_id` | `str` | |
| `frame` | `list[list[list[int]]]` / `List[ndarray]` | outer dim = frames in this step |
| `state` | `GameState` | |
| `levels_completed` | `int`, `ge=0, le=254` | **renamed from `score`** |
| `win_levels` | `int`, `ge=0, le=254` | **renamed from `win_score`**; total levels in game |
| `action_input` | `ActionInput` | echoes back `{id, data, reasoning}` |
| `guid` | `str \| None` | |
| `full_reset` | `bool` | `True` on first reset, `False` on later ones |
| `available_actions` | `list[int]` | **ints, not `GameAction`** |

`score` and `win_score` are **absent** in 0.9.9 — confirmed by introspection, not just
the changelog. `[RUN]` There is also `FrameData.is_empty()`. `[SRC]`

`GameState` = `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`. `[SRC]`

---

## 2. Action space

`GameAction` is an enum of 8 members (`evidence/action_space.txt`): `[RUN]`

| member | id | payload type |
|---|---|---|
| `RESET` | 0 | `SimpleAction` |
| `ACTION1`-`ACTION5` | 1-5 | `SimpleAction` |
| `ACTION6` | 6 | `ComplexAction` (x, y) |
| `ACTION7` | 7 | `SimpleAction` |

- **`ACTION7` exists.** `[RUN]` It is a normal simple action in the enum; whether a game
  *offers* it is a per-game matter (6 of 25 games do). The "Undo" semantics are
  `[DOCS]` only — not exercised here.

### `available_actions` varies a lot per game `[RUN]`

All 25 games surveyed at reset. Ten distinct action sets:

| available_actions | games |
|---|---|
| `[6]` | ft09, lp85, r11l, s5i5, tn36, vc33 |
| `[6,7]` | su15 |
| `[5,6,7]` | sb26 |
| `[3,4,6,7]` | bp35 |
| `[1,2,3,4]` | ls20, tr87, tu93 |
| `[1,2,3,4,5]` | g50t, re86, wa30 |
| `[1,2,3,4,6]` | dc22, ka59, sc25 |
| `[1,2,3,4,6,7]` | lf52, sk48 |
| `[1,2,3,4,5,6]` | cd82, cn04, m0r0, sp80 |
| `[1,2,3,4,5,6,7]` | ar25 |

Six games are **click-only** (`[6]`). Nineteen expose ACTION6; six expose ACTION7.
Use `env.action_space` to get these as `GameAction` objects instead of raw ints. `[SRC]`

> **Caveat:** `available_actions` is advisory, not enforced. The official `Random`
> agent template sends ACTION5/6/7 to `ls20` (which advertises only `[1,2,3,4]`) and
> the environment **accepts them without error**. `[RUN]`
> (`evidence/random_agent_3games.txt`)

### ACTION6 coordinates `[RUN]` (`evidence/action6.txt`)

**Only one of the two plausible forms actually works:**

```python
# WORKS — data reaches the engine, echoed back in action_input.data
env.step(GameAction.ACTION6, data={"x": 10, "y": 20})

# DOES NOT WORK via the wrapper — action_input.data comes back {}
a = GameAction.ACTION6
a.set_data({"x": 40, "y": 5})
env.step(a)
```

`set_data()` mutates the enum member's `action_data`, but `LocalEnvironmentWrapper.step`
reads the `data=` kwarg, so the payload is dropped. Use `data=`.

- **Coordinate bounds are NOT enforced on this path.** `ComplexAction` declares
  `x, y: int = Field(0, ge=0, le=63)`, but `step(ACTION6, data={"x": 64, "y": 0})` and
  `{"x": -1}` were both **accepted without an exception**. `[RUN]` Validation is
  bypassed; clamp coordinates yourself.
- **Origin:** `arcengine/base_game.py` indexes `rendered_pixels[y, x]`, so **x = column,
  y = row, origin top-left**. `[SRC]` This could *not* be confirmed by run — in `vc33`
  every click, at any coordinate, changed only cell `(row 0, col 63)`, which is a HUD
  counter, not a cursor (`evidence/action6_origin.txt`). **`[UNVERIFIED]` by run.**

---

## 3. Reset, GAME_OVER, and level progression

From `evidence/game_over.txt` and `evidence/level_transition.txt`, random policy,
20,000 steps per game, resetting on every terminal state:

| game | episodes | GAME_OVER | WIN | best level | mean episode length |
|---|---|---|---|---|---|
| ls20 | 154 | 154 | 0 | 0 | 129 |
| vc33 | 400 | 400 | 0 | 0 | 50 |
| ft09 | 67 | 67 | 0 | **1** | 295 |
| tr87 | 156 | 156 | 0 | 0 | 128 |
| sb26 | 103 | 103 | 0 | 0 | 193 |

**GAME_OVER behaviour — this contradicts the research notes.** `[RUN]`

- In `GAME_OVER`, `available_actions` **still lists the game's normal actions**, not
  `[RESET]`. e.g. ls20 reports `[1,2,3,4]` while terminal.
- Sending a non-RESET action while in `GAME_OVER` is **accepted with no exception**.
  It returns `state=GAME_OVER` and **`n_frames=0`** (`frame.is_empty() == True`).
  There is no HTTP 400 — there is no HTTP layer at all in local mode.
- `env.reset()` recovers cleanly, giving `state=NOT_FINISHED`, `levels_completed=0`.
- `full_reset` is `True` on the very first reset of an env and `False` on subsequent
  resets — a usable "new episode vs. first episode" signal.

**Level completion shows up as `levels_completed` incrementing, with `state` staying
`NOT_FINISHED`.** `[RUN]` Captured exactly (`ft09`, seed 7, step 1915):

```
  step action   state         lvl win n_frames avail  full_reset
  1914 ACTION6  NOT_FINISHED    0   6        5 [6]    False
  1915 ACTION6  NOT_FINISHED    1   6        2 [6]    False   <== level completed
```

Note the frame count drops 5 -> 2 at the boundary. `state` does **not** become `WIN` at
a level completion; `WIN` presumably requires `levels_completed == win_levels`, which
no run reached. `[UNVERIFIED]`

`win_levels` (total levels per game) ranges **6 to 10** across the 25 games `[RUN]`:
6 for cd82/cn04/dc22/ft09/m0r0/r11l/sc25/sp80/tr87, up to 10 for lf52.

---

## 4. Determinism

**Confirmed deterministic.** `[RUN]` (`evidence/determinism.txt`)

A fixed 40-action sequence replayed on `ls20`, `vc33`, `ft09`, comparing SHA-256 of
every returned grid at every step:

- Two rollouts in the **same process**: byte-identical at every step, all 3 games.
- A third rollout in a **separate process**: identical final digests
  (`ls20 e838ffabd09b302b`, `vc33 f58c258500bae4cb`, `ft09 4cd76b009d421f90`).

`Arcade.make()` takes a `seed: int = 0` parameter. `[SRC]` Seed sensitivity was not
explored. `[UNVERIFIED]`

---

## 5. Speed

`[RUN]` (`evidence/speed.txt`) — 2,000 steps per game, no rendering, random actions:

| game | raw steps/s | + numpy conversion |
|---|---|---|
| ls20 | 439 | 488 |
| vc33 | 822 | 753 |
| ft09 | 529 | 561 |
| tr87 | 456 | 463 |
| **mean** | **562** | **566** |

- **This contradicts the claimed 1,000-2,000 steps/s.** On this machine the environment
  runs at roughly **440-820 steps/s**, i.e. about **2-4x slower** than the research note
  claims. Plan sample budgets accordingly.
- Converting the frame to a numpy array costs **nothing measurable** — `FrameDataRaw.frame`
  is already a list of `ndarray`, so `np.asarray` is a no-op view.
- The framework's own reported figure agrees: `play_local.py` logged 238-578 "avg fps".

---

## 6. Offline operation

`[RUN]` (`evidence/offline.txt`)

- There is a **fourth operation mode the research notes miss**: `OperationMode` is
  `NORMAL`, `ONLINE`, `OFFLINE`, **`COMPETITION`**. `[SRC]`
- `Arcade(operation_mode=OperationMode.OFFLINE)` discovered **all 25 games** from the
  local cache and played `ls20` for 20 steps with **no API fetch**. `[RUN]`
- Game sources cache to `environment_files/<game>/<version>/<game>.py` on first use
  (25 directories after this session). First contact *does* require network; everything
  after is genuinely local.

---

## 7. Kaggle packaging (ARC-AGI-3-Kaggle-Starter)

Repo cloned to `vendor/ARC-AGI-3-Kaggle-Starter`.

**`agent/my_agent.py`** is the only file you edit. It defines `class MyAgent(Agent)`
with `MAX_ACTIONS = 80` and two methods — `is_done(frames, latest_frame) -> bool` and
`choose_action(frames, latest_frame) -> GameAction`. `[RUN]`

**`scripts/build_notebook.py`** splices that file into `notebooks/submission.ipynb`:
it reads `agent/my_agent.py` verbatim, embeds it as a notebook cell, and wraps it with
the Kaggle gateway plumbing that runs the agent over the hidden games and writes
`/kaggle/working/submission.parquet`. Accelerator is one line at the top:
`ACCELERATOR = "t4"`, chosen from `_ACCELERATORS = {cpu, t4, p100, rtx6000}`
(`cpu` maps to `{"name": "none", "gpu": False}`). `[RUN]`

**`scripts/slim_framework.py`** overwrites the framework's `agents/__init__.py` with a
stub registering only `Random`. Upstream `__init__.py` eagerly imports langgraph,
smolagents, openai and friends; the slim version avoids installing them. The starter
notes this is the same trick the official "Stochastic Goose" sample uses on Kaggle.

**Make targets** `[RUN]`:

| target | does |
|---|---|
| `setup` | venv + `arc-agi`/`kaggle`/pandas/pyarrow + clone framework + slim it |
| `play-local` | run `my_agent.py` on all games (`GAME=ls20` for one, `STEPS=n`) |
| `verify-local` | 50-step smoke test on ls20 + vc33 |
| `list-games` | `play_local.py --list` |
| `pull-sample` | download the Stochastic Goose reference notebook |
| `notebook` | build `notebooks/submission.ipynb` from `my_agent.py` |
| `submit` | `notebook` + push to Kaggle |
| `status` | tail the latest Kaggle run |
| `clean` | remove venv, downloaded games, vendored repos |

### The Makefile does not run on Windows `[RUN]`

`PYTHON ?= python3.12`, `$(VENV)/bin/python`, `cat`, `chmod` are all POSIX. On Windows
the venv is `.venv/Scripts/python.exe`. We bypassed `make` and called the scripts
directly. Three further portability defects found while doing so:

1. **`slim_framework.py` writes cp1252 on Windows.** `INIT.write_text(SLIM)` has no
   `encoding=`, so the em-dash in its docstring lands as byte `0x97`, and the next
   import dies with `SyntaxError: 'utf-8' codec can't decode byte 0x97`. Fix:
   `write_text(SLIM, encoding="utf-8")`.
2. **`play_local.py` crashes when stdout is piped.** It prints a `->` arrow glyph; under
   the default Windows console encoding that raises `UnicodeEncodeError`. Workaround:
   `PYTHONIOENCODING=utf-8`.
3. **`--max-steps` can only lower the cap, never raise it** —
   `MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS, args.max_steps)`. With
   `MAX_ACTIONS = 80`, `--max-steps 200` silently does nothing, contradicting the
   script's own `--help` text ("overrides MyAgent.MAX_ACTIONS").

### Random agent run `[RUN]` (`evidence/random_agent_3games.txt`)

```
  ft09     levels=  0  actions=   81  state=GameState.NOT_FINISHED
  ls20     levels=  0  actions=   81  state=GameState.NOT_FINISHED
  vc33     levels=  0  actions=   81  state=GameState.NOT_FINISHED
Aggregate scorecard score: 0.0
```

---

## 8. Claim-by-claim verdict vs. `RESEARCH_NOTES.md`

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `arc-agi` requires Python 3.12 | **PARTLY WRONG** — metadata says `>=3.12`, so 3.13/3.14 are not excluded. Repos *pin* 3.12 via `.python-version`. | PyPI, repo `[DOCS]` |
| 2 | `Arcade()`, `arc.make("ls20", render_mode="terminal")` | CONFIRMED (signature also takes `seed`, `scorecard_id`, `save_recording`, `include_frame_data`, `renderer`) | `[SRC]`/`[RUN]` |
| 3 | Local/offline since 0.9.3; cached in `environment_files/` | CONFIRMED | `offline.txt` `[RUN]` |
| 4 | Online play needs `ARC_API_KEY`; only online posts to leaderboard | PARTLY CONFIRMED — no key needed for local; SDK auto-fetches an **anonymous** key. Leaderboard half `[UNVERIFIED]` | `[RUN]` |
| 5 | Not Gymnasium; `step(action, data=...)`, `reset()`, `observation_space`, `action_space`, `info` | **CONFIRMED exactly** | `[SRC]` |
| 6 | Official repos as listed | CONFIRMED (both cloned) | `[RUN]` |
| 7 | A separate `arc-agi-3` PyPI package exists; confirm which is current | RESOLVED — `arc-agi-3` is **v0.0.1, one release**, a stub. Current package is **`arc-agi` 0.9.9** (21 releases). `MAX_ACTIONS` lives in the Agents framework's base `Agent`, not that package. | PyPI `[DOCS]` |
| 8 | Frame is 64x64, ints 0-15 | 64x64 CONFIRMED. **0-15 NOT confirmed** — max observed 12. | `inspect_frame.txt` `[RUN]` |
| 9 | One step can return a SEQUENCE of frames | **CONFIRMED** — ft09 returns 5 frames/step mid-game; bp35 and lf52 return 2 at reset | `level_transition.txt` `[RUN]` |
| 10 | Strictly turn-based | CONSISTENT with all runs (determinism holds, nothing advances without a step) but not directly tested by idling | `[RUN]` partial |
| 11 | `state` in `{NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER}` | CONFIRMED | `[SRC]` |
| 12 | 0.9.3 renamed `score` to `levels_completed`, `win_score` to `win_levels` | CONFIRMED — old names absent from the model | `inspect_frame.txt` `[RUN]` |
| 13 | RESET, ACTION1-5, ACTION7 simple; ACTION6 complex | CONFIRMED | `action_space.txt` `[RUN]` |
| 14 | ACTION6 is a click, `data={"x","y"}`, (0,0) top-left | Mechanism CONFIRMED (**must use `data=`; `set_data()` is silently dropped**). Top-left origin only `[SRC]`, not reproducible by run. Bounds `ge=0,le=63` **not enforced**. | `action6.txt` `[RUN]` |
| 15 | ACTION7 = Undo, added 0.9.2 | Exists CONFIRMED; **Undo semantics `[DOCS]` only** | `[RUN]`/`[DOCS]` |
| 16 | Each game exposes its valid subset via `available_actions` | CONFIRMED, 10 distinct sets across 25 games. **But it is advisory — out-of-set actions are accepted.** | `action_space.txt` `[RUN]` |
| 17 | In GAME_OVER only RESET is valid; others return HTTP 400 | **CONTRADICTED** — `available_actions` is unchanged in GAME_OVER, non-RESET actions are accepted with no error and return an **empty frame list**. No HTTP layer locally. | `game_over.txt` `[RUN]` |
| 18 | 25 public games, exact ID list | **CONFIRMED — all 25 IDs match exactly**, versioned as claimed (`ft09-0d8bbf25`) | `list_games.txt` `[RUN]` |
| 19 | 55 semi-private + 55 fully-private eval games | `[UNVERIFIED]` — not reachable locally | — |
| 20 | Each game has >=6 levels; level 1 is a tutorial | >=6 CONFIRMED (`win_levels` 6-10). "Tutorial" claim `[UNVERIFIED]` | `action_space.txt` `[RUN]` |
| 21 | Effectively deterministic | **CONFIRMED** — byte-identical across repeats and across processes | `determinism.txt` `[RUN]` |
| 22 | Multi-life mechanics in some games | `[UNVERIFIED]` | — |
| 23 | Random solves a non-tutorial level <1 in 10,000 | CONSISTENT — 100,000 random steps over 5 games produced exactly **one** level completion (ft09), and zero WINs | `game_over.txt` `[RUN]` |
| 24 | >1,000-2,000 steps/s locally without rendering | **CONTRADICTED** — measured **440-820 steps/s**, mean 562 | `speed.txt` `[RUN]` |
| 25 | RHAE scoring formula, human baseline, weights, action cap | `[UNVERIFIED]` — scoring is server-side; `arc.get_scorecard()` returned only an aggregate `0.0` locally | — |
| 26 | Public-set scores never on official leaderboard | `[UNVERIFIED]` | — |
| 27 | `MyAgent` with `is_done`/`choose_action` in `agent/my_agent.py` | CONFIRMED | `[RUN]` |
| 28 | Accelerator set in `build_notebook.py` (`cpu`,`t4`,`p100`,`rtx6000`) | **CONFIRMED — all four**, `ACCELERATOR = "t4"` default | `[RUN]` |
| 29 | Internet disabled during evaluation | `[UNVERIFIED]` here, but consistent with the local-cache design | `[DOCS]` |
| 30 | Output is `submission.parquet` | CONFIRMED in `build_notebook.py` | `[RUN]` |
| 31 | Hidden set ~110 games, ~9h budget, 1/day vs 5/day | `[UNVERIFIED]`. Note the starter README explicitly says **"one of your 5 daily submissions"**, favouring 5/day. | `[DOCS]` |
| 32 | Deadlines: 26 Oct / 2 Nov / 9 Nov (brief says 8 Nov) | `[UNVERIFIED]` — not checkable from the SDK. **Confirm against the official Kaggle page before relying on it.** | — |

---

## 9. Windows notes (this machine)

**CrowdStrike Falcon deletes venv launcher stubs.** `[RUN]` `uv venv` writes a small
(~47 KB) trampoline `python.exe`; it was deleted within ~5 s everywhere outside `%TEMP%`,
leaving `pythonw.exe` untouched and every subsequent command failing with
`Access is denied (os error 5)` or "cannot find the file specified".

Workaround used — **create the venv with stdlib `venv`, not `uv venv`**:

```bash
"C:/Users/Lenovo/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe" \
    -m venv C:/Users/Lenovo/dev/arc-venv
```

`python -m venv` **copies the real 262 KB interpreter** instead of writing a stub, and
that binary is not flagged. Package installs still use `uv pip install --python <venv>`,
which is unaffected. The venv sits outside the repo at `C:\Users\Lenovo\dev\arc-venv`
(it is a rebuildable artifact and was gitignored anyway).

If this recurs on another machine, the clean fix is an IT exclusion for the project
path rather than anything done locally.

---

## 10. Open questions for the next phase

1. Is the ACTION6 origin really top-left? Needs a game where a click visibly moves a
   cursor (`ft09` or `lp85` rather than `vc33`).
2. Do colour values ever exceed 12? Sample all 25 games and take a global max.
3. What does `state == WIN` look like, and does it need `levels_completed == win_levels`?
   No run reached it.
4. Does `Arcade.make(seed=...)` change anything? All runs used the default `seed=0`.
5. Frames-per-step varies 1-5; is the sequence an animation to be subsampled, or do
   intermediate frames carry information an agent needs?
