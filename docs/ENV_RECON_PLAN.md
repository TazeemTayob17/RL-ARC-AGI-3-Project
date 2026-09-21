# ARC-AGI-3 environment recon & verification

## Context

This is the groundwork phase of an RL honours project targeting ARC-AGI-3 (ARC Prize
2026). Before any learning agent is written, we need a trustworthy picture of what the
environment actually *is*: observation format, action space, reset/termination
semantics, determinism, and throughput. Those four facts determine almost every
downstream design choice (replay buffer shape, action head, episode boundaries,
whether we can afford model-free sample counts at all).

`docs/RESEARCH_NOTES.md` already exists and contains ~40 desk-research claims, some
from official docs and some from community posts. It explicitly asks to be treated as
unverified. The deliverable is `docs/ENV_FINDINGS.md`: the same territory, but every
fact tagged with how we know it, and every research claim marked
confirmed / contradicted / unconfirmable.

**Explicitly out of scope:** no learning agents, no training code, no reward shaping.
Random-policy rollouts only, used purely as instrumentation.

## Decisions already made

- **Interpreter: install `uv`, pin Python 3.12.** No 3.12 exists on this machine
  (3.14 is default `python`, plus 3.13 and an Anaconda 3.9). `uv` fetches a managed
  CPython 3.12 without touching system PATH, and matches both official repos
  (`.python-version` = `3.12`) and the Kaggle image.

## Recon already completed (read-only, this session)

| Finding | Evidence |
|---|---|
| `arc-agi` 0.9.9 on PyPI, `requires_python = ">=3.12"` | PyPI JSON API |
| Deps: `arcengine>=0.9.3`, flask, matplotlib, pydantic, python-dotenv, pillow, requests | PyPI JSON API |
| `arcengine` 0.9.3, pure-python wheel, needs numpy + pydantic | PyPI JSON API |
| `arc-agi-3` is v0.0.1 with a single release — **not** the current package | PyPI JSON API |
| ARC-AGI-3-Agents pins `.python-version` = `3.12`, `requires-python = ">=3.12"` | raw.githubusercontent |
| Changelog 0.9.3 confirms `score`→`levels_completed`, `win_score`→`win_levels` | repo README |
| Changelog 0.9.2 confirms `available_actions` on FrameData and `ACTION7` | repo README |
| Kaggle starter tree: `agent/my_agent.py`, `scripts/build_notebook.py`, `scripts/play_local.py`, `scripts/slim_framework.py`, `Makefile` | GitHub trees API |
| `build_notebook.py` `_ACCELERATORS` = t4 / p100 / rtx6000 — **no `cpu` key** | raw file grep |
| Kaggle starter README says "one of your **5** daily submissions" | raw README |

## Known Windows gotchas (must be handled, not worked around silently)

1. **The starter `Makefile` is POSIX-only.** It uses `PYTHON ?= python3.12`,
   `$(VENV)/bin/python`, `cat`, `chmod`. On Windows the venv layout is
   `.venv/Scripts/python.exe`. We will **not** run `make`; we invoke
   `scripts/play_local.py` directly and *document* what each make target does.
2. **"Offline" is offline-after-first-fetch.** `play_local.py`'s own docstring: game
   source is "downloaded on first call into `./environment_files/` and cached". So the
   first run needs internet; genuine offline behaviour is a separate test (step 6).
3. **`ARC_API_KEY`** is only needed for online play against three.arcprize.org. Local
   `OperationMode.NORMAL` should need none — that itself is a claim to verify.

## Steps

### 0. Commit this plan to the repo
Write this document verbatim to `docs/ENV_RECON_PLAN.md` (alongside the existing
`docs/RESEARCH_NOTES.md`) so the plan is version-controlled with the work it describes.
Do this first, before any install, so it lands in the initial commit in step 2.

### 1. Toolchain
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
```
Report `uv --version` and `uv python list`. If the install script is blocked by
execution policy, fall back to the python.org 3.12 installer and say so.

### 2. Project repo + venv
In `C:\Users\Lenovo\Documents\RL - Hons\Project` (currently holds only `docs/`):
- `git init`, add a Python `.gitignore` (must exclude `.venv/`, `environment_files/`,
  `vendor/`, `*.env`).
- `uv venv --python 3.12 .venv`
- `uv pip install arc-agi`
- `uv pip freeze > requirements.txt` — exact pinned versions of the full transitive
  closure, which is what the task asks for.
- Record the resolved `arc-agi` / `arcengine` / `numpy` versions in the findings doc,
  since every claim below is version-specific.

### 3. Clone and read the official repos
Clone into `vendor/` (gitignored, so they don't pollute our history):
- `vendor/ARC-AGI-3-Agents` — read `agents/agent.py` (base `Agent`, the `main()` loop,
  `is_done`/`choose_action` contract), `agents/structs.py` (**this is where
  `FrameData`, `GameAction`, `GameState` are defined** — the primary source for the
  step-5 field-name questions), and `main.py`.
- `vendor/ARC-AGI-3-Kaggle-Starter` — read `agent/my_agent.py`,
  `scripts/build_notebook.py`, `scripts/play_local.py`, `scripts/slim_framework.py`,
  `Makefile`.

Write the packaging summary: how `my_agent.py` is spliced into a notebook, why
`slim_framework.py` strips the langgraph/smolagents imports, what each make target
does and its Windows equivalent, and how `submission.parquet` is produced.

### 4. Probe scripts (this is the real work)
Write throwaway instrumentation under `scripts/probe/` — these are diagnostics, not
agents, and each one prints raw evidence we paste into the findings doc.

- `list_games.py` — `Arcade(operation_mode=OperationMode.NORMAL).get_environments()`;
  print count + every `game_id`. Directly tests the "25 public games" and the exact ID
  list in the notes.
- `inspect_frame.py` — reset one game, then dump: `type(frame)`, `dir(frame)`,
  `frame.model_fields` if pydantic, `np.array(frame.frame).shape`, observed min/max
  cell values, `len(frame.frame)` (**the multi-frames-per-step claim**),
  `frame.state`, `frame.available_actions`, and whether `levels_completed`/`win_levels`
  or the old `score`/`win_score` exist. Introspect, don't assume the note is right.
- `action_space.py` — enumerate `GameAction` members; confirm ACTION7 exists; print
  `available_actions` for **every** game at reset to show how it varies; send an
  ACTION6 with coordinates and record the exact call signature that works
  (`action.set_data({"x":..,"y":..})` vs `env.step(action, data=...)` — the notes
  assume the latter; verify against `structs.py`).
- `determinism.py` — from a fresh reset, replay an identical fixed action sequence
  twice (and a third time in a fresh process), hash each frame, compare. Report
  per-step divergence index if any.
- `speed.py` — N steps, no rendering, wall-clock steps/sec; report with and without
  the frame→numpy conversion so the number is honest about what it measures.
- `game_over.py` — drive a game to GAME_OVER (random policy, long budget), then try a
  non-RESET action and record what actually happens **locally** (exception type? no-op?
  — the notes' "HTTP 400" is an online-API claim and may not apply offline), then RESET
  and confirm recovery. Also capture a level transition and show exactly which field
  changes (`levels_completed` increment, `state` value at the boundary).

### 5. Random-agent runs
Run the framework's random agent on **at least 3 games** (`ls20`, `vc33`, plus one
more from the discovered list) with a bounded step cap. Prefer
`vendor/ARC-AGI-3-Kaggle-Starter/scripts/play_local.py` since it already wires
`MyAgent` into `Agent.main()`; fall back to `uv run main.py --agent=random --game=ls20`
inside the Agents repo if the starter's POSIX assumptions bite. Save raw logs to
`docs/evidence/` and cite them by filename in the findings doc.

### 6. Offline check
After the first successful run has populated `environment_files/`, re-run one game with
networking disabled (or with `ONLINE_ONLY` unset and the machine's adapter off, if you
allow it — otherwise document that this test was skipped rather than claim it passed).

### 7. Write `docs/ENV_FINDINGS.md`
Structure:
1. Setup & exact versions (from `requirements.txt`).
2. Observation format — shape, colour range, frames-per-step, real `FrameData` field
   names.
3. Action space — enum members, per-game `available_actions` table, ACTION6 signature,
   ACTION7.
4. Reset / GAME_OVER / level progression.
5. Determinism.
6. Speed.
7. Kaggle packaging summary.
8. **Claim-by-claim verdict table against `docs/RESEARCH_NOTES.md`** — one row per
   claim, verdict ∈ {CONFIRMED-BY-RUN, CONFIRMED-FROM-DOCS, CONTRADICTED,
   UNCONFIRMABLE-HERE}, with the evidence file or command for each.

Every factual line gets an inline tag: `[RUN]` (observed this session, with the
command), `[DOCS]` (repo/PyPI/official docs, with the URL), or `[UNVERIFIED]`.

Claims already flagged as likely CONTRADICTED or UNCONFIRMABLE, to be settled by the
runs:
- "requires Python 3.12" → actually `>=3.12` (3.13/3.14 not excluded by metadata).
- Kaggle accelerator list includes `cpu` → `_ACCELERATORS` has only t4/p100/rtx6000.
- Submission limit 1/day → starter README says 5.
- "GAME_OVER → HTTP 400" → no HTTP layer in local mode; expect different semantics.
- Scoring/RHAE, hidden-set size, run budget, deadlines → UNCONFIRMABLE from a local
  SDK; mark as docs-only, do not launder into "confirmed".

## Verification

- `uv run python -c "import arc_agi, arcengine; print(arc_agi.__version__)"` succeeds
  in the venv.
- `requirements.txt` has pinned `==` versions and a fresh `uv pip install -r` resolves.
- Random agent completes ≥3 distinct games and logs land in `docs/evidence/`.
- Every numbered section of `ENV_FINDINGS.md` cites at least one `[RUN]` command, and
  the verdict table has a row for every claim in `RESEARCH_NOTES.md` with no blanks.
- `git status` clean apart from intended files; `.venv/`, `vendor/`,
  `environment_files/` all ignored.

## Risks

- ARC Prize may have changed the local-execution API between 0.9.3 and 0.9.9; if
  `Arcade`/`OperationMode` differ from `play_local.py`'s usage, trust the installed
  package's own source and note the drift.
- Reaching a genuine level completion or GAME_OVER under a random policy may need a
  large step budget (the notes themselves claim random solves a non-tutorial level
  <1/10,000). If level-2+ progression can't be observed, report it as
  UNCONFIRMABLE with the step count attempted rather than inferring.
