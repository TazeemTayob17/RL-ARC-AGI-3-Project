# ARC-AGI-3 Research Notes (to verify hands-on)

Source: desk research, Sept 2026. Some facts come from official docs, some from community posts.
Anything marked [UNVERIFIED] was not from an official page. Treat everything here as a claim to check.

## Environment / SDK
- Install: `arc-agi` toolkit from PyPI. Reportedly requires Python 3.12.
- Basic use: `import arc_agi; arc = arc_agi.Arcade(); env = arc.make("ls20", render_mode="terminal")`
- Since SDK 0.9.3, environments run locally/offline; games are cached in `environment_files/` after first download.
- Online play against three.arcprize.org needs an `ARC_API_KEY`; only online play posts to scorecards/leaderboard.
- Interface is NOT Gymnasium: `env.step(action, data=...)` returns FrameDataRaw; also `env.reset()`, `env.observation_space` (last FrameData), `env.action_space` (list of GameActions), `env.info`.
- Official repos: arcprize/ARC-AGI-3-Agents (base `Agent` class, random agent: `uv run main.py --agent=random --game=ls20`), arcprize/ARCEngine (engine), arcprize/ARC-AGI-3-Kaggle-Starter (competition dev kit).
- A separate PyPI package `arc-agi-3` also exists (possibly older, with `MAX_ACTIONS`) — confirm which one is current.

## Observations
- Frame is a 64x64 grid, each cell an integer colour 0–15.
- One step can return a SEQUENCE of frames (transition animations).
- Strictly turn-based: nothing changes until the agent acts.
- `FrameData.state` ∈ {NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER}.
- SDK 0.9.3 renamed score fields (breaking change): `score` → `levels_completed`, `win_score` → `win_levels`.

## Actions
- `GameAction.RESET`, `ACTION1`–`ACTION5`, `ACTION7` are simple actions.
- `ACTION6` is a click: `data={"x": ..., "y": ...}`, (0,0) = top-left.
- `ACTION7` = Undo (added SDK 0.9.2).
- Each game exposes only its valid subset via `FrameData.available_actions`.
- In GAME_OVER, only RESET is valid; other actions return HTTP 400 (500 = server error).

## Games, levels, determinism
- 25 public games. IDs: ls20, ft09, vc33, tr87, wa30, g50t, re86, cd82, r11l, lp85, sb26, su15, tn36, cn04, lf52, s5i5, sc25, tu93, sp80, m0r0, ar25, bp35, sk48, ka59, dc22. Full IDs are versioned (e.g. `ft09-0d8bbf25`).
- 55 semi-private + 55 fully-private games exist for official evaluation (not accessible).
- Each game has multiple levels (≥6 claimed); level 1 is an easy tutorial.
- Effectively deterministic; some games have multi-life mechanics.
- Claimed: a random policy should solve a non-tutorial level less than 1 in 10,000 times.
- Claimed env speed: >1,000–2,000 steps/sec locally without rendering.

## Scoring (RHAE)
- Per level: `S = min(1.15, (human_actions / ai_actions)^2)`.
- Human baseline = upper-median human per level (changed 14 April 2026 from 2nd-best; cap raised from 1.0 to 1.15).
- Per game: weighted mean over levels with weight = level number; unsolved levels score 0 but still count.
- Total = mean over games, 0–100%.
- Per-level action cap: 5x the human baseline actions (from technical report).
- Public-set scores never go on the official leaderboard.

## Kaggle (ARC Prize 2026 – ARC-AGI-3)
- Code competition. Agent is `class MyAgent(Agent)` with `is_done(self, frames, latest_frame) -> bool` and `choose_action(self, frames, latest_frame) -> GameAction`, in `agent/my_agent.py`; starter builds the notebook.
- Accelerator set in `scripts/build_notebook.py` (`cpu`, `t4`, `p100`, `rtx6000`).
- Internet disabled during evaluation.
- Output: `submission.parquet`.
- [UNVERIFIED] Hidden set ~110 games, ~9-hour run budget, submission limit 1/day (other sources say 5/day).
- Deadlines: entry & team merge 26 Oct 2026; final submission 2 Nov; paper track 9 Nov (course brief says 8 Nov — check).
