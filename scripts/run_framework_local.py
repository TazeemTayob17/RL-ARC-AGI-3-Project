"""Run kaggle/my_agent.py through the real ARC-AGI-3-Agents Agent.main() loop, in-process, on local games.

Mirrors the Kaggle gateway loop (vendor/kaggle-starter/scripts/play_local.py) but loads our agent and
writes a JSON result with the official SDK scorecard. Needs vendor/ARC-AGI-3-Agents (gitignored clone).

  python scripts/run_framework_local.py --games ls20,ft09 --max-actions 200 --mode type_uniform
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "vendor" / "ARC-AGI-3-Agents"
# Same slimming the Kaggle sample does: avoid importing every LLM agent. Written as UTF-8 (Windows-safe).
(FW / "agents" / "__init__.py").write_text(
    '"""Slimmed: random agent only."""\nfrom dotenv import load_dotenv\nfrom .agent import Agent, Playback\n'
    'from .swarm import Swarm\nfrom .templates.random_agent import Random\nload_dotenv()\n'
    'AVAILABLE_AGENTS = {"random": Random}\n', encoding="utf-8")
sys.path[:0] = [str(FW), str(ROOT / "kaggle")]

import arc_agi
from arc_agi import OperationMode

p = argparse.ArgumentParser()
p.add_argument("--games", default=None, help="comma-separated short ids; default all 25")
p.add_argument("--max-actions", type=int, default=200)
p.add_argument("--mode", default="type_uniform", choices=["type_uniform", "flat_uniform"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
a = p.parse_args()

spec = importlib.util.spec_from_file_location("my_agent", ROOT / "kaggle" / "my_agent.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.MODE, mod.SEED = a.mode, a.seed
mod.MyAgent.MAX_ACTIONS = a.max_actions

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
if a.games:
    ids = [g for g in ids if g in set(a.games.split(","))]
per_game, t0 = [], time.time()
for g in ids:
    env = arc.make(g)
    ag = mod.MyAgent(card_id="local", game_id=g, agent_name=f"random.{a.mode}", ROOT_URL="http://localhost",
                     record=False, arc_env=env, tags=["local"])
    ag.main()
    f = ag.frames[-1]
    per_game.append(dict(game=g, state=f.state.value, levels_completed=f.levels_completed, loop_iterations=ag.action_counter))
card = arc.get_scorecard()
res = dict(mode=a.mode, seed=a.seed, max_actions=a.max_actions, wall_s=round(time.time() - t0, 1),
           sdk_score=card.score, total_actions=card.total_actions, total_levels_completed=card.total_levels_completed,
           per_game=per_game, scorecard=json.loads(card.model_dump_json()))
print(json.dumps({k: v for k, v in res.items() if k != "scorecard"}, indent=1))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
