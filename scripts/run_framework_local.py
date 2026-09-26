"""Run a repository agent through the real ARC-AGI-3-Agents Agent.main() loop on local games.

Examples:
  python scripts/run_framework_local.py --agent random_type --games ls20,ft09 --max-actions 200
  python scripts/run_framework_local.py --agent dqn_v0 --games ls20,ft09 --max-actions 600 --seed 5209
  python scripts/run_framework_local.py --agent dqn_v1 --games ls20,ft09 --max-actions 600 --seed 5209
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "vendor" / "ARC-AGI-3-Agents"
(FW / "agents" / "__init__.py").write_text(
    '\"\"\"Slimmed: random agent only.\"\"\"\nfrom dotenv import load_dotenv\nfrom .agent import Agent, Playback\n'
    'from .swarm import Swarm\nfrom .templates.random_agent import Random\nload_dotenv()\n'
    'AVAILABLE_AGENTS = {"random": Random}\n', encoding="utf-8")
sys.path[:0] = [str(FW), str(ROOT / "kaggle"), str(ROOT)]

import arc_agi
from arc_agi import OperationMode

p = argparse.ArgumentParser()
p.add_argument("--games", default=None, help="comma-separated short ids; default all 25")
p.add_argument("--max-actions", type=int, default=600)
p.add_argument("--agent", default="dqn_v0", choices=["random_type", "random_flat", "dqn_v0", "dqn_v1", "ppo_v0"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
a = p.parse_args()

if a.agent.startswith("random_"):
    path = ROOT / "kaggle" / "my_agent.py"
    spec = importlib.util.spec_from_file_location("my_agent_eval", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.MODE = "type_uniform" if a.agent == "random_type" else "flat_uniform"
    mod.SEED = a.seed
    mod.MyAgent.MAX_ACTIONS = a.max_actions
else:
    version = "V1" if a.agent == "dqn_v1" else "V0"
    algo = "ppo" if a.agent == "ppo_v0" else "dqn"
    cfg = {"algo": algo, "version": version, "game": "", "budget": a.max_actions + 1, "seed": a.seed}
    os.environ["ARC_LEARNER_CONFIG"] = json.dumps(cfg)
    path = ROOT / "kaggle" / "learner_agent.py"
    spec = importlib.util.spec_from_file_location("learner_agent_eval", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.MyAgent.MAX_ACTIONS = a.max_actions

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
if a.games:
    wanted = set(a.games.split(","))
    ids = [g for g in ids if g in wanted]

per_game, t0 = [], time.time()
for g in ids:
    env = arc.make(g)
    ag = mod.MyAgent(card_id="local", game_id=g, agent_name=a.agent, ROOT_URL="http://localhost",
                     record=False, arc_env=env, tags=["local", a.agent])
    ag.main()
    f = ag.frames[-1]
    per_game.append(dict(game=g, state=f.state.value, levels_completed=f.levels_completed,
                         loop_iterations=ag.action_counter))

card = arc.get_scorecard()
res = dict(agent=a.agent, seed=a.seed, max_actions=a.max_actions, wall_s=round(time.time() - t0, 1),
           sdk_score=card.score, total_actions=card.total_actions, total_levels_completed=card.total_levels_completed,
           per_game=per_game, scorecard=json.loads(card.model_dump_json()))
print(json.dumps({k: v for k, v in res.items() if k != "scorecard"}, indent=1))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
