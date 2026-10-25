"""Team Learner adapter for the experimentally selected DQN V1 baseline."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..adapter import N_SIMPLE, OPTION_ACT, SIMPLE_IDS, Observation, Transition, action_to_index, index_to_action
from ..intrinsic import NoveltyCounter, option_bonus
from ..objects import EffectMemory, build_view
from .action_bank import ActionKey
from .object_actions import OPTION_KEY, build_action_set
from .agent import DQNAgent
from .exploration import SELECTOR, StructuredExplorer
from .prior import load_prior, random_prior, state_outputs

ROOT = Path(__file__).resolve().parents[2]

OPTION_EXPLORE_PROB = 0.25      # I4: share of epsilon-exploration draws that take the frontier option when it is legal (fixed, not tuned)

SYSTEM = {
    "id": "dqn_v1_selected",
    "family": "selected_baseline",
    "learn": True,
    "n_step": 1,
    "per": False,
    "distributional": False,
    "dueling": True,
    "exploration_policy": "d14",
    "structured_probability": 1.0,
}


def _available(mask: np.ndarray) -> list[int]:
    out = [aid for i, aid in enumerate(SIMPLE_IDS) if bool(mask[i])]
    if bool(mask[N_SIMPLE:].any()):
        out.append(6)
    return sorted(out)


def _to_index(a: ActionKey) -> int:
    if a == OPTION_KEY:
        return OPTION_ACT
    return action_to_index(int(a.action_id), (int(a.x), int(a.y)) if int(a.action_id) == 6 else None)


def _from_index(idx: int) -> ActionKey:
    aid, xy = index_to_action(int(idx))
    return ActionKey(aid, *(xy or (-1, -1)))


class DQNV1:
    name = "dqn_v1"

    def __init__(self, params: dict, updates_disabled: bool = False, *, prior_path: Path, fold: int,
                 seed_offset: int = 0, deployment: bool = False):
        """The run controller supplies the checkpoint path, fold and seed offset (see src/checkpoint_resolver.py)."""
        self.p = dict(params)
        self.updates_disabled = bool(updates_disabled)
        self.device = torch.device("cpu")
        self.checkpoint, self.fold = Path(prior_path), int(fold)
        self.seed_offset, self.deployment = int(seed_offset), bool(deployment)
        self.novelty = NoveltyCounter(float(self.p.get("intrinsic_beta", 0.0)))
        self.objects_on = bool(self.p.get("object_clicks", False))               # I3: click objects, not pixels
        self.memory = EffectMemory(bool(self.p.get("effect_memory", True)) and self.objects_on)

    def _cfg(self):
        return {
            "object_actions": self.objects_on,
            "click_action_id": 6,
            "click_size": 64,
            "max_action_id": 7,
            "training": self.p,
        }

    def start_run(self, seed: int) -> None:
        self.seed = int(seed)
        self.prior_random = self.p.get("prior_mode") == "random"             # ablation: no pretrained prior
        self.prior = random_prior(self.device) if self.prior_random else load_prior(self.checkpoint, self.device)
        self.novelty.reset()
        run_seed = self.seed * 100003 + self.seed_offset
        random.seed(run_seed); np.random.seed(run_seed & 0xFFFFFFFF); torch.manual_seed(run_seed)
        self.explorer = StructuredExplorer()
        self.system = dict(SYSTEM)
        if self.updates_disabled:
            self.system["learn"] = False
        self.agent = DQNAgent(130, self.system, self._cfg(), run_seed + 17, self.device)
        self.pending = None
        self._feat_cache = None
        self._set_cache = None
        self.memory.reset(self.seed)
        self.option_rng = random.Random(f"dqn-frontier-{self.seed}")     # touched only when the option is legal
        self.n_option_explore = 0
        self.last_label = None
        self.since_progress = 0

    def _features(self, obs: Observation):
        """Pure function of the observation. The outcome of one step is the input of the next, so the last result is kept."""
        key = (obs.frame.tobytes(), obs.prev_frame.tobytes(), obs.mask.tobytes(), obs.progress, obs.high_water)
        if self._feat_cache is not None and self._feat_cache[0] == key:
            return self._feat_cache[1]
        available = _available(obs.mask)
        emb, simple, simple_logits, a6_logits = state_outputs(
            self.prior, obs.frame, obs.prev_frame, available, self.device
        )
        if self.prior_random:
            simple_logits = np.zeros_like(simple_logits)         # a random head has no preference worth following
        state = np.concatenate([
            emb.reshape(-1),
            np.asarray([obs.progress / 5.0, obs.high_water / 5.0], dtype=np.float32),
        ]).astype(np.float32)
        out = (state, emb, simple, simple_logits, a6_logits, available)
        self._feat_cache = (key, out)
        return out

    def _action_set(self, obs: Observation, available):
        """Choice set of an observation. The outcome of one step is the input of the next, so the last set is kept:
        the set the learner trains on is the set it acted on."""
        key = (obs.frame.tobytes(), obs.progress, obs.mask.tobytes(), self.memory.n_recorded, obs.option_ok)   # memory changed -> rebuild
        if self._set_cache is not None and self._set_cache[0] == key:
            return self._set_cache[1]
        view = build_view(obs.frame, obs.progress, self.memory) if 6 in available else None
        oset = build_action_set(self.agent.bank, available, view, self.memory, obs.option_ok)
        self._set_cache = (key, oset)
        return oset

    def act(self, obs: Observation) -> int:
        state, emb, simple, simple_logits, a6_logits, available = self._features(obs)

        oset = self._action_set(obs, available) if self.objects_on else None

        def explore_fn(rng, action_index):
            if obs.option_ok and self.option_rng.random() < OPTION_EXPLORE_PROB:      # I4: the option is part of exploration too
                self.n_option_explore += 1
                return OPTION_KEY, "option"
            return self.explorer.choose(rng, available, obs.frame, simple, simple_logits, a6_logits, oset)

        action, label = self.agent.act(state, available, explore_fn=explore_fn, cand_set=oset, option_ok=obs.option_ok)
        if action != OPTION_KEY:
            self.explorer.observe(obs.frame, action)  # counts greedy actions too, matching the selected experiment
        idx = _to_index(action)
        self.pending = (state, tuple(available), action, idx, oset)
        self.last_label = label
        return idx

    def observe(self, t: Transition) -> None:
        if self.pending is None:
            raise RuntimeError("observe() called without a pending V1 action")
        state, available, action, idx, oset = self.pending
        if int(t.action) != int(idx):
            raise AssertionError(("V1 pending action mismatch", idx, t.action))
        if oset is not None:
            self._record_effect(oset, action, t)

        reward = float(t.reward)
        if t.info.get("state") == "WIN" and reward == 0.0:
            reward = 1.0  # exact selected-baseline reward convention

        next_set = None
        if t.terminal:
            next_state = state.copy()
            next_available = tuple()
        else:
            next_state, _, _, _, _, nav = self._features(t.next_obs)
            next_available = tuple(nav)
            if oset is not None:
                next_set = self._action_set(t.next_obs, nav)

        bonus = option_bonus(self.novelty, t, float(self.p.get("gamma", 0.99)))   # I1: training signal only, kept apart from the extrinsic reward
        self.agent.observe(state, available, action, reward + bonus, next_state, next_available, bool(t.terminal),
                           steps=int(t.info.get("k", 1)),
                           **({} if oset is None else {"cand": oset.rows, "chosen": oset.row_of(action),
                                                       "next_cand": None if next_set is None else next_set.rows}),
                           **({"next_opt": t.next_obs.option_ok} if oset is None and t.next_obs.option_ok else {}))
        if reward > 0 or t.info.get("state") == "WIN":
            self.since_progress = 0
        else:
            self.since_progress += 1
        self.pending = None

    def _record_effect(self, oset, action: ActionKey, t: Transition) -> None:
        """Tell the effect memory what the object click did (frame changed or not, level up, game over)."""
        c = oset.cands[oset.row_of(action)]
        if c is None:
            return
        self.memory.record(oset.skey, c, bool(t.info.get("frame_changed")), level_up=t.reward > 0,
                           game_over=bool(t.terminal and t.info.get("state") == "GAME_OVER"))

    def finish(self, final_obs: Optional[Observation]) -> None:
        self.agent.finish()

    def diagnostics(self) -> dict:
        return {
            "algo": "dqn",
            "version": "V1",
            "selected_system": SYSTEM["id"],
            "fold": self.fold,
            "prior_checkpoint": "none (random frozen encoder)" if self.prior_random else str(self.checkpoint.relative_to(ROOT)),
            "selector": SELECTOR["id"],
            "deployment_checkpoint": self.deployment,
            "intrinsic": self.novelty.summary(),
            "object_clicks": self.objects_on,
            "effect_memory": self.memory.summary() if self.objects_on else None,
            "option_explore_draws": self.n_option_explore,
            **self.agent.summary(),
        }

    def model_info(self) -> dict:
        return {
            "architecture": "selected_onehot_stack2_vector_effect+dueling_action_value",
            "parameters": sum(p.numel() for p in self.agent.online.parameters()),
            "frozen_prior_parameters": sum(p.numel() for p in self.prior.parameters()),
            "action_representation": "shared 10-D action vector over the legal action set",
            "auxiliary_objectives": ["frozen binary action-effect prior"],
            "memory": "downsample16 state-action counts inside structured epsilon exploration",
            "checkpoint_mode": ("random frozen encoder, no pretraining" if self.prior_random else
                                "public cross-fold, seed-11 prior (faithful final-ablation reproduction)"),
        }

    def state_dict(self) -> dict:
        return {
            "online": self.agent.online.state_dict(),
            "target": self.agent.target.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "action_count": self.agent.action_count,
            "stats": asdict(self.agent.stats),
            "replay_data": self.agent.replay.data,
            "replay_priorities": self.agent.replay.priorities,
            "replay_pos": self.agent.replay.pos,
            "epsilon_rng": self.agent.epsilon_rng.getstate(),
            "explore_rng": self.agent.explore_rng.getstate(),
            "replay_rng": self.agent.replay_rng.getstate(),
            "pair_counts": dict(self.explorer.pair_counts),
        }

    def load_state_dict(self, d: dict) -> None:
        self.agent.online.load_state_dict(d["online"])
        self.agent.target.load_state_dict(d["target"])
        self.agent.optimizer.load_state_dict(d["optimizer"])
        self.agent.action_count = int(d["action_count"])
        for k, v in d["stats"].items():
            setattr(self.agent.stats, k, v)
        self.agent.replay.data = list(d["replay_data"])
        self.agent.replay.priorities[:] = d["replay_priorities"]
        self.agent.replay.pos = int(d["replay_pos"])
        self.agent.epsilon_rng.setstate(d["epsilon_rng"])
        self.agent.explore_rng.setstate(d["explore_rng"])
        self.agent.replay_rng.setstate(d["replay_rng"])
        self.explorer.pair_counts.update(d["pair_counts"])
