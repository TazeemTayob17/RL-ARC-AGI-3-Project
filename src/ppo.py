"""PPO (Schulman et al. 2017) with GAE (Schulman et al. 2016): one live environment, sequential rollouts.

Provisional V0 policy factorisation (plan 5.3):
    log p(a) = log p(type) [+ log p(location | ACTION6) for clicks]
    entropy  = H(type) + p(ACTION6) * H(location | ACTION6)
Masks stored at collection time are reused in the update (they live in the stored Observation).
GAE never propagates across a terminal boundary; a rollout cutoff bootstraps from the last valid observation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .adapter import N_SIMPLE, N_TYPES, CLICK_TYPE, OPTION_ACT, Observation, Transition, type_mask
from .intrinsic import NoveltyCounter, option_bonus
from .model import ActorCritic, ObsBatch, collate, model_info
from .objects import K_MAX, EffectMemory, build_view

NEG = -1e9


def _type_terms(b: ObsBatch, actions: torch.Tensor, opt_ok):
    """Type mask, click flags and action types. With `opt_ok` (I4) the frontier option is a further action type."""
    tm = torch.cat([b.mask[:, :N_SIMPLE], b.mask[:, N_SIMPLE:].any(1, keepdim=True)], 1)
    if opt_ok is None:
        is_click = actions >= N_SIMPLE
        return tm, is_click, torch.where(is_click, torch.full_like(actions, CLICK_TYPE), actions)
    tm = torch.cat([tm, opt_ok[:, None]], 1)
    is_opt = actions == OPTION_ACT
    is_click = (actions >= N_SIMPLE) & ~is_opt
    typ = torch.where(is_click, torch.full_like(actions, CLICK_TYPE), actions)
    return tm, is_click, torch.where(is_opt, torch.full_like(actions, N_TYPES), typ)


def evaluate_actions(net: ActorCritic, b: ObsBatch, actions: torch.Tensor, opt_ok=None):
    """Joint log-prob, entropy and value for stored actions under the current policy (shared by act and update)."""
    tl, cl, v = net(b)
    tm, is_click, typ = _type_terms(b, actions, opt_ok)
    lp_t = F.log_softmax(tl.masked_fill(~tm, NEG), 1)
    p_t = lp_t.exp()
    lp_c = F.log_softmax(cl.masked_fill(~b.mask[:, N_SIMPLE:], NEG), 1)
    p_c = lp_c.exp()
    loc = (actions - N_SIMPLE).clamp(min=0, max=4095)
    logp = lp_t.gather(1, typ[:, None]).squeeze(1)
    logp = logp + torch.where(is_click, lp_c.gather(1, loc[:, None]).squeeze(1), torch.zeros_like(logp))
    ent = -(p_t * lp_t).sum(1) + p_t[:, CLICK_TYPE] * -(p_c * lp_c).sum(1)
    return logp, ent, v


def object_logits(cl: torch.Tensor, obj_index: torch.Tensor) -> torch.Tensor:
    """I3 pointer head: one logit per object = mean of the click head's pixel logits inside the object, plus a final
    random-pixel slot (mean over the whole board). cl [B, 4096]; obj_index long [B, 4096] (-1 = no object)."""
    idx = torch.where(obj_index < 0, torch.full_like(obj_index, K_MAX), obj_index)
    sums = cl.new_zeros(cl.shape[0], K_MAX + 1).scatter_add(1, idx, cl)
    cnt = cl.new_zeros(cl.shape[0], K_MAX + 1).scatter_add(1, idx, torch.ones_like(cl))
    return torch.cat([sums[:, :K_MAX] / cnt[:, :K_MAX].clamp(min=1.0), cl.mean(1, keepdim=True)], 1)


def evaluate_object_actions(net: ActorCritic, b: ObsBatch, actions: torch.Tensor, slots: torch.Tensor,
                            obj_index: torch.Tensor, valid: torch.Tensor, opt_ok=None):
    """Like evaluate_actions, with the click location replaced by the choice among objects. `valid` [B, K_MAX + 1]
    is the choice set stored at collection time (memory-dropped candidates are False), so log-probs stay consistent."""
    tl, cl, v = net(b)
    tm, is_click, typ = _type_terms(b, actions, opt_ok)
    lp_t = F.log_softmax(tl.masked_fill(~tm, NEG), 1)
    p_t = lp_t.exp()
    lp_s = F.log_softmax(object_logits(cl, obj_index).masked_fill(~valid, NEG), 1)
    p_s = lp_s.exp()
    logp = lp_t.gather(1, typ[:, None]).squeeze(1)
    logp = logp + torch.where(is_click, lp_s.gather(1, slots[:, None]).squeeze(1), torch.zeros_like(logp))
    ent = -(p_t * lp_t).sum(1) + p_t[:, CLICK_TYPE] * -(p_s * lp_s).sum(1)
    return logp, ent, v


class PPO:
    name = "ppo"

    def __init__(self, arch: dict, params: dict, updates_disabled: bool = False):
        self.arch, self.p, self.updates_disabled = arch, params, updates_disabled
        self.objects_on = bool(params.get("object_clicks", False))              # I3: click objects, not pixels
        self.memory = EffectMemory(bool(params.get("effect_memory", True)) and self.objects_on)
        self.option_on = params.get("frontier_mode") == "learned"                  # I4: frontier option as an action type

    def start_run(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.net = ActorCritic({**self.arch, "extra_types": 1} if self.option_on else self.arch)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"], eps=1e-5)
        self.buf: list = []
        self.novelty = NoveltyCounter(float(self.p.get("intrinsic_beta", 0.0)))
        self._pending = None
        self.n_act = self.rollout_updates = self.grad_steps = self.partial_rollouts = 0
        self.ema: dict = {}
        self.memory.reset(seed)
        self.n_cand = self.n_kept = self.n_click_acts = self.n_option_picks = 0

    # ---------------------------------------------------------------- acting
    def act(self, obs: Observation) -> int:
        self.n_act += 1
        with torch.no_grad():
            tl, cl, v = self.net(collate([obs]))
            tm = torch.from_numpy(type_mask(obs.mask))
            if self.option_on:
                tm = torch.cat([tm, torch.tensor([bool(obs.option_ok)])])
            lp_t = F.log_softmax(tl[0].masked_fill(~tm, NEG), 0)
            typ = int(Categorical(logits=lp_t).sample())
            extra = None
            if self.objects_on:
                view = build_view(obs.frame, obs.progress, self.memory) if obs.mask[N_SIMPLE:].any() else None
                obj_index = view.obj_index.astype(np.int64) if view else np.full(4096, -1, dtype=np.int64)
                valid = np.zeros(K_MAX + 1, dtype=bool)
                valid[K_MAX] = True
                if view:
                    valid[view.keep] = True
                extra = [obj_index, valid, 0, view]
            if typ < N_SIMPLE:
                idx, lp = typ, lp_t[typ]
            elif typ == N_TYPES:                                    # I4: the frontier option (needs option_on)
                idx, lp = OPTION_ACT, lp_t[typ]
                self.n_option_picks += 1
            elif self.objects_on:
                lp_s = F.log_softmax(object_logits(cl, torch.from_numpy(extra[0])[None])[0]
                                     .masked_fill(~torch.from_numpy(extra[1]), NEG), 0)
                slot = int(Categorical(logits=lp_s).sample())
                view = extra[3]
                x, y = view.slot_xy if slot == K_MAX else (view.cands[slot].x, view.cands[slot].y)
                extra[2] = slot
                idx, lp = N_SIMPLE + y * 64 + x, lp_t[CLICK_TYPE] + lp_s[slot]
                self.n_click_acts += 1
                self.n_cand += len(view.cands)
                self.n_kept += len(view.keep)
            else:
                lp_c = F.log_softmax(cl[0].masked_fill(~torch.from_numpy(obs.mask[N_SIMPLE:]), NEG), 0)
                loc = int(Categorical(logits=lp_c).sample())
                idx, lp = N_SIMPLE + loc, lp_t[CLICK_TYPE] + lp_c[loc]
        self._pending = (obs, idx, float(lp), float(v[0]), extra)
        return idx

    def observe(self, t: Transition) -> None:
        obs, idx, lp, v, extra = self._pending
        assert t.action == idx, "observe() must follow the matching act()"
        if extra is not None and N_SIMPLE <= idx < OPTION_ACT and extra[2] < K_MAX:
            c = extra[3].cands[extra[2]]
            self.memory.record(extra[3].skey, c, bool(t.info.get("frame_changed")), level_up=t.reward > 0,
                               game_over=bool(t.terminal and t.info.get("state") == "GAME_OVER"))
        self.buf.append((obs, idx, lp, v, t.reward, t.terminal, option_bonus(self.novelty, t, self.p["gamma"]),
                         int(t.info.get("k", 1)), extra, bool(obs.option_ok)))   # I1 bonus kept apart; k = primitive steps (option)
        if len(self.buf) >= self.p["rollout"]:
            self._end_rollout(0.0 if t.terminal else self._value(t.next_obs))

    def finish(self, final_obs: Optional[Observation]) -> None:
        """Process a partial final rollout safely (bootstrap only if its last transition was non-terminal)."""
        if not self.buf:
            return
        self.partial_rollouts += 1
        if len(self.buf) < self.p["minibatches"]:
            self.buf.clear()
            return
        last_terminal = self.buf[-1][5]
        self._end_rollout(0.0 if (last_terminal or final_obs is None) else self._value(final_obs))

    def _value(self, obs: Observation) -> float:
        with torch.no_grad():
            return float(self.net(collate([obs]))[2][0])

    # ---------------------------------------------------------------- learning
    def _end_rollout(self, last_value: float) -> None:
        buf, self.buf = self.buf, []
        if self.updates_disabled:
            return
        p = self.p
        obs = [b[0] for b in buf]
        actions = torch.tensor([b[1] for b in buf], dtype=torch.long)
        old_lp = torch.tensor([b[2] for b in buf], dtype=torch.float32)
        values = np.array([b[3] for b in buf], dtype=np.float32)
        rewards = np.array([b[4] + b[6] for b in buf], dtype=np.float32)   # extrinsic + I1 bonus (0 when beta = 0)
        dones = np.array([b[5] for b in buf], dtype=np.float32)
        ks = np.array([b[7] for b in buf])
        oo = torch.tensor([b[9] for b in buf], dtype=torch.bool) if self.option_on else None
        if self.objects_on:
            oi = torch.from_numpy(np.stack([b[8][0] for b in buf]))
            ov = torch.from_numpy(np.stack([b[8][1] for b in buf]))
            os_ = torch.tensor([b[8][2] for b in buf], dtype=torch.long)
        gamma = p["gamma"] if (ks == 1).all() else p["gamma"] ** ks.astype(np.float32)   # gamma**k per step (options)
        adv = compute_gae(rewards, values, dones, last_value, gamma, p["gae_lambda"])
        ret = torch.from_numpy(adv + values)
        adv = torch.from_numpy(adv)
        n = len(buf)
        mb = int(np.ceil(n / p["minibatches"]))
        for _ in range(p["epochs"]):
            perm = torch.randperm(n)
            for chunk in perm.split(mb):
                ci = chunk.tolist()
                if self.objects_on:
                    logp, ent, v = evaluate_object_actions(self.net, collate([obs[i] for i in ci]), actions[chunk],
                                                           os_[chunk], oi[chunk], ov[chunk], None if oo is None else oo[chunk])
                else:
                    logp, ent, v = evaluate_actions(self.net, collate([obs[i] for i in ci]), actions[chunk],
                                                    None if oo is None else oo[chunk])
                a = adv[chunk]
                if len(ci) > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old_lp[chunk]).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - p["clip"], 1 + p["clip"]) * a).mean()
                vf = 0.5 * ((v - ret[chunk]) ** 2).mean()
                loss = pg + p["vf_coef"] * vf - p["ent_coef"] * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), p["grad_clip"])
                self.opt.step()
                self.grad_steps += 1
                self._ema(loss=float(loss.detach()), pg=float(pg.detach()), vf=float(vf.detach()), entropy=float(ent.detach().mean()),
                          kl=float((old_lp[chunk] - logp.detach()).mean()),
                          clipfrac=float(((ratio - 1).abs() > p["clip"]).float().mean()))
        self.rollout_updates += 1

    def _ema(self, **kw) -> None:
        for k, v in kw.items():
            self.ema[k] = v if k not in self.ema else 0.98 * self.ema[k] + 0.02 * v

    # ---------------------------------------------------------------- reporting / checkpointing
    def diagnostics(self) -> dict:
        return {"algo": "ppo", "actions_seen": self.n_act, "rollout_update_events": self.rollout_updates,
                "gradient_steps": self.grad_steps, "partial_rollouts": self.partial_rollouts,
                "updates_disabled": self.updates_disabled, "intrinsic": self.novelty.summary(),
                "object_clicks": self.objects_on,
                "effect_memory": self.memory.summary() if self.objects_on else None,
                "frontier_option": self.option_on, "option_type_picks": self.n_option_picks,
                "mean_candidates": self.n_cand / max(self.n_click_acts, 1),
                "mean_kept": self.n_kept / max(self.n_click_acts, 1),
                **{f"{k}_ema": v for k, v in self.ema.items()}}

    def model_info(self) -> dict:
        return model_info(self.net, self.arch)

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict(),
                "counters": {"n_act": self.n_act, "rollout_updates": self.rollout_updates,
                          "grad_steps": self.grad_steps, "partial_rollouts": self.partial_rollouts}}

    def load_state_dict(self, d: dict) -> None:
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        for k, v in d["counters"].items():
            setattr(self, k, v)


def compute_gae(rewards, values, dones, last_value, gamma, lam) -> np.ndarray:
    """GAE that never crosses a terminal boundary: done[t] cuts both the bootstrap and the trace.
    `gamma` is a scalar, or one discount per step (gamma**k for an option of k primitive steps)."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        nxt = last_value if t == n - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        g = gamma[t] if isinstance(gamma, np.ndarray) else gamma
        delta = rewards[t] + g * nxt * nonterm - values[t]
        gae = delta + g * lam * nonterm * gae
        adv[t] = gae
    return adv
