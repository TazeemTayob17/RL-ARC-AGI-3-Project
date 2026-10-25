"""Improvement 4: frontier-directed exploration option (docs/further-improvements.md, I4; Go-Explore-style).

A wrapper around any Learner, built like the I2 repeat option (src/options.py) and sharing its semi-MDP contract.

State graph. Built only from the transitions the wrapper sees: a node is an exact observed grid at a level (progress),
an edge is "primitive action a taken in node s led to node s'". A node's candidate actions are its legal simple actions
plus the click candidates of src/objects.py (one per object of that frame), fixed when the node is first seen. An action is
*untried* at a node until it has been taken there. The graph is per run; nothing is read from outside the run.

The option. "Go to the frontier": walk the shortest known path (breadth first, at most MAX_PATH edges) to the nearest node
that still has an untried action (the current node counts, at distance 0), then take one of its untried actions, chosen
uniformly with the wrapper's own RNG. The option is offered to the inner learner as one extra choice, available only when
such a node exists (`Observation.option_ok`). The learner picks it with its own policy:

    DQN  one extra row in the Q-network's choice set, Q(s, go-to-frontier), learned like every other action
    PPO  one extra action type, with its own logit, probability and policy-gradient credit

The option ends at the first of:  done (the untried action was taken), deviation (a path step did not reach the node the
graph expected: the game is not deterministic from that state, or hidden state differs), level_up, terminal, budget,
illegal (a path action is no longer legal).

Every primitive step is a real environment step charged to the budget. The inner learner sees the whole option as ONE
semi-Markov transition: first observation, action OPTION_ACT, discounted reward sum, last next-observation, k = number of
primitive steps (`info["k"]`; the primitive transitions are in `info["steps"]`), exactly like src/options.py.

Modes (fixed by configuration before any run):
    learned   the inner learner decides when to take the option (the V4 / A4 conditions)
    forced    the option is taken whenever it is available; the inner learner acts only when the graph is exhausted and
              is not shown the option's transitions. A training-free graph-explorer reference, not a candidate agent.
    off       the graph is recorded but the option is never offered; the wrapped learner is bit-for-bit the unwrapped one

Only the public Observation / Transition fields are read. No game identity, no SDK access.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import replace
from typing import Optional

import numpy as np

from .adapter import N_SIMPLE, OPTION_ACT, Observation, Transition
from .objects import candidates

MAX_PATH = 40                # longest walk to a frontier node; farther frontiers are not offered
TERMINAL = "terminal"        # graph node for GAME_OVER / WIN (never walked through)
MODES = ("learned", "forced", "off")
REASONS = ("done", "deviation", "level_up", "terminal", "budget", "illegal")


def node_key(obs: Observation) -> tuple:
    return (int(obs.progress), hash(obs.frame.tobytes()))


def candidate_actions(obs: Observation) -> list:
    """Action indices (src/adapter.py) worth trying at a node: legal simple actions, then one click per object."""
    acts = [i for i in range(N_SIMPLE) if obs.mask[i]]
    if obs.mask[N_SIMPLE:].any():
        acts += [N_SIMPLE + c.y * 64 + c.x for c in candidates(obs.frame)]
    return acts


class StateGraph:
    """Per-run graph of observed transitions with the set of untried candidate actions of every node."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.untried: dict = {}          # node -> set of candidate actions not yet taken there
        self.edges: dict = {}            # node -> {action: next node or TERMINAL}
        self.version = 0                 # grows with every recorded transition (plan cache key)
        self._plan_cache: tuple = (None, None)
        self.n_edges = 0

    def ensure(self, obs: Observation) -> tuple:
        k = node_key(obs)
        if k not in self.untried:
            self.untried[k] = set(candidate_actions(obs))
        return k

    def record(self, t: Transition) -> bool:
        """Add the edge of one primitive transition. True if it reached a node not seen before."""
        k0 = self.ensure(t.obs)
        new = False
        if t.terminal:
            k1 = TERMINAL
        else:
            new = node_key(t.next_obs) not in self.untried
            k1 = self.ensure(t.next_obs)
        out = self.edges.setdefault(k0, {})
        if t.action not in out:
            self.n_edges += 1
        out[t.action] = k1
        self.untried[k0].discard(t.action)
        self.version += 1
        return new

    def plan(self, obs: Observation) -> Optional[tuple]:
        """(actions to walk, expected node after each walked action, untried actions at the target) or None."""
        start = self.ensure(obs)
        if self._plan_cache[0] == (self.version, start):
            return self._plan_cache[1]
        found = None
        seen = {start}
        queue = deque([(start, (), ())])
        while queue:
            node, path, nodes = queue.popleft()
            free = self.untried.get(node)
            if free and not path:                         # at the current node: only actions legal right now
                free = {a for a in free if obs.mask[a]}
            if free:
                found = (list(path), list(nodes), sorted(free))
                break
            if len(path) >= MAX_PATH:
                continue
            for a, nxt in sorted(self.edges.get(node, {}).items()):
                if nxt == TERMINAL or nxt in seen or nxt not in self.untried:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + (a,), nodes + (nxt,)))
        if found is not None and found[0] and not obs.mask[found[0][0]]:
            found = None
        self._plan_cache = ((self.version, start), found)
        return found


class FrontierOption:
    """Learner wrapper implementing the frontier option. Satisfies the Learner protocol (src/learner.py)."""

    name = "frontier_option"

    def __init__(self, inner, *, mode: str = "learned", gamma: float = 0.99):
        if mode not in MODES or not 0.0 < gamma <= 1.0:
            raise ValueError(f"mode in {MODES} and 0 < gamma <= 1 required")
        self.inner, self.mode, self.gamma = inner, mode, float(gamma)

    # ---------------------------------------------------------------- lifecycle
    def start_run(self, seed: int) -> None:
        self.inner.start_run(seed)
        self.rng = random.Random(f"frontier-{int(seed)}")           # independent of every inner RNG
        self.graph = StateGraph()
        self._active = False
        self._driven = False                  # True when the inner learner chose the option (it must see the outcome)
        self._actions: list = []
        self._expected: list = []
        self._pos = 0
        self._macro: list = []
        self.decisions = self.offered = self.options = self.option_steps = self.primitive_steps = 0
        self.forced_options = self.new_in_option = self.new_outside = self.walk_ok = self.walks = 0
        self.stops: Counter = Counter()
        self.length_hist: Counter = Counter()

    # ---------------------------------------------------------------- acting
    def act(self, obs: Observation) -> int:
        if self._active:
            nxt = self._actions[self._pos]
            if obs.mask[nxt]:
                return nxt
            self._flush("illegal")            # hand the partial option over before the next decision
        self.decisions += 1
        plan = self.graph.plan(obs) if self.mode != "off" else None
        if plan is None:
            return self.inner.act(obs)
        self.offered += 1
        if self.mode == "forced":
            self.forced_options += 1
            return self._start(plan, driven=False)
        a = self.inner.act(replace(obs, option_ok=True))
        if a == OPTION_ACT:
            return self._start(plan, driven=True)
        return a

    def _start(self, plan: tuple, driven: bool) -> int:
        path, nodes, free = plan
        self._actions = list(path) + [self.rng.choice(free)]
        self._expected = list(nodes)
        self._pos, self._macro, self._active, self._driven = 0, [], True, driven
        self.options += 1
        if path:
            self.walks += 1
        return self._actions[0]

    def _annotated(self, t: Transition) -> Transition:
        """The transition as the inner learner should see it: its next observation says whether the option is available."""
        if self.mode != "learned" or t.terminal:      # only a learner that can choose the option is told about it
            return t
        ok = self.graph.plan(t.next_obs) is not None
        return replace(t, next_obs=replace(t.next_obs, option_ok=ok))

    def observe(self, t: Transition) -> None:
        self.primitive_steps += 1
        new = self.graph.record(t)
        if not self._active:
            self.new_outside += int(new)
            self.inner.observe(self._annotated(t))
            return
        self.option_steps += 1
        self.new_in_option += int(new)
        self._macro.append(t)
        i, self._pos = self._pos, self._pos + 1
        if t.terminal:
            reason = "terminal"
        elif t.truncated:
            reason = "budget"
        elif t.reward > 0:
            reason = "level_up"
        elif self._pos >= len(self._actions):
            reason = "done"
        elif node_key(t.next_obs) != self._expected[i]:
            reason = "deviation"
        else:
            return
        self._flush(reason)

    def _flush(self, reason: str) -> None:
        macro, self._macro, self._active = self._macro, [], False
        k = len(macro)
        self.stops[reason] += 1
        self.length_hist[k] += 1
        if len(self._actions) > 1 and reason == "done":       # the whole walk reached the node the graph expected
            self.walk_ok += 1
        if not self._driven or not macro:
            return
        reward = 0.0
        for i, s in enumerate(macro):
            reward += self.gamma ** i * float(s.reward)
        last = macro[-1]
        t = Transition(replace(macro[0].obs, option_ok=True), OPTION_ACT, reward, last.next_obs, last.terminal,
                       last.truncated, {**last.info, "k": k, "steps": list(macro), "option": "frontier"})
        self.inner.observe(self._annotated(t))

    def finish(self, final_obs: Optional[Observation]) -> None:
        if self._active:
            self._flush("budget")
        self.inner.finish(final_obs)

    # ---------------------------------------------------------------- reporting / checkpointing
    def diagnostics(self) -> dict:
        d = dict(self.inner.diagnostics())
        steps = max(self.primitive_steps, 1)
        d["frontier"] = {
            "mode": self.mode, "gamma": self.gamma, "max_path": MAX_PATH, "decisions": self.decisions,
            "decisions_with_option": self.offered, "option_offer_rate": self.offered / max(self.decisions, 1),
            "options_started": self.options, "option_choice_rate": self.options / max(self.offered, 1),
            "option_steps": self.option_steps, "primitive_steps": self.primitive_steps,
            "mean_option_length": self.option_steps / max(self.options, 1),
            "walks_started": self.walks, "walks_completed_as_expected": self.walk_ok,
            "path_success_rate": self.walk_ok / max(self.walks, 1),
            "new_states_in_options": self.new_in_option, "new_states_outside_options": self.new_outside,
            "new_states_per_100_option_steps": 100.0 * self.new_in_option / max(self.option_steps, 1),
            "new_states_per_100_other_steps": 100.0 * self.new_outside / max(steps - self.option_steps, 1),
            "graph_nodes": len(self.graph.untried), "graph_edges": self.graph.n_edges,
            "nodes_with_untried": sum(1 for v in self.graph.untried.values() if v),
            "length_histogram": {str(k): v for k, v in sorted(self.length_hist.items())},
            "stop_reasons": {r: self.stops.get(r, 0) for r in REASONS}}
        return d

    def model_info(self) -> dict:
        return getattr(self.inner, "model_info", lambda: {})()

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "rng": self.rng.getstate()}

    def load_state_dict(self, d: dict) -> None:
        self.inner.load_state_dict(d["inner"])
        self.rng.setstate(d["rng"])
