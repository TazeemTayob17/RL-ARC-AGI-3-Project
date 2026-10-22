"""Improvement 2 (candidate A): event-terminated repeat option (plan 7.2 / 7.3).

A wrapper around any Learner. The inner learner still makes every decision. After a decision whose action changed
the frame, the wrapper may repeat that same action, with probability `prob` (drawn once per decision, from the
wrapper's own RNG), until the effect stops. The option ends at the first of:

    cap            `cap` primitive steps have been taken (including the first)
    no_effect      a repetition left the frame unchanged
    cycle          a repetition returned to a frame already seen inside this option
    level_up       a step raised the high-water mark (reward > 0)
    terminal       GAME_OVER or WIN
    illegal        the action is no longer in the legal mask
    budget         the budget was exhausted mid-option (truncation)

Every primitive step is a real environment step charged to the budget. The inner learner sees the whole option as ONE
semi-Markov transition: the first observation, the first action, the discounted sum of the primitive rewards, the last
next-observation and k = number of steps (`info["k"]`; the primitive transitions are in `info["steps"]`). Learners
discount that transition by gamma**k. With `prob = 0` or `cap = 1` the wrapper forwards every transition unchanged, so
the wrapped learner is bit-for-bit the unwrapped learner.

Only the public Observation / Transition fields are read (frames, mask, reward, terminal flags). No game identity, no
SDK access, no new learned parameters: `cap` and `prob` are configuration fixed before any run.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Optional

import numpy as np

from .adapter import Observation, Transition

REASONS = ("cap", "no_effect", "cycle", "level_up", "terminal", "illegal", "budget", "not_engaged")


def _key(frame: np.ndarray) -> bytes:
    return frame.tobytes()


def _changed(t: Transition) -> bool:
    return not np.array_equal(t.obs.frame, t.next_obs.frame)


class EventRepeat:
    """Learner wrapper implementing the repeat option. Satisfies the Learner protocol (src/learner.py)."""

    name = "event_repeat"

    def __init__(self, inner, *, cap: int = 8, prob: float = 0.5, gamma: float = 0.99):
        if cap < 1 or not 0.0 <= prob <= 1.0 or not 0.0 < gamma <= 1.0:
            raise ValueError("cap >= 1, 0 <= prob <= 1, 0 < gamma <= 1 required")
        self.inner, self.cap, self.prob, self.gamma = inner, int(cap), float(prob), float(gamma)

    # ---------------------------------------------------------------- lifecycle
    def start_run(self, seed: int) -> None:
        self.inner.start_run(seed)
        self.rng = random.Random(f"event-repeat-{int(seed)}")       # independent of every inner RNG
        self._macro: list = []                # primitive transitions of the option in progress
        self._seen: set = set()               # frames visited inside the option (cycle check)
        self._repeating = False
        self._action: Optional[int] = None
        self.decisions = self.steps = self.options = self.repeated_steps = 0
        self.length_hist: Counter = Counter()
        self.stops: Counter = Counter()

    # ---------------------------------------------------------------- acting
    def act(self, obs: Observation) -> int:
        if self._repeating:
            return self._action
        self.decisions += 1
        return self.inner.act(obs)

    def observe(self, t: Transition) -> None:
        self.steps += 1
        if not self._macro:
            self._seen = {_key(t.obs.frame)}
        else:
            self.repeated_steps += 1
        self._macro.append(t)
        reason = self._stop_reason(t)
        if reason is None:
            self._repeating, self._action = True, int(t.action)
            self._seen.add(_key(t.next_obs.frame))
            return
        self._repeating = False
        self._flush(reason)

    def _stop_reason(self, t: Transition) -> Optional[str]:
        """None -> repeat the action once more; otherwise the reason the option ends here."""
        if t.terminal:
            return "terminal"
        if t.truncated:
            return "budget"
        if t.reward > 0:
            return "level_up"
        if not _changed(t):
            return "no_effect"                 # also covers a first step with no effect: nothing to repeat
        if len(self._macro) >= self.cap:
            return "cap"
        if not bool(t.next_obs.mask[t.action]):
            return "illegal"
        if _key(t.next_obs.frame) in self._seen:
            return "cycle"
        if len(self._macro) == 1 and not (self.prob > 0.0 and self.rng.random() < self.prob):
            return "not_engaged"               # drawn once per decision, after its first step had an effect
        return None

    def _flush(self, reason: str) -> None:
        macro, self._macro = self._macro, []
        k = len(macro)
        self.length_hist[k] += 1
        self.stops[reason] += 1
        if k > 1:
            self.options += 1
            reward = 0.0
            for i, s in enumerate(macro):
                reward += self.gamma ** i * float(s.reward)
            last = macro[-1]
            t = Transition(macro[0].obs, macro[0].action, reward, last.next_obs, last.terminal, last.truncated,
                           {**last.info, "k": k, "steps": list(macro)})
        else:
            t = macro[0]                       # unchanged object: the wrapped learner sees exactly what it would unwrapped
        self.inner.observe(t)

    def finish(self, final_obs: Optional[Observation]) -> None:
        if self._macro:                        # run ended mid-option (e.g. the loop stopped): hand over what was taken
            self._repeating = False
            self._flush("budget")
        self.inner.finish(final_obs)

    # ---------------------------------------------------------------- reporting / checkpointing
    def diagnostics(self) -> dict:
        d = dict(self.inner.diagnostics())
        d["option"] = {"cap": self.cap, "prob": self.prob, "gamma": self.gamma, "decisions": self.decisions,
                       "primitive_steps": self.steps, "options_longer_than_one": self.options,
                       "repeated_steps": self.repeated_steps,
                       "mean_option_length": self.steps / max(self.decisions, 1),
                       "length_histogram": {str(k): v for k, v in sorted(self.length_hist.items())},
                       "stop_reasons": {r: self.stops.get(r, 0) for r in REASONS}}
        return d

    def model_info(self) -> dict:
        return getattr(self.inner, "model_info", lambda: {})()

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "rng": self.rng.getstate(),
                "counters": [self.decisions, self.steps, self.options, self.repeated_steps]}

    def load_state_dict(self, d: dict) -> None:
        self.inner.load_state_dict(d["inner"])
        self.rng.setstate(d["rng"])
        self.decisions, self.steps, self.options, self.repeated_steps = d["counters"]
