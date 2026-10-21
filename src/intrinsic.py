"""Improvement 1: count-based novelty bonus (plan 7.2).

bonus(s' | transition) = beta / sqrt(n(s')) where n counts exact observed grids, scoped per run and per level
(progress). It is a training signal only: it is stored beside, never inside, the extrinsic reward, and never enters
scoring. Rules fixed before any run:

* a "state" is the exact 64x64 grid, not "the frame changed";
* zero bonus when the transition returns to the state the previous transition started from (action/undo loop);
* beta = 0 disables the module entirely and leaves learner behaviour bit-for-bit unchanged (no state is touched);
* counts are cleared by reset(), i.e. at every run start; they persist across GAME_OVER resets inside a run.

The counter only reads the public Observation fields of a Transition (frame, progress).
"""
from __future__ import annotations

import math

from .adapter import Transition


class NoveltyCounter:
    def __init__(self, beta: float = 0.0):
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.beta = float(beta)
        self.reset()

    def reset(self) -> None:
        self.counts: dict = {}
        self._before = None         # state the previous transition started from (loop detection)
        self._last_next = None      # state the previous transition ended in (episode-break detection)
        self.n_calls = self.n_nonzero = self.n_loop_zeroed = 0
        self.total = 0.0

    @staticmethod
    def _key(obs):
        return (int(obs.progress), hash(obs.frame.tobytes()))

    def bonus(self, t: Transition) -> float:
        if self.beta == 0.0:
            return 0.0
        k0, k1 = self._key(t.obs), self._key(t.next_obs)
        if k0 != self._last_next:                   # new episode (after a reset): no history to compare with
            self._before = None
        self.counts.setdefault(k0, 1)               # the starting state counts as seen once
        n = self.counts[k1] = self.counts.get(k1, 0) + 1
        loop = k1 == self._before
        self._before, self._last_next = k0, k1
        self.n_calls += 1
        if loop:
            self.n_loop_zeroed += 1
            return 0.0
        b = self.beta / math.sqrt(n)
        self.n_nonzero += 1
        self.total += b
        return b

    def summary(self) -> dict:
        return {"beta": self.beta, "calls": self.n_calls, "nonzero": self.n_nonzero,
                "loop_zeroed": self.n_loop_zeroed, "total_bonus": self.total,
                "mean_bonus": self.total / max(self.n_nonzero, 1), "unique_states": len(self.counts)}
