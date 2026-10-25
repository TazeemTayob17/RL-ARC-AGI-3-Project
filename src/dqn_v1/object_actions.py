"""I3 for the DQN: the per-state choice set of simple actions plus one row per object (and a random-pixel slot).

A row is the usual 10-D action vector (id one-hot, x, y, valid) followed by the object features of src/objects.py
(zeros for simple actions and for the random-pixel slot). The Q-network scores the rows exactly as before; only the
rows are different.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..objects import FEAT_DIM, GRID, EffectMemory, ObjectView, features
from .action_bank import ActionBank, ActionKey

CLICK_ID = 6


@dataclass
class ObjectActionSet:
    keys: list
    rows: np.ndarray
    click_rows: list                 # row indices of click candidates (objects and the slot)
    cands: list                      # Candidate per row (None for simple actions and the slot)
    index: dict = field(default_factory=dict)
    skey: int | None = None          # state key the effect memory files this step's outcome under

    def row_of(self, a: ActionKey) -> int:
        return self.index[(int(a.action_id), int(a.x), int(a.y))]


def build_action_set(bank: ActionBank, available, view: ObjectView | None, memory: EffectMemory) -> ObjectActionSet:
    """Rows for every simple action in `available`, then (if ACTION6 is legal) the kept objects and the slot."""
    simple = [a for a in available if int(a) != CLICK_ID]
    keys, simple_rows = bank.encode_candidates(simple)
    keys = list(keys)
    rows = [np.concatenate([simple_rows, np.zeros((len(keys), FEAT_DIM), dtype=np.float32)], axis=1)] if keys else []
    cands: list = [None] * len(keys)
    click_rows: list = []
    if CLICK_ID in {int(a) for a in available} and view is not None:
        extra = []
        for i in view.keep:
            c = view.cands[i]
            key = ActionKey(CLICK_ID, c.x, c.y)
            extra.append(np.concatenate([bank.encode(key), features(c, memory.stats(view.skey, c))]))
            click_rows.append(len(keys)); keys.append(key); cands.append(c)
        sx, sy = view.slot_xy
        key = ActionKey(CLICK_ID, sx, sy)
        extra.append(np.concatenate([bank.encode(key), np.zeros(FEAT_DIM, dtype=np.float32)]))
        click_rows.append(len(keys)); keys.append(key); cands.append(None)
        rows.append(np.stack(extra).astype(np.float32))
    arr = np.concatenate(rows, axis=0).astype(np.float32)
    index = {}
    for i, k in enumerate(keys):
        index[(int(k.action_id), int(k.x), int(k.y))] = i
    return ObjectActionSet(keys, arr, click_rows, cands, index, None if view is None else view.skey)
