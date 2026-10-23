"""Selected structured epsilon explorer from the final DQN baseline."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict

import numpy as np

from .action_bank import ActionKey

SELECTOR = {
    "a6_salience_w": 1.0,
    "action6_prob": 0.7,
    "family": "split_specialist",
    "id": "D14_split_effect_count",
    "kind": "score",
    "preserve_type_prior": True,
    "salience_component_w": 0.5,
    "salience_contrast_w": 0.35,
    "salience_rarity_w": 1.0,
    "simple_count_w": 0.7,
    "simple_effect_w": 0.7,
    "split_by_type": True,
    "temperature": 0.6,
}


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return np.zeros_like(x)
    sd = float(x.std())
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def _sample_softmax(rng: random.Random, vals: np.ndarray, temperature: float) -> int:
    v = np.asarray(vals, dtype=np.float64) / max(float(temperature), 1e-6)
    v = v - np.max(v)
    p = np.exp(np.clip(v, -50.0, 50.0))
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        return rng.randrange(len(v))
    p /= total
    u, c = rng.random(), 0.0
    for i, q in enumerate(p.tolist()):
        c += q
        if u <= c:
            return i
    return len(v) - 1


def _state_key(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=np.uint8)
    out = []
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            tile = g[y:y + 16, x:x + 16].reshape(-1)
            out.append(int(np.bincount(tile, minlength=16).argmax()))
    return hashlib.blake2b(bytes(out), digest_size=12).hexdigest()


def _bucket(a: ActionKey):
    return (6, int(a.x) // 8, int(a.y) // 8) if int(a.action_id) == 6 else (int(a.action_id),)


def _component_rarity(g: np.ndarray) -> np.ndarray:
    """-log((size + 1) / (cells + 1)) of each cell's 4-connected same-colour component.

    Row runs joined with a union-find: the component sizes, and so every value, equal those of a cell-by-cell flood fill.
    """
    h, w = g.shape
    flat = g.reshape(-1)
    change = np.ones((h, w), dtype=bool)
    change[:, 1:] = g[:, 1:] != g[:, :-1]
    starts = np.flatnonzero(change.reshape(-1))
    run_of_cell = np.cumsum(change.reshape(-1)) - 1
    length = np.diff(np.append(starts, h * w))
    row, x0 = starts // w, starts % w
    x1 = x0 + length
    color = flat[starts].tolist()
    x0l, x1l = x0.tolist(), x1.tolist()
    row_first = np.searchsorted(row, np.arange(h + 1)).tolist()
    parent = list(range(len(starts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y in range(1, h):
        i, i_end = row_first[y - 1], row_first[y]
        j, j_end = row_first[y], row_first[y + 1]
        while i < i_end and j < j_end:
            if color[i] == color[j] and x0l[i] < x1l[j] and x0l[j] < x1l[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
            if x1l[i] < x1l[j]:
                i += 1
            else:
                j += 1
    root = np.asarray([find(i) for i in range(len(starts))], dtype=np.int64)
    size = np.bincount(root, weights=length, minlength=len(starts))
    value = np.zeros(len(starts), dtype=np.float64)
    for r in np.unique(root).tolist():
        value[r] = -math.log((int(size[r]) + 1.0) / (g.size + 1.0))
    return value[root][run_of_cell].reshape(h, w)


def _visual_salience(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    counts = np.bincount(np.clip(g.ravel(), 0, 15), minlength=16).astype(np.float64)
    rarity = -np.log((counts[np.clip(g, 0, 15)] + 1.0) / (g.size + 16.0))
    contrast = np.zeros_like(rarity)
    contrast[1:, :] += (g[1:, :] != g[:-1, :]); contrast[:-1, :] += (g[:-1, :] != g[1:, :])
    contrast[:, 1:] += (g[:, 1:] != g[:, :-1]); contrast[:, :-1] += (g[:, :-1] != g[:, 1:])
    component = _component_rarity(g)
    return rarity + 0.35 * contrast + 0.5 * component


class StructuredExplorer:
    def __init__(self):
        self.pair_counts = defaultdict(int)
        self.structured_calls = 0

    def choose(self, rng: random.Random, available: list[int], grid: np.ndarray, simple: list[int],
               prior_simple_logits: np.ndarray, prior_a6_logits: np.ndarray | None):
        sk = _state_key(grid)
        count_simple = np.asarray([1.0 / math.sqrt(self.pair_counts[(sk, (aid,))] + 1.0) for aid in simple])
        simple_score = 0.7 * _zscore(prior_simple_logits) + 0.7 * _zscore(count_simple)
        a6_score = _zscore(_visual_salience(grid)).reshape(-1) if prior_a6_logits is not None else None
        if a6_score is not None and simple:
            choose_a6 = rng.random() < 0.7
        else:
            choose_a6 = a6_score is not None
        if choose_a6:
            j = _sample_softmax(rng, a6_score, 0.6)
            action = ActionKey(6, j % 64, j // 64)
        else:
            i = _sample_softmax(rng, simple_score, 0.6)
            action = ActionKey(int(simple[i]))
        self.structured_calls += 1
        return action, "d14"

    def observe(self, grid: np.ndarray, action: ActionKey):
        self.pair_counts[(_state_key(grid), _bucket(action))] += 1
