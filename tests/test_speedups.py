"""R6: the speed-ups must be behaviour-preserving. Each fast path is compared with the original slow computation."""
import math

import numpy as np

from src.dqn_v1.action_bank import ActionBank, ActionKey
from src.dqn_v1.exploration import _component_rarity


def _flood_fill_reference(g):
    """The original cell-by-cell flood fill."""
    seen = np.zeros((64, 64), dtype=bool)
    component = np.zeros((64, 64), dtype=np.float64)
    for y0 in range(64):
        for x0 in range(64):
            if seen[y0, x0]:
                continue
            color = int(g[y0, x0]); q = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while q:
                y, x = q.pop(); cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < 64 and 0 <= xx < 64 and not seen[yy, xx] and int(g[yy, xx]) == color:
                        seen[yy, xx] = True; q.append((yy, xx))
            val = -math.log((len(cells) + 1.0) / (g.size + 1.0))
            for y, x in cells:
                component[y, x] = val
    return component


def _grids():
    rng = np.random.default_rng(0)
    yield np.zeros((64, 64), dtype=np.int64)
    yield rng.integers(0, 16, (64, 64))
    yield rng.integers(0, 2, (64, 64))
    g = np.zeros((64, 64), dtype=np.int64)
    g[::2, :] = 1                                   # horizontal stripes
    yield g.copy()
    g[:, 10] = 1                                    # stripes joined by a column
    yield g
    spiral = np.zeros((64, 64), dtype=np.int64)     # one long snake
    spiral[::2, :] = 1
    spiral[1::4, 63] = 1
    spiral[3::4, 0] = 1
    yield spiral
    yield np.kron(rng.integers(0, 5, (8, 8)), np.ones((8, 8), dtype=np.int64))    # blocks


def test_component_rarity_equals_flood_fill_exactly():
    for g in _grids():
        assert np.array_equal(_component_rarity(g), _flood_fill_reference(g))


def test_candidate_cache_equals_fresh_encoding():
    bank = ActionBank()
    for available in ([1, 2, 3], [6], [1, 6, 7], [6, 1, 1]):
        keys, arr = bank.encode_candidates(available)
        fresh_keys = bank.candidates(available)
        fresh = np.stack([bank.encode(a) for a in fresh_keys])
        assert keys == fresh_keys and np.array_equal(arr, fresh)
        for i in (0, len(keys) // 2, len(keys) - 1):
            k = keys[i]
            assert bank.index_of(available, (k.action_id, k.x, k.y)) == i
    assert bank.encode_candidates([1, 6, 7])[1] is bank.encode_candidates([7, 6, 1])[1]


def test_click_index_matches_original_linear_search():
    bank = ActionBank()
    keys, _ = bank.encode_candidates([2, 6])
    for a in (ActionKey(2), ActionKey(6, 0, 0), ActionKey(6, 63, 63), ActionKey(6, 17, 40)):
        linear = next(i for i, k in enumerate(keys) if (k.action_id, k.x, k.y) == (a.action_id, a.x, a.y))
        assert bank.index_of([2, 6], (a.action_id, a.x, a.y)) == linear
