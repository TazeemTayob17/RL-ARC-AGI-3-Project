"""Improvement 3: object-level click candidates and effect memory (docs/further-improvements.md, I3).

The 4,096 click pixels are replaced by one candidate per object of the current public frame. An object is a
4-connected single-colour component; a component larger than TILE_AREA is cut into TILE x TILE tiles so that a large
background cannot hide a small interactive region. Each candidate is clicked at one representative pixel (the object
cell closest to its centroid), so the environment action is an ordinary ACTION6 click and nothing about the adapter,
the budget or the scorer changes.

The effect memory only reads the Transition the learner is shown (frames, whether the frame changed, the state, the
action it took). It never sees a game id. All constants are fixed here, once, and are not tuned.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

GRID = 64
NCOL = 16
TILE = 16               # large components are cut into TILE x TILE tiles
TILE_AREA = 512         # a component with more cells than this is tiled
K_MAX = 96              # candidate cap per frame
RETRY_PROB = 0.05       # a candidate the memory would exclude is kept with this probability
DEAD_AFTER = 3          # tries without any effect before an object signature counts as dead
FEAT_DIM = NCOL + 11    # per-candidate feature vector length (see features())


def component_labels(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-connected same-colour component label of every cell, and the size of every component.

    Row runs joined with a union-find. Labels are 0..n-1 in order of first appearance (row-major).
    """
    h, w = g.shape
    flat = np.asarray(g).reshape(-1)
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
    _, comp_of_run = np.unique(root, return_inverse=True)
    n = int(comp_of_run.max()) + 1
    size = np.bincount(comp_of_run, weights=length, minlength=n).astype(np.int64)
    return comp_of_run[run_of_cell].reshape(h, w), size


@dataclass(frozen=True)
class Candidate:
    x: int                      # representative pixel
    y: int
    colour: int
    area: int
    bbox: tuple                 # x0, y0, x1, y1 (inclusive)
    centroid: tuple             # cx, cy
    pos_sig: tuple              # identifies this object at this place (colour, area, bbox)
    shape_sig: tuple            # identifies this kind of object anywhere (colour, area, width, height)


def candidates(frame: np.ndarray) -> list:
    """Object candidates of a frame, at most K_MAX, in a fixed deterministic order (see _priority)."""
    return candidates_with_cells(frame)[0]


def candidates_with_cells(frame: np.ndarray) -> tuple:
    """(candidates, cells): cells[i] are the flat pixel indices (y * 64 + x) that candidate i covers."""
    g = np.asarray(frame, dtype=np.int64)
    labels, size = component_labels(g)
    n = len(size)
    ys, xs = np.divmod(np.arange(GRID * GRID), GRID)
    order = np.argsort(labels.reshape(-1), kind="stable")
    bounds = np.searchsorted(labels.reshape(-1)[order], np.arange(n + 1))
    out: list = []
    for c in range(n):
        cells = order[bounds[c]:bounds[c + 1]]
        if size[c] > TILE_AREA:
            tile = (ys[cells] // TILE) * (GRID // TILE) + xs[cells] // TILE
            for t in np.unique(tile).tolist():
                part = cells[tile == t]
                out.append((_make(g, part, ys, xs), part))
        else:
            out.append((_make(g, cells, ys, xs), cells))
    if len(out) > K_MAX:
        out.sort(key=lambda co: _priority(co[0]))
        out = out[:K_MAX]
    return [c for c, _ in out], [cells for _, cells in out]


def _priority(c: Candidate) -> tuple:
    """Which candidates survive the cap: smaller objects first (interactive things are usually small), then position."""
    return (c.area, c.y, c.x)


def _make(g: np.ndarray, cells: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> Candidate:
    cy, cx = ys[cells], xs[cells]
    mx, my = float(cx.mean()), float(cy.mean())
    k = int(np.argmin((cx - mx) ** 2 + (cy - my) ** 2 + 1e-6 * (cy * GRID + cx)))
    x0, y0, x1, y1 = int(cx.min()), int(cy.min()), int(cx.max()), int(cy.max())
    col = int(g[cy[0], cx[0]])
    area = int(len(cells))
    return Candidate(int(cx[k]), int(cy[k]), col, area, (x0, y0, x1, y1), (mx, my),
                     (col, area, x0, y0, x1, y1), (col, area, x1 - x0 + 1, y1 - y0 + 1))


def state_key(frame: np.ndarray, progress: int) -> int:
    return hash((int(progress), np.asarray(frame).tobytes()))


# ---------------------------------------------------------------- effect memory
class EffectMemory:
    """Per-run record of what clicking each object did.

    * exact: (state, object) -> outcome of the last click ("none" = the frame did not change).
    * by signature: object at this place / kind of object anywhere -> [tries, effective tries].
    A candidate is *known dead* when its exact (state, object) pair was a no-op, or when its place-signature has been
    tried DEAD_AFTER times with no effect. Known-dead candidates are dropped from the choice set except with
    probability RETRY_PROB (so a hidden-state effect is never ruled out for good). The retry draw uses its own RNG, so
    it never touches the learner's streams. With `enabled=False` nothing is recorded and nothing is dropped.
    """

    def __init__(self, enabled: bool = True, retry_prob: float = RETRY_PROB, dead_after: int = DEAD_AFTER):
        self.enabled, self.retry_prob, self.dead_after = bool(enabled), float(retry_prob), int(dead_after)
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.exact: dict = {}
        self.by_pos: dict = {}
        self.by_shape: dict = {}
        self.rng = random.Random(1_000_003 * int(seed) + 17)
        self.n_offered = self.n_dropped = self.n_retried = self.n_recorded = self.n_all_dead = 0

    def known_dead(self, skey: int, c: Candidate) -> bool:
        if self.exact.get((skey, c.pos_sig)) == "none":
            return True
        tries, hits = self.by_pos.get(c.pos_sig, (0, 0))
        return tries >= self.dead_after and hits == 0

    def filter(self, skey: int, cands: list) -> list:
        """Indices of the candidates kept as the choice set (never empty if `cands` is not)."""
        self.n_offered += len(cands)
        if not self.enabled or not cands:
            return list(range(len(cands)))
        keep, dropped = [], []
        for i, c in enumerate(cands):
            if self.known_dead(skey, c):
                dropped.append(i)
            else:
                keep.append(i)
        for i in dropped:
            if self.rng.random() < self.retry_prob:
                keep.append(i)
                self.n_retried += 1
        self.n_dropped += len(dropped)
        if not keep:                                   # everything looks dead: do not leave the click without choices
            self.n_all_dead += 1
            return list(range(len(cands)))
        return sorted(keep)

    def record(self, skey: int, c: Candidate, effective: bool, level_up: bool = False, game_over: bool = False) -> None:
        if not self.enabled:
            return
        self.n_recorded += 1
        outcome = "gameover" if game_over else "levelup" if level_up else "changed" if effective else "none"
        self.exact[(skey, c.pos_sig)] = outcome
        for table, sig in ((self.by_pos, c.pos_sig), (self.by_shape, c.shape_sig)):
            tries, hits = table.get(sig, (0, 0))
            table[sig] = (tries + 1, hits + int(effective))

    def stats(self, skey: int, c: Candidate) -> tuple:
        """Features describing what is known about a candidate: place tries / hit rate, kind tries / hit rate, last outcome."""
        pt, ph = self.by_pos.get(c.pos_sig, (0, 0))
        st, sh = self.by_shape.get(c.shape_sig, (0, 0))
        last = self.exact.get((skey, c.pos_sig))
        return (min(pt, 8) / 8.0, ph / pt if pt else 0.5, min(st, 8) / 8.0, sh / st if st else 0.5,
                {"none": 1.0, "changed": 2.0, "levelup": 3.0, "gameover": 4.0}.get(last, 0.0) / 4.0)

    def summary(self) -> dict:
        n = max(self.n_offered, 1)
        return {"enabled": self.enabled, "recorded": self.n_recorded, "candidates_offered": self.n_offered,
                "dropped_as_dead": self.n_dropped, "dropped_share": self.n_dropped / n, "retried": self.n_retried,
                "all_dead_fallbacks": self.n_all_dead, "pos_signatures": len(self.by_pos),
                "shape_signatures": len(self.by_shape), "exact_entries": len(self.exact)}


def features(c: Candidate, stats: tuple) -> np.ndarray:
    """Per-candidate vector: colour one-hot, log-area, bbox size, centroid, border flag, and the memory statistics."""
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    v[min(c.colour, NCOL - 1)] = 1.0
    x0, y0, x1, y1 = c.bbox
    base = NCOL
    v[base] = math.log1p(c.area) / math.log1p(GRID * GRID)
    v[base + 1] = (x1 - x0 + 1) / GRID
    v[base + 2] = (y1 - y0 + 1) / GRID
    v[base + 3] = c.centroid[0] / (GRID - 1)
    v[base + 4] = c.centroid[1] / (GRID - 1)
    v[base + 5] = 1.0 if (x0 == 0 or y0 == 0 or x1 == GRID - 1 or y1 == GRID - 1) else 0.0
    v[base + 6:base + 11] = stats
    return v


@dataclass
class ObjectView:
    """The click choice set of one observation: all candidates, the ones the memory keeps, and a random-pixel slot."""
    cands: list
    obj_index: np.ndarray       # int16 [4096]: candidate covering each pixel, -1 if none (capped away)
    keep: list                  # indices into cands offered to the policy
    slot_xy: tuple              # pixel clicked if the random-pixel slot is chosen
    skey: int


def build_view(frame: np.ndarray, progress: int, memory: EffectMemory) -> ObjectView:
    cands, cells = candidates_with_cells(frame)
    obj_index = np.full(GRID * GRID, -1, dtype=np.int16)
    for i, cell in enumerate(cells):
        obj_index[cell] = i
    skey = state_key(frame, progress)
    keep = memory.filter(skey, cands)
    taken = {(cands[i].x, cands[i].y) for i in keep}
    slot_xy = (memory.rng.randrange(GRID), memory.rng.randrange(GRID))
    for _ in range(64):                               # the slot must be a different pixel from every object's click pixel
        if slot_xy not in taken:
            break
        slot_xy = (memory.rng.randrange(GRID), memory.rng.randrange(GRID))
    return ObjectView(cands, obj_index, keep, slot_xy, skey)
