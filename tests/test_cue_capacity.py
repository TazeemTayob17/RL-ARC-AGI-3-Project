"""Distant-cue capacity check (plan 6): can whole-board context reach the click head?

Supervised, not RL: maximise the probability mass the click head puts on the *cued* square. Each square is visible
locally (colours 3 and 4) but which one is correct depends on a far-away cue, so a local-only click head cannot beat
0.5 mass on the right square; reaching ~1.0 proves the global-context path (pooled board vector broadcast into the
click head) is wired and trainable. Control: the same model with the context path zeroed must stay near 0.5.
This tests the representation/action path, not RL sample efficiency.
"""
import numpy as np
import torch

from src.adapter import Observation, build_mask
from src.model import ActorCritic, collate
from src.synthetic import CueTask

TINY = {"id": "onehotcnn_stack2", "channels": 8, "context_dim": 32}
MASK = build_mask([6])


def make_batch(rng, n):
    """n random cue-task frames (random square positions and cues) + a mask of the correct square's pixels."""
    t = CueTask(int(rng.integers(1 << 30)))
    obs, correct = [], torch.zeros(n, 64, 64, dtype=torch.bool)
    for i in range(n):
        t.reset()
        f = t._draw()
        obs.append(Observation(f, f, MASK, -1, 0, 0, 10 ** 6))
        x, y = t.boxes[t.cue]
        correct[i, y:y + CueTask.S, x:x + CueTask.S] = True
    return collate(obs), correct.flatten(1)


def train_and_score(ablate_context: bool, steps=250, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ActorCritic(TINY)
    if ablate_context:                                   # click head sees local features only
        for prm in (net.click.ctx.weight, net.click.ctx.bias):
            prm.data.zero_()
            prm.requires_grad_(False)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(steps):
        b, correct = make_batch(rng, 16)
        mass = torch.softmax(net(b)[1], 1).masked_fill(~correct, 0).sum(1)
        loss = -(mass + 1e-6).log().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        b, correct = make_batch(np.random.default_rng(999), 64)
        return float(torch.softmax(net(b)[1], 1).masked_fill(~correct, 0).sum(1).mean())


def test_global_context_reaches_click_head_and_ablation_cannot():
    full = train_and_score(ablate_context=False)
    local_only = train_and_score(ablate_context=True)
    assert full > 0.9, f"global-context click head reached only {full:.3f} mass on the cued square"
    assert local_only < 0.6, f"context-ablated head should be near 0.5, got {local_only:.3f}"
