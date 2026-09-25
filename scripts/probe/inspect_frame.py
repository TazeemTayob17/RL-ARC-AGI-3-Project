"""Probe: observation format from a live reset + step.

Tests claims: 64x64 grid, colours 0-15, one step returns a SEQUENCE of frames,
FrameData field names (levels_completed/win_levels vs score/win_score).
"""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import FrameData, FrameDataRaw, GameAction

GAME = "ls20"

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make(GAME)

print("=" * 60)
print("TYPES")
print("=" * 60)
print("env type          :", type(env).__name__)
print("FrameDataRaw flds :", list(FrameDataRaw.model_fields))
print("FrameData    flds :", list(FrameData.model_fields))
print("old 'score' field :", "score" in FrameData.model_fields)
print("old 'win_score'   :", "win_score" in FrameData.model_fields)

print()
print("=" * 60)
print("RESET")
print("=" * 60)
f = env.reset()
print("type(reset())     :", type(f).__name__)
print("state             :", f.state, "| repr:", repr(f.state))
print("levels_completed  :", f.levels_completed)
print("win_levels        :", f.win_levels)
print("available_actions :", f.available_actions, "| elem type:", type(f.available_actions[0]).__name__ if f.available_actions else "-")
print("action_space      :", [a.name for a in env.action_space])
print("guid              :", f.guid)
print("full_reset        :", f.full_reset)

print()
print("FRAME STRUCTURE")
print("len(f.frame)      :", len(f.frame), " <- frames returned by this ONE call")
arr = np.array(f.frame)
print("np.array(frame).shape:", arr.shape)
print("dtype             :", arr.dtype)
print("min / max cell    :", arr.min(), "/", arr.max())
print("unique values     :", sorted(np.unique(arr).tolist()))
print("per-sub-frame shape:", [np.array(g).shape for g in f.frame])

print()
print("=" * 60)
print("STEPS - frames per step")
print("=" * 60)
for i in range(8):
    acts = env.action_space
    a = acts[i % len(acts)] if acts else GameAction.RESET
    if a.is_complex():
        r = env.step(a, data={"x": 32, "y": 32})
    else:
        r = env.step(a)
    ar = np.array(r.frame)
    print(f"step {i}: action={a.name:<8} n_frames={len(r.frame)} shape={ar.shape} "
          f"min={ar.min()} max={ar.max()} state={r.state.value} lvl={r.levels_completed}")
