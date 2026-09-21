"""Probe: how ACTION6 coordinates are passed, and coordinate validation."""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("vc33")          # vc33 exposes ONLY ACTION6
f = env.reset()
print("vc33 available_actions:", f.available_actions)
base = np.array(f.frame[0])

print()
print("--- form A: env.step(GameAction.ACTION6, data={'x':..,'y':..}) ---")
r = env.step(GameAction.ACTION6, data={"x": 10, "y": 20})
print("  ok, n_frames:", len(r.frame), "echoed action_input.data:", r.action_input.data)

print()
print("--- form B: action.set_data({...}) then step(action) ---")
a = GameAction.ACTION6
a.set_data({"x": 40, "y": 5})
print("  action_data after set_data:", a.action_data)
r2 = env.step(a)
print("  ok, n_frames:", len(r2.frame), "echoed action_input.data:", r2.action_input.data)

print()
print("--- coordinate bounds (ComplexAction: x,y ge=0 le=63) ---")
for x, y in [(0, 0), (63, 63), (64, 0), (-1, 0)]:
    try:
        rr = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        print(f"  ({x:>3},{y:>3}) -> accepted, state={rr.state.value}")
    except Exception as ex:
        print(f"  ({x:>3},{y:>3}) -> {type(ex).__name__}: {str(ex).splitlines()[0][:90]}")

print()
print("--- does a click change the frame? (origin convention) ---")
env2 = arc.make("vc33")
f2 = env2.reset()
g0 = np.array(f2.frame[-1])
r3 = env2.step(GameAction.ACTION6, data={"x": 0, "y": 0})
g1 = np.array(r3.frame[-1])
print("  frame changed after click at (0,0):", not np.array_equal(g0, g1))
diff = np.argwhere(g0 != g1)
print("  n changed cells:", len(diff), "| first few (row,col):", diff[:5].tolist())
