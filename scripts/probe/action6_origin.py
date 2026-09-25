"""Probe: map ACTION6 (x,y) onto grid (row,col) to establish the origin convention."""
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
print(f"{'click (x,y)':<14} {'changed (row,col) cells':<40}")
print("-" * 58)
for x, y in [(0, 0), (10, 0), (0, 10), (63, 0), (0, 63), (20, 40)]:
    env = arc.make("vc33")
    f = env.reset()
    g0 = np.array(f.frame[-1])
    r = env.step(GameAction.ACTION6, data={"x": x, "y": y})
    g1 = np.array(r.frame[-1])
    d = np.argwhere(g0 != g1)
    print(f"({x:>2},{y:>2}){'':<7} {str(d.tolist()[:4]):<40}")

print()
print("Interpretation: if click (x,y) changes cell (row=y, col=x) -> x=column, y=row, origin top-left.")
