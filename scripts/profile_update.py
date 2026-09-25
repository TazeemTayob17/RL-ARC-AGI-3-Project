"""Profile one DQN-style update (fwd online b, fwd online nb, fwd target nb, backward) at several settings."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ObsBatch, QNet

B = 32
for channels in (8, 16, 32):
    for threads in (4, 8):
        torch.set_num_threads(threads)
        net = QNet({"id": "onehotcnn_stack2", "channels": channels, "context_dim": 32 if channels == 8 else 128})
        b = ObsBatch(torch.randint(0, 13, (B, 64, 64)), torch.randint(0, 13, (B, 64, 64)),
                     torch.randint(-1, 4000, (B,)), torch.zeros(B, 2), torch.ones(B, 4102, dtype=torch.bool))
        for _ in range(2):
            net(b)
        t = time.perf_counter()
        for _ in range(3):
            with torch.no_grad():
                net(b); net(b)
        fwd2 = (time.perf_counter() - t) / 3
        t = time.perf_counter()
        for _ in range(3):
            net(b).mean().backward()
        fb = (time.perf_counter() - t) / 3
        t = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                net(ObsBatch(b.frame[:1], b.prev_frame[:1], b.prev_action[:1], b.progress[:1], b.mask[:1]))
        act = (time.perf_counter() - t) / 20
        print(f"channels={channels:>2} threads={threads}  2 no-grad fwd={fwd2*1e3:6.0f} ms  fwd+bwd={fb*1e3:6.0f} ms  "
              f"=> update ~{(fwd2 + fb)*1e3:6.0f} ms | single-obs act ~{act*1e3:5.1f} ms", flush=True)
