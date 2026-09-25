"""Which ops dominate a QNet forward+backward (batch 32)?"""
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ObsBatch, QNet

ch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
B = 32
net = QNet({"id": "onehotcnn_stack2", "channels": ch, "context_dim": 32 if ch == 8 else 128})
b = ObsBatch(torch.randint(0, 13, (B, 64, 64)), torch.randint(0, 13, (B, 64, 64)),
             torch.randint(-1, 4000, (B,)), torch.zeros(B, 2), torch.ones(B, 4102, dtype=torch.bool))
net(b).mean().backward()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    net(b).mean().backward()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=40))
