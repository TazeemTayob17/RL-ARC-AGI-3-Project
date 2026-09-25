"""Record the pinned environment (PROJECT_PLAN 3.1) to results/env_pins.json."""
import json, platform, subprocess, sys, os
from importlib.metadata import version

pins = {
    "python": sys.version,
    "os": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "packages": {p: version(p) for p in ("arc-agi", "arcengine", "numpy", "pydantic")},
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
}
try:
    import psutil
    pins["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
except ImportError:
    pass
os.makedirs("results", exist_ok=True)
json.dump(pins, open("results/env_pins.json", "w"), indent=2)
print(json.dumps(pins, indent=2))
