"""Anti-cheating guard (plan 1, 'Anti-cheating and anti-overfitting rules').

Static scan of agent code. Agent code may use only the public `Observation` (adapter.py); it must not reach the
game files, the SDK, human baselines, the network, or the game's identity. The run controller (run.py, config.py,
checkpoint_resolver.py) and the SDK adapters (adapter.py, bridge.py) are the only places allowed to know the game id.

LEARNER files are scanned strictly. SHELL files (the Kaggle wrapper) only for game ids and forbidden resources.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 25 public game ids (also read from configs/splits.json at scan time so the list cannot silently go stale).
PUBLIC_GAME_IDS = frozenset({
    "ar25", "bp35", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "r11l", "s5i5", "sb26", "sk48", "su15",
    "tn36", "tr87", "tu93", "vc33", "cd82", "ls20", "m0r0", "sp80", "re86", "cn04", "wa30", "sc25",
})

FORBIDDEN_MODULES = {"arc_agi", "arcengine", "agents", "subprocess", "socket", "requests", "urllib", "http", "glob"}
FORBIDDEN_STRINGS = ("environment_files", "baseline_actions", "human_actions", "ARC_API_KEY")
FORBIDDEN_NAMES = {"game", "game_id", "game_ids", "baseline_actions", "environment_files", "env_files"}

# Files that may know the game id or touch the SDK. Everything else under src/ is agent code and is scanned strictly,
# so a new learner file is covered the moment it exists (no list to forget to update).
CONTROLLER_FILES = frozenset({"src/run.py", "src/config.py", "src/checkpoint_resolver.py", "src/adapter.py",
                              "src/bridge.py", "src/synthetic.py", "src/guard.py", "src/runtime_guard.py"})


def discover_learner_files(root: Path = ROOT) -> list[str]:
    found = (p.relative_to(root).as_posix() for p in sorted((root / "src").rglob("*.py")))
    return [f for f in found if f not in CONTROLLER_FILES]


def discover_shell_files(root: Path = ROOT) -> list[str]:
    return [p.relative_to(root).as_posix() for p in sorted((root / "kaggle").glob("*.py"))]


LEARNER_FILES = discover_learner_files()
SHELL_FILES = discover_shell_files()


def game_ids() -> frozenset:
    ids = set(PUBLIC_GAME_IDS)
    try:
        split = json.loads((ROOT / "configs" / "splits.json").read_text())
        for key in ("dev", "reserved"):
            ids.update(split.get(key, []))
    except (OSError, ValueError):
        pass
    return frozenset(ids)


def _mentions_game(s: str, ids: frozenset) -> bool:
    low = s.lower()
    return any(low == g or low.startswith(g + "-") or low.startswith(g + "_") for g in ids)


def scan_source(text: str, name: str = "<src>", *, strict: bool = True) -> list[str]:
    """Return human-readable violations for one source text."""
    ids = game_ids()
    out: list[str] = []
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES and strict:
                    out.append(f"{name}:{node.lineno}: imports {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_MODULES and strict:
                out.append(f"{name}:{node.lineno}: imports from {node.module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if any(f in s for f in FORBIDDEN_STRINGS):
                out.append(f"{name}:{node.lineno}: forbidden resource string {s[:40]!r}")
            if _mentions_game(s.strip(), ids) or any(s.strip() == g for g in ids):
                out.append(f"{name}:{node.lineno}: game id literal {s[:40]!r}")
        elif strict and isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            out.append(f"{name}:{node.lineno}: attribute .{node.attr}")
        elif strict and isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"{name}:{node.lineno}: name {node.id}")
        elif strict and isinstance(node, ast.arg) and node.arg in FORBIDDEN_NAMES:
            out.append(f"{name}:{node.lineno}: parameter {node.arg}")
    return out


def scan_files(learner_files=LEARNER_FILES, shell_files=SHELL_FILES) -> list[str]:
    out: list[str] = []
    for rel, strict in [(f, True) for f in learner_files] + [(f, False) for f in shell_files]:
        path = ROOT / rel
        if path.exists():
            out += scan_source(path.read_text(encoding="utf-8"), rel, strict=strict)
    return out


if __name__ == "__main__":
    problems = scan_files()
    print("\n".join(problems) if problems else "agent guard: clean")
    raise SystemExit(1 if problems else 0)
