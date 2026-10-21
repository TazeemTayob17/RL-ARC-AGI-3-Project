"""Controller-side resolution of the DQN prior checkpoint and seed offset.

This is the ONLY place that maps a game id to a checkpoint. The learner (src/dqn_v1) receives a path, a fold number
and a seed offset and never sees the game id (anti-cheating rule; src/guard.py enforces it).

Public games use the cross-validation fold whose training folds exclude that game (the selected-baseline protocol, so
Phase 3 results are reproduced exactly). Any other game id (private competition games) uses ONE predeclared
deployment checkpoint, DEPLOYMENT_FOLD. That checkpoint was trained on the other four public folds, so it has seen
public-game data; this is disclosed, and replacing it with a dev-games-only checkpoint needs the teammate's
training code (see docs/PROJECT_PLAN.md 2a).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "dqn_v1"
CHECKPOINT_STEM = "E-onehotcnn__T-stack2__A-vector__U-effect"
DEPLOYMENT_FOLD = 0

PUBLIC_GAMES = [
    "ar25", "bp35", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "r11l", "s5i5",
    "sb26", "sk48", "su15", "tn36", "tr87", "tu93", "vc33", "cd82", "ls20", "m0r0",
    "sp80", "re86", "cn04", "wa30", "sc25",
]
PUBLIC_FOLDS = [
    ["bp35", "sb26", "tu93", "sk48", "r11l"],
    ["vc33", "tn36", "tr87", "wa30", "s5i5"],
    ["lf52", "ls20", "su15", "ar25", "sp80"],
    ["dc22", "re86", "ka59", "lp85", "g50t"],
    ["sc25", "ft09", "cn04", "cd82", "m0r0"],
]


@dataclass(frozen=True)
class PriorSpec:
    path: Path
    fold: int
    seed_offset: int
    deployment: bool          # True when the game is not a public game (fallback checkpoint)


def resolve_prior(game: str) -> PriorSpec:
    """Game id -> (checkpoint path, fold, seed offset). Seeds for public games match the Phase 3 protocol exactly."""
    base = game.split("-")[0]                      # competition ids look like "ls20-9607627b"
    for fold, games in enumerate(PUBLIC_FOLDS):
        if base in games:
            offset = PUBLIC_GAMES.index(base) * 9173
            return PriorSpec(_path(fold), fold, offset, False)
    offset = (zlib.crc32(base.encode()) % len(PUBLIC_GAMES)) * 9173
    return PriorSpec(_path(DEPLOYMENT_FOLD), DEPLOYMENT_FOLD, offset, True)


def _path(fold: int) -> Path:
    return ASSETS / "checkpoints" / f"{CHECKPOINT_STEM}__f{fold}__s11.pt"
