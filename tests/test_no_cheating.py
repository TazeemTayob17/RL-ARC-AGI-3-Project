"""Phase 4 step 1: the agent guard passes on clean agent code and fails on planted violations."""
import textwrap

from src.guard import scan_files, scan_source


def test_agent_code_is_clean():
    assert scan_files() == []


def test_guard_flags_planted_violations():
    bad = textwrap.dedent('''
        import arcengine
        from pathlib import Path

        class Cheater:
            def __init__(self, game):
                self.game = game
                self.table = {"ls20": [1, 2, 3]}
            def act(self, obs):
                return Path("environment_files/ls20/game.py").read_text()
    ''')
    problems = scan_source(bad, "planted.py")
    joined = "\n".join(problems)
    for needle in ("imports arcengine", "parameter game", "attribute .game", "game id literal",
                   "forbidden resource string"):
        assert needle in joined, (needle, joined)


def test_guard_allows_ordinary_code():
    ok = textwrap.dedent('''
        import numpy as np
        def act(obs):
            return int(np.argmax(obs.mask))
    ''')
    assert scan_source(ok, "ok.py") == []
