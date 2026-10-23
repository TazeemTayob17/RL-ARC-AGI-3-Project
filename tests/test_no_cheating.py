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


# ---- R7: auto-discovery and runtime audit
import socket

import pytest

from src import guard
from src.runtime_guard import CheatingViolation, learner_section, violations


def test_new_src_file_is_scanned_strictly(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "brand_new_learner.py").write_text("x = 1")
    (tmp_path / "src" / "run.py").write_text("x = 1")
    assert guard.discover_learner_files(tmp_path) == ["src/brand_new_learner.py"]


def test_current_learner_files_are_discovered():
    found = set(guard.LEARNER_FILES)
    for f in ("src/dqn.py", "src/ppo.py", "src/options.py", "src/dqn_v1/learner.py", "src/dqn_v1/prior.py"):
        assert f in found
    assert "src/run.py" not in found and "src/adapter.py" not in found


def test_runtime_guard_blocks_game_file_reads(tmp_path):
    target = tmp_path / "environment_files" / "x" / "game.py"
    target.parent.mkdir(parents=True)
    target.write_text("secret")
    n = len(violations)
    with pytest.raises(CheatingViolation):
        with learner_section():
            open(target).read()
    assert len(violations) == n + 1


def test_runtime_guard_blocks_network_and_sdk_import():
    with pytest.raises(CheatingViolation):
        with learner_section():
            socket.getaddrinfo("localhost", 80)
    from src import runtime_guard
    with pytest.raises(CheatingViolation):                 # a cached module raises no import event, so call the hook
        with learner_section():
            runtime_guard._hook("import", ("arcengine", None, None, None, None))


def test_runtime_guard_allows_ordinary_work_and_is_inert_outside(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_text("ok")
    with learner_section():
        assert f.read_text() == "ok"
    (tmp_path / "environment_files").mkdir()
    (tmp_path / "environment_files" / "a.txt").write_text("sdk side")   # outside a section: not restricted
