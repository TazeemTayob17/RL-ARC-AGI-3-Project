"""Runtime anti-cheating audit (complements the static scan in guard.py).

While a learner method is running (`learner_section()`), the process may not read game files, touch the network,
start processes, or import the SDK. Violations raise `CheatingViolation` (a BaseException, so a learner's own
`except Exception` cannot swallow it) and stop the run: the halt-and-report rule. SDK code and the run controller run
outside the section and are not restricted. Audit hooks cannot be removed, so the hook is installed once and is inert
outside a section.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

FORBIDDEN_PATH_PARTS = ("environment_files", "baseline_actions", "human_actions")
FORBIDDEN_PATH_SEGMENTS = {"arc_agi", "arcengine"}
FORBIDDEN_IMPORTS = {"arc_agi", "arcengine"}
FORBIDDEN_EVENTS = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.bind",
                    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"}
PATH_EVENTS = {"open", "os.listdir", "os.scandir", "os.stat"}


class CheatingViolation(BaseException):
    pass


_state = threading.local()
_installed = False
checked_events = 0
violations: list = []


def _bad_path(p) -> bool:
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    if not isinstance(p, str):
        return False
    low = p.replace("\\", "/").lower()
    if any(s in low for s in FORBIDDEN_PATH_PARTS):
        return True
    return any(seg in FORBIDDEN_PATH_SEGMENTS for seg in low.split("/"))


def _hook(event: str, args: tuple) -> None:
    global checked_events
    if not getattr(_state, "depth", 0):
        return
    reason = None
    if event in PATH_EVENTS:
        checked_events += 1
        if args and _bad_path(args[0]):
            reason = f"{event} {args[0]!r}"
    elif event in FORBIDDEN_EVENTS:
        checked_events += 1
        reason = event
    elif event == "import":
        checked_events += 1
        if args and str(args[0]).split(".")[0] in FORBIDDEN_IMPORTS:
            reason = f"import {args[0]}"
    if reason:
        violations.append(reason)
        raise CheatingViolation(f"learner code attempted a forbidden operation during a run: {reason}")


def install() -> None:
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


@contextmanager
def learner_section():
    install()
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def summary() -> dict:
    return {"checked_events": checked_events, "violations": list(violations)}
