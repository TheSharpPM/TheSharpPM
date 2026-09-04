"""Shared setup for the Agent Sharp tests.

The tests exercise pure logic - publish gates, span payload shaping - so
they must run without network access, without API keys, and without the
pinned runtime dependencies installed. Any dependency that is genuinely
installed is used as-is; only the missing ones get a stub, so the same
tests run locally on a bare checkout and in CI after
`pip install -r requirements.txt`.
"""
import os
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _stub_if_missing(name, **attrs):
    """Install a stub module only when the real one is not importable."""
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


def load_agent_sharp():
    """Import scripts/agent_sharp.py with its imports satisfied.

    Returns the module. Importing it is side-effect free: it reads
    GROQ_API_KEY from the environment but makes no calls until a tool
    or the run loop is invoked.
    """
    class _RequestException(Exception):
        pass

    _stub_if_missing("feedparser")
    _stub_if_missing("requests", RequestException=_RequestException)
    _stub_if_missing("groq", Groq=lambda **kwargs: None)
    _stub_if_missing("langfuse", Langfuse=object)

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

    import agent_sharp
    return agent_sharp


class Results:
    """Minimal pass/fail tally - avoids adding pytest as a dependency."""

    def __init__(self):
        self.rows = []

    def check(self, label, passed, detail=None):
        self.rows.append((label, passed))
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
        if not passed and detail:
            print(f"        actual: {str(detail)[:200]}")

    def report(self):
        """Print the tally. Returns a process exit code."""
        failed = [label for label, passed in self.rows if not passed]
        print()
        print(f"{len(self.rows) - len(failed)}/{len(self.rows)} passed")
        if failed:
            print("FAILURES:")
            for label in failed:
                print(f"  - {label}")
            return 1
        print("ALL TESTS PASS")
        return 0
