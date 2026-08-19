"""Make the test suite hermetic.

``src/tools/search_tools.py`` calls ``load_dotenv()`` at import time, so simply
importing most of this project pulls real API keys into ``os.environ``. Several
tests are written around the assumption that reaching a model raises — which is
what makes them free, fast, and deterministic — and that assumption silently
stopped being true whenever an earlier test in the same session happened to
import the tools package.

The symptom was order-dependent: ``tests/test_eval.py::TestJudge`` passed alone
and failed after ``tests/test_pipeline.py``, because by then the key was loaded
and the "this must fail" call succeeded against the real API. The cost of that
bug is not a red test — it is that a full-suite run quietly spends money.

Scrubbing the credentials for the whole session makes the offline assumption
true by construction rather than by luck. Tests that want a key set one
themselves with ``monkeypatch.setenv``.
"""

import os

import pytest

#: Model credentials only. ``TAVILY_API_KEY`` is deliberately left alone:
#: ``tests/test_tools.py`` exercises the real search API on purpose and already
#: guards on the key's presence, and yfinance is likewise hit live. Those are
#: existing, deliberate integration tests. What was *not* deliberate is a
#: synthesis-tier model call escaping from a unit test, which is both the
#: expensive path and the one whose assertions assume it cannot happen.
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True, scope="session")
def _no_real_api_calls():
    """Remove credentials for the duration of the test session."""
    saved = {name: os.environ.pop(name, None) for name in _CREDENTIAL_VARS}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


@pytest.fixture(autouse=True)
def _rescrub_after_dotenv(monkeypatch):
    """Re-scrub per test.

    ``load_dotenv()`` runs at import time, and imports can happen part-way
    through a session as new test modules are collected — so a session-scoped
    scrub alone can be undone by the first test that imports the tools package.
    """
    for name in _CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
