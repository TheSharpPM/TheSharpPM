"""Tests for what a tool call costs the run, and what it buys.

Iterations are the binding constraint: the 2026-09-06 run used 13 of
its 15. Two numbers govern that budget and they are NOT the same
number, which is the thing these tests pin down:

  _research_calls()  - source material only. Gate 1b's minimum.
  _calls_spent()     - every call that cost an iteration, read_memory
                       included. The force-publish budget.

Collapsing them either way causes a specific failure. Count read_memory
toward the minimum and the agent can satisfy "do not pick the theme
from one article" by re-reading its own archive. Leave it out of the
budget and three read_memory calls buy three extra iterations of drift.

Run with:  python3 scripts/tests/test_call_budget.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Results, load_agent_sharp  # noqa: E402

agent = load_agent_sharp()

MIN = agent.MIN_RESEARCH_CALLS
FORCE = agent.FORCE_PUBLISH_AT_RESEARCH_CALLS


def counts(**kw):
    agent.TOOL_CALL_COUNTS.clear()
    agent.TOOL_CALL_COUNTS.update(kw)


def main():
    r = Results()

    print("--- the two numbers are different ---")
    counts(fetch_feeds=2, web_search=1, fetch_article=3, read_memory=2)
    r.check("research counts only source material",
            agent._research_calls() == 6, agent._research_calls())
    r.check("spent counts read_memory too",
            agent._calls_spent() == 8, agent._calls_spent())

    counts(read_memory=5)
    r.check("read_memory alone is no research at all",
            agent._research_calls() == 0, agent._research_calls())
    r.check("...but it did spend five iterations",
            agent._calls_spent() == 5, agent._calls_spent())

    print("\n--- read_memory no longer buys extra iterations ---")
    # The old rule keyed the force-publish purely on source calls, so a
    # run could reach iteration 11 (8 research + 3 memory) with the
    # budget still unspent. Now the budget notices.
    counts(fetch_feeds=3, web_search=2, fetch_article=1, read_memory=2)
    r.check(f"{MIN} research + 2 memory = {FORCE} spent, so publish is forced",
            agent._should_force_publish(),
            (agent._research_calls(), agent._calls_spent()))

    counts(fetch_feeds=3, web_search=2, fetch_article=1)
    r.check(f"{MIN} research and no memory calls is not yet the budget",
            not agent._should_force_publish(),
            (agent._research_calls(), agent._calls_spent()))

    print("\n--- and never forces a publish Gate 1b would refuse ---")
    # The deadlock: pinned to publish_edition while the gate still says
    # no. Checked exhaustively rather than by example, because the
    # margin between the minimum and the budget is only two calls.
    deadlocks = []
    for research in range(0, FORCE + 6):
        for memory in range(0, 8):
            counts(fetch_feeds=research, read_memory=memory)
            if agent._should_force_publish() and agent._research_calls() < MIN:
                deadlocks.append((research, memory))
    r.check("no research/memory combination forces an unpublishable turn",
            not deadlocks, deadlocks[:5])

    # The guard is load-bearing: without the research half of the rule,
    # the combinations below WOULD deadlock. Proves the test above is
    # not vacuous.
    would_deadlock = [
        (res, mem)
        for res in range(0, FORCE)
        for mem in range(0, 8)
        if res < MIN and res + mem >= FORCE
    ]
    r.check("the guard is what prevents it (budget alone would deadlock)",
            len(would_deadlock) > 0, len(would_deadlock))

    print("\n--- repeat read_memory collapses to the digest ---")
    counts()
    first = agent.execute_tool("read_memory", {})
    r.check("the first call returns the full archive payload",
            "editions" in first and "note" not in first, list(first))

    second = agent.execute_tool("read_memory", {})
    r.check("the second call returns the digest, not the archive",
            "editions" not in second and "note" in second, list(second))
    r.check("the digest carries the three things memory is for",
            all(k in second for k in ("recent_headline_themes",
                                      "recent_target_audiences",
                                      "recent_hook_sources")), list(second))
    r.check("the digest is smaller than the payload it replaces",
            len(str(second)) < len(str(first)),
            (len(str(second)), len(str(first))))
    r.check("and it tells the agent to stop asking",
            "third time" in second["note"], second["note"][:80])

    # It must answer rather than refuse: the publish gates send the agent
    # to read_memory by name when they reject an audience or hook source,
    # so a hard refusal here could strand a run that needs the reminder.
    r.check("a repeat call is answered, not refused",
            "error" not in second, second)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
