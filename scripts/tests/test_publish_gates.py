"""Tests for the publish-time quality gates in tool_publish_edition.

Covers the four gates that check the sourcing of an edition:

  Gate 10  citation provenance - cited URLs were actually seen this run
  Gate 11  recency             - cited sources are not stale
  Gate 12  pull quotes         - quotes appear verbatim in the article
  Gate 13  theme repetition    - headline is not a recent theme reworded

Each gate is checked in both directions: it rejects what it should, and
it stays out of the way otherwise. The non-blocking cases matter as much
as the rejections - a gate that fires on missing data would cost a run.

Run with:  python3 scripts/tests/test_publish_gates.py
"""
import atexit
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import REPO_ROOT, Results, load_agent_sharp  # noqa: E402

agent = load_agent_sharp()

# tool_publish_edition writes the edition and rewrites index.json on
# success. Point it at a scratch directory so a passing test can never
# mutate agent/ in the repo.
#
# The real index is read via REPO_ROOT, not agent.AGENT_DIR, because the
# module defines that as a bare relative Path("agent") - correct for the
# workflow, which always runs from the repository root, but it would
# make these tests depend on the caller's working directory.
_TMP = Path(tempfile.mkdtemp(prefix="agent_sharp_gates_"))
# Registered at creation rather than wrapped around main(), so the
# directory is removed even if setup below fails before main() runs.
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_REAL_INDEX = (REPO_ROOT / "agent" / "index.json").read_bytes()
agent.AGENT_DIR = _TMP
agent.INDEX_FILE = _TMP / "index.json"


def days_ago(n):
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


# Four distinct trusted domains: must_reads and the contrarian have to be
# different URLs, and hook_source has to match one of them.
LENNY = "https://www.lennysnewsletter.com/p/some-real-piece"
SVPG = "https://www.svpg.com/articles/a-second-piece"
TALK = "https://www.producttalk.org/2026/09/a-third-piece"
MTP = "https://www.mindtheproduct.com/2026/09/fourth-piece"

ARTICLE_BODY = (
    "Teams keep shipping features nobody asked for. "
    "The discovery habit is what separates the good from the loud. "
    "Measure outcomes, not output velocity."
)


def reset(dates=None, bodies=None):
    """Restore a pristine index and run state before each case.

    A test whose edition publishes rewrites the scratch index, which
    would then trip the target_audience and hook_source rotation gates
    for every later case.
    """
    agent.INDEX_FILE.write_bytes(_REAL_INDEX)
    for stale in _TMP.glob("*.json"):
        if stale.name != "index.json":
            stale.unlink()
    agent.SEEN_SOURCES.clear()
    agent.FETCHED_TEXT.clear()
    agent.TOOL_RESULTS_LOG.clear()
    agent.TOOL_CALL_COUNTS.clear()
    agent.TOOL_CALL_COUNTS.update(
        {"fetch_feeds": 4, "web_search": 3, "fetch_article": 3}
    )
    for url in (LENNY, SVPG, TALK, MTP):
        agent.SEEN_SOURCES[url] = {
            "title": "T", "source": "S",
            "date": (dates or {}).get(url, days_ago(3)),
        }
    agent.FETCHED_TEXT.update(
        bodies if bodies is not None else {LENNY: ARTICLE_BODY}
    )


def edition(**overrides):
    """A valid edition that clears every pre-existing gate.

    target_audience and hook_source are picked so the rotation gates
    (which read the real published history) do not fire first and mask
    the gate under test.
    """
    base = dict(
        headline_theme="Discovery habits beat feature velocity",
        editorial=" ".join(
            ["Teams keep shipping features nobody asked for."] * 40
        ),
        must_reads=[
            {"title": "A", "url": LENNY, "source": "Lenny's Newsletter",
             "why": "The author is right that discovery is a habit, not a phase."},
            {"title": "B", "url": SVPG, "source": "SVPG",
             "why": "Wrong on cadence, right that outcomes beat output."},
            {"title": "C", "url": TALK, "source": "Product Talk",
             "why": "The sharpest case against velocity worship this month."},
        ],
        key_takeaways=["one", "two", "three"],
        pm_homework=["do a thing", "do another thing", "and a third"],
        contrarian={"title": "D", "url": MTP, "source": "Mind the Product",
                    "note": "Argues the opposite and mostly lands the point."},
        target_audience="Lead/Principal PM",
        hook_source="Product Talk",
    )
    base.update(overrides)
    return base


def publish(**overrides):
    return agent.tool_publish_edition(**edition(**overrides))


def rejected_for(result, needle):
    """True when the publish was refused for the expected reason."""
    return (
        isinstance(result, dict)
        and str(result.get("error", "")).startswith("Refusing to publish")
        and needle in result["error"]
    )


def main():
    r = Results()

    print("--- baseline ---")
    reset()
    out = publish()
    r.check("a clean edition clears every gate and publishes",
            not (isinstance(out, dict) and out.get("error")),
            out.get("error") if isinstance(out, dict) else None)

    print("\n--- Gate 10: citation provenance ---")
    reset()
    invented = edition()["must_reads"][:2] + [{
        "title": "Fake",
        "url": "https://www.lennysnewsletter.com/p/invented-slug",
        "source": "Lenny's Newsletter",
        "why": "A hallucinated but entirely plausible link.",
    }]
    out = publish(must_reads=invented, hook_source="Lenny's Newsletter")
    r.check("a hallucinated URL on a trusted domain is rejected",
            rejected_for(out, "never returned by"), out.get("error"))

    reset()
    out = publish(contrarian={
        "title": "D", "url": "https://www.svpg.com/articles/invented",
        "source": "SVPG", "note": "n" * 40})
    r.check("a hallucinated contrarian URL is rejected",
            rejected_for(out, "never returned by"), out.get("error"))

    print("\n--- Gate 11: recency ---")
    reset(dates={LENNY: days_ago(400), SVPG: days_ago(2),
                 TALK: days_ago(4), MTP: days_ago(5)})
    out = publish()
    r.check("a source past the age limit is rejected",
            rejected_for(out, "published 400 days ago"), out.get("error"))

    reset(dates={LENNY: days_ago(agent.MAX_SOURCE_AGE_DAYS - 1),
                 SVPG: days_ago(2), TALK: days_ago(4), MTP: days_ago(5)})
    out = publish()
    r.check("a source just inside the age limit passes",
            not rejected_for(out, "days ago"), out.get("error"))

    reset(dates={url: "" for url in (LENNY, SVPG, TALK, MTP)})
    out = publish()
    r.check("an unknown publication date does not block publishing",
            not rejected_for(out, "days ago"), out.get("error"))

    reset(dates={LENNY: "not-a-date", SVPG: days_ago(1),
                 TALK: days_ago(1), MTP: days_ago(1)})
    out = publish()
    r.check("an unparseable publication date does not block publishing",
            not rejected_for(out, "days ago"), out.get("error"))

    print("\n--- Gate 12: pull quotes ---")
    reset()
    quoted = edition()["must_reads"]
    quoted[0]["pull_quote"] = (
        "The discovery habit is what separates the good from the loud."
    )
    out = publish(must_reads=quoted)
    r.check("a verbatim pull quote passes",
            not rejected_for(out, "does not appear"), out.get("error"))

    reset()
    quoted = edition()["must_reads"]
    # Same words, different typography: em dash for a comma, added bang.
    quoted[0]["pull_quote"] = "Measure outcomes — not output velocity!"
    out = publish(must_reads=quoted)
    r.check("typography differences alone still pass",
            not rejected_for(out, "does not appear"), out.get("error"))

    reset()
    quoted = edition()["must_reads"]
    quoted[0]["pull_quote"] = (
        "Every team that ships fast wins the quarter, full stop."
    )
    out = publish(must_reads=quoted)
    r.check("a fabricated pull quote is rejected",
            rejected_for(out, "does not appear"), out.get("error"))

    reset()
    quoted = edition()["must_reads"]
    # SVPG was seen in a feed but never fetched, so there is no body to
    # check the quote against and the gate must not guess.
    quoted[1]["pull_quote"] = (
        "Something unverifiable because we only ever saw a summary."
    )
    out = publish(must_reads=quoted)
    r.check("a quote on a source that was never fetched is not checked",
            not rejected_for(out, "does not appear"), out.get("error"))

    reset()
    quoted = edition()["must_reads"]
    quoted[0]["pull_quote"] = "loud. Measure"
    out = publish(must_reads=quoted)
    r.check("a quote below the minimum length is skipped, not rejected",
            not rejected_for(out, "does not appear"), out.get("error"))

    print("\n--- Gate 13: theme repetition ---")
    reset()  # restore the pristine index BEFORE reading what it contains
    recent = agent._recent_headline_themes()
    print(f"  lookback themes: {recent}")
    out = publish(headline_theme=recent[0] if recent else "Nothing to compare")
    r.check("re-running the most recent headline verbatim is rejected",
            rejected_for(out, "repeats the theme"), out.get("error"))

    reset()
    out = publish(headline_theme="Pricing experiments nobody bothered to design")
    r.check("a genuinely different theme passes",
            not rejected_for(out, "repeats the theme"), out.get("error"))

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
