"""Tests for the editorial quality signals emitted alongside a publish.

The publish gates answer "is this publishable?". These metrics answer
"is it any good?" - the question the trace could not previously show at
all. Every metric here exists because the 2026-09-06 edition failed it
while passing all thirteen gates, so each test below replays a real
failure rather than an invented one:

  sources_read_ratio  three of five cited pieces were never fetched
  recycled_citations  three of five had already run in the last month
  stats_used          five real figures were available, none were used
  topic_overlap       the fourth straight edition on evaluation, which
                      headline_overlap scored at 0.0

Also covers Gate 6's descriptive-opener rule, which those same must_reads
walked straight through.

Run with:  python3 scripts/tests/test_quality_metrics.py
"""
import atexit
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Results, load_agent_sharp  # noqa: E402

agent = load_agent_sharp()

_TMP = Path(tempfile.mkdtemp(prefix="agent_sharp_quality_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
agent.AGENT_DIR = _TMP
agent.INDEX_FILE = _TMP / "index.json"


def seed_editions(editions):
    """Write editions to the scratch agent dir and index them, newest
    first. Mirrors what tool_publish_edition leaves behind."""
    index = {"editions": []}
    for ed in editions:
        (_TMP / f"{ed['edition']}.json").write_text(
            json.dumps(ed), encoding="utf-8"
        )
        index["editions"].append({
            "date": ed["edition"],
            "headline_theme": ed["headline_theme"],
            "file": f"agent/{ed['edition']}.json",
        })
    agent.INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")


def edition(date, theme, titles, urls=None):
    urls = urls or [f"https://svpg.com/{i}" for i in range(len(titles))]
    return {
        "edition": date,
        "headline_theme": theme,
        "must_reads": [
            {"title": t, "url": u} for t, u in zip(titles, urls)
        ],
        "contrarian": None,
    }


def reset_run_state(fetched=None, results=None):
    agent.FETCHED_TEXT.clear()
    agent.FETCHED_TEXT.update(fetched or {})
    agent.TOOL_RESULTS_LOG.clear()
    agent.TOOL_RESULTS_LOG.extend(results or [])


def main():
    r = Results()

    # ── Gate 6: descriptive openers ──────────────────────────────────
    print("\n--- Gate 6: 'why' must react, not summarise ---")

    # The three that shipped on 2026-09-06 past the phrase list.
    for why, verb in [
        ("Provides a concrete, step-by-step framework that transforms "
         "vague optimism into measurable outcomes.", "Provides"),
        ("Shows why speed metrics are misleading and why disciplined "
         "evaluation is the only path to real impact.", "Shows"),
        ("Highlights cultural resistance and argues that solid data is "
         "the antidote to habit-driven skepticism.", "Highlights"),
    ]:
        m = agent._DESCRIPTIVE_WHY_RE.match(why)
        r.check(
            f"a 'why' opening with '{verb}' is rejected",
            m is not None and m.group(1).lower() == verb.lower(),
            why,
        )

    # "This article provides" was already caught by the phrase list; the
    # opener rule must catch the subject-stripped form too.
    r.check(
        "the 'This article provides' form is still rejected",
        agent._DESCRIPTIVE_WHY_RE.match("This article provides a framework.")
        is not None,
    )

    # False positives are the expensive failure here: a rejected publish
    # costs a whole retry, and the phrase list already had to drop bare
    # "shows how" for exactly this reason.
    for why in [
        "Cagan shows how governance rots when velocity is the metric.",
        "The author provides cover for leaders who never wanted evals.",
        "Azhar is wrong that the limits are physical - they are commercial.",
        "It exposes the myth that bigger models deliver more value.",
        "Biddle overstates the case, but the north-star point lands.",
    ]:
        r.check(
            f"opinionated 'why' passes: {why[:44]}...",
            agent._DESCRIPTIVE_WHY_RE.match(why) is None,
            why,
        )

    # ── sources_read_ratio ───────────────────────────────────────────
    print("\n--- cited sources the agent actually read ---")
    seed_editions([])
    reset_run_state(fetched={"https://svpg.com/a": "body of a"})

    m = agent._quality_metrics(
        "A brand new angle on pricing",
        "No numbers here at all.",
        [{"title": "A", "url": "https://svpg.com/a"},
         {"title": "B", "url": "https://svpg.com/b"}],
        {"title": "C", "url": "https://stratechery.com/c"},
    )
    r.check(
        "only fetched pieces count as read",
        m["cited_sources_read"] == 1 and m["cited_sources_total"] == 3,
        m,
    )
    r.check(
        "sources_read_ratio reports the fraction",
        m["sources_read_ratio"] == round(1 / 3, 3),
        m["sources_read_ratio"],
    )

    reset_run_state(fetched={
        "https://svpg.com/a": "x", "https://svpg.com/b": "y",
    })
    m = agent._quality_metrics(
        "Another angle", "",
        [{"title": "A", "url": "https://svpg.com/a"},
         {"title": "B", "url": "https://svpg.com/b"}],
        None,
    )
    r.check(
        "an edition citing only what it read scores 1.0",
        m["sources_read_ratio"] == 1.0,
        m["sources_read_ratio"],
    )

    # ── recycled_citations ───────────────────────────────────────────
    print("\n--- citations recycled from recent editions ---")
    seed_editions([
        edition("2026-08-31", "Speed Isn't Enough",
                ["The AI Productivity Paradox"],
                ["https://www.svpg.com/the-ai-productivity-paradox/"]),
        edition("2026-08-23", "When Experiments Stifle Vision",
                ["The Call to Adventure"],
                ["https://www.producttalk.org/the-call-to-adventure/"]),
    ])
    reset_run_state()
    m = agent._quality_metrics(
        "Pricing is the only lever left", "",
        [{"title": "The AI Productivity Paradox",
          "url": "https://www.svpg.com/the-ai-productivity-paradox/"},
         {"title": "Something genuinely new",
          "url": "https://stratechery.com/new"}],
        {"title": "The Call to Adventure",
         "url": "https://www.producttalk.org/the-call-to-adventure/"},
    )
    r.check(
        "re-cited pieces from recent editions are counted",
        m["recycled_citations"] == 2, m,
    )

    # A past edition written before urls were recorded still has titles.
    seed_editions([
        {"edition": "2026-08-31", "headline_theme": "Old one",
         "must_reads": [{"title": "The AI Productivity Paradox"}],
         "contrarian": None},
    ])
    m = agent._quality_metrics(
        "Fresh angle", "",
        [{"title": "The AI Productivity Paradox",
          "url": "https://www.svpg.com/the-ai-productivity-paradox/"}],
        None,
    )
    r.check(
        "recycling is caught by title when the old edition has no url",
        m["recycled_citations"] == 1, m,
    )

    # ── the self-comparison trap ─────────────────────────────────────
    print("\n--- the edition must not be compared against itself ---")
    # These metrics run AFTER the index write, unlike the gates, so the
    # edition being judged is already the newest entry in index.json.
    today = agent.edition_date()
    seed_editions([
        edition(today, "AI Evaluation Over Hype", ["Piece One"],
                ["https://svpg.com/one"]),
        edition("2026-08-31", "Speed Isn't Enough", ["Other"],
                ["https://svpg.com/other"]),
    ])
    reset_run_state()
    m = agent._quality_metrics(
        "AI Evaluation Over Hype", "",
        [{"title": "Piece One", "url": "https://svpg.com/one"}],
        None,
    )
    r.check(
        "an edition does not count as recycling itself",
        m["recycled_citations"] == 0, m,
    )
    r.check(
        "an edition does not overlap itself at 1.0",
        m["headline_overlap"] < 1.0, m["headline_overlap"],
    )

    # ── stats available vs used ──────────────────────────────────────
    print("\n--- real figures available vs figures used ---")
    seed_editions([])
    reset_run_state(results=[
        "89% of executives say AI increased speed, but only 6% see ROI",
        "token share hit 62% , up from 28% two months earlier",
    ])
    m = agent._quality_metrics(
        "Some theme",
        "Evaluation is the gate. No figures appear in this editorial.",
        [{"title": "A", "url": "https://svpg.com/a"}], None,
    )
    r.check(
        "figures present in the research are counted",
        m["stats_available"] >= 4, m,
    )
    r.check(
        "an editorial that dodges every number scores zero",
        m["stats_used"] == 0, m,
    )

    m = agent._quality_metrics(
        "Some theme",
        "Only 6% can point to ROI, which is the whole problem.",
        [{"title": "A", "url": "https://svpg.com/a"}], None,
    )
    r.check(
        "a used figure is counted",
        m["stats_used"] == 1, m,
    )

    # ── topic_overlap sees what headline_overlap misses ──────────────
    print("\n--- topic drift the headline gate cannot see ---")
    seed_editions([
        edition("2026-08-31", "Speed Isn't Enough: Rethink AI Success Metrics",
                ["The AI Productivity Paradox", "AI Evals for Product Teams"]),
    ])
    reset_run_state()
    m = agent._quality_metrics(
        "AI Evaluation Over Hype: The New Non-Negotiable", "",
        [{"title": "AI Evals: A Hands-On Guide for Product Teams",
          "url": "https://www.producttalk.org/ai-evals/"},
         {"title": "The AI Productivity Paradox",
          "url": "https://www.svpg.com/the-ai-productivity-paradox/"}],
        None,
    )
    r.check(
        "headline wording alone looks unrelated",
        m["headline_overlap"] < agent.HEADLINE_THEME_OVERLAP,
        m["headline_overlap"],
    )
    r.check(
        "whole-edition vocabulary shows the repetition",
        m["topic_overlap"] > m["headline_overlap"], m,
    )

    # ── no history, no crash ─────────────────────────────────────────
    print("\n--- first run, empty history ---")
    agent.INDEX_FILE.unlink(missing_ok=True)
    reset_run_state()
    m = agent._quality_metrics("First ever theme", "", [], None)
    r.check(
        "an empty edition on a fresh install returns zeros, not an error",
        m["sources_read_ratio"] == 0.0 and m["recycled_citations"] == 0,
        m,
    )

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
