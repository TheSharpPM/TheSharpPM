"""Tests for batch article fetching and the read-before-citing gate.

These two ship together on purpose. The gate - refusing to cite a piece
the agent never fetched - is the fix for a sources_read_ratio of 0.4,
but it makes the agent read more, and the 2026-09-06 run already used
13 of its 15 iterations. Batch fetching is what pays for it: three
articles per tool call instead of one.

Run with:  python3 scripts/tests/test_batch_fetch.py
"""
import atexit
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import REPO_ROOT, Results, load_agent_sharp  # noqa: E402

agent = load_agent_sharp()
import article_cache  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="batch_fetch_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_CACHE = _TMP / "cache"
_CACHE.mkdir()
article_cache.CACHE_DIR = _CACHE

# Publishing rewrites the edition index, so point the agent at a scratch
# copy of the real one: the rotation and theme gates need to read real
# history, but must not write to agent/.
_REAL_INDEX = (REPO_ROOT / "agent" / "index.json").read_bytes()
agent.AGENT_DIR = _TMP
agent.INDEX_FILE = _TMP / "index.json"

# Four citable domains, mirroring the proven fixture in
# test_publish_gates.py: must_reads and the contrarian must be distinct
# URLs on the whitelist, and hook_source must match one of them.
LENNY = "https://www.lennysnewsletter.com/p/some-real-piece"
SVPG = "https://www.svpg.com/articles/a-second-piece"
TALK = "https://www.producttalk.org/2026/09/a-third-piece"
MTP = "https://www.mindtheproduct.com/2026/09/fourth-piece"

for url in (LENNY, SVPG, TALK, MTP):
    article_cache.put(url, f"body of {url} " + "x" * 5000)


def reset():
    agent.INDEX_FILE.write_bytes(_REAL_INDEX)
    for stale in _TMP.glob("*.json"):
        if stale.name != "index.json":
            stale.unlink()
    agent.FETCHED_URLS.clear()
    agent.FETCHED_TEXT.clear()
    agent.SEEN_SOURCES.clear()
    agent.TOOL_RESULTS_LOG.clear()
    agent.TOOL_CALL_COUNTS.clear()
    # Gate 1/1b come before the read gate: without a plausible research
    # history every publish is refused for not having gathered anything,
    # which silently masks the gate actually under test.
    agent.TOOL_CALL_COUNTS.update(
        {"fetch_feeds": 4, "web_search": 3, "fetch_article": 3}
    )


def edition(**over):
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
    base.update(over)
    return base


def main():
    r = Results()

    print("--- batch shape ---")
    reset()
    res = agent.tool_fetch_article(urls=[LENNY, SVPG, TALK])
    r.check("a batch returns one entry per URL",
            res.get("count") == 3 and len(res["articles"]) == 3, res.keys())
    r.check("batch bodies are trimmed to the triage size",
            all(len(x["text"]) <= agent.MAX_BATCH_ARTICLE_CHARS
                for x in res["articles"]),
            [len(x["text"]) for x in res["articles"]])

    # The whole point of the triage size: three bodies must survive the
    # sliding-window trim that caps a recent tool result. Measured on
    # realistically long URLs, because the budget is the serialised JSON
    # - keys and URLs included - not the bodies alone.
    long_urls = [
        "https://www.producttalk.org/2026/09/continuous-discovery-habits-the-assumption-mapping-guide-for-teams/",
        "https://www.lennysnewsletter.com/p/how-the-best-product-teams-actually-run-their-discovery-process",
        "https://www.svpg.com/articles/the-product-model-transformation-and-why-most-attempts-stall-early/",
    ]
    for u in long_urls:
        article_cache.put(u, "x" * 5000)
    payload = len(json.dumps(agent.tool_fetch_article(urls=long_urls),
                             ensure_ascii=False))
    r.check(f"a 3-URL batch of long URLs ({payload} chars) fits under the "
            f"trim cap ({agent.RECENT_TOOL_RESULT_CAP_CHARS})",
            payload <= agent.RECENT_TOOL_RESULT_CAP_CHARS, payload)

    r.check("a single url still gets the full slice",
            len(agent.tool_fetch_article(url=LENNY)["text"])
            == agent.MAX_ARTICLE_CHARS)
    r.check("more than the cap is refused",
            "error" in agent.tool_fetch_article(urls=[LENNY, SVPG, TALK, MTP]))
    r.check("neither url nor urls is an error",
            "error" in agent.tool_fetch_article())
    r.check("an empty list is an error",
            "error" in agent.tool_fetch_article(urls=[]))

    print("\n--- a batch read counts as having read ---")
    reset()
    res = agent.tool_fetch_article(urls=[LENNY, SVPG])
    agent._record_seen_sources("fetch_article", {"urls": [LENNY, SVPG]}, res)
    r.check("every article in a batch lands in FETCHED_TEXT",
            LENNY in agent.FETCHED_TEXT and SVPG in agent.FETCHED_TEXT,
            list(agent.FETCHED_TEXT))
    r.check("FETCHED_TEXT holds the FULL body, not the triage slice",
            len(agent.FETCHED_TEXT[LENNY]) > agent.MAX_BATCH_ARTICLE_CHARS,
            len(agent.FETCHED_TEXT[LENNY]))

    print("\n--- dedup across batches ---")
    reset()
    agent.execute_tool("fetch_article", {"urls": [LENNY, SVPG]})
    again = agent.execute_tool("fetch_article", {"urls": [LENNY, SVPG]})
    r.check("a fully repeated batch is refused",
            "duplicate_fetch" in str(again.get("error", "")), again)
    partial = agent.execute_tool("fetch_article", {"urls": [LENNY, TALK]})
    r.check("a partly-new batch still fetches the new URL",
            "error" not in partial
            and [x["url"] for x in partial["articles"]] == [TALK], partial)

    print("\n--- a failed fetch is remembered, so it is not retried ---")
    # An uncacheable, unsafe target: no network, and _fetch_one must still
    # return the url alongside the error or execute_tool cannot record
    # which URL broke - which is how a dead link burns a second iteration.
    bad = "http://localhost:8080/admin"
    reset()
    out = agent.execute_tool("fetch_article", {"url": bad})
    r.check("a single failed fetch carries its url", out.get("url") == bad, out)
    r.check("a single failed fetch is recorded as errored",
            agent.FETCHED_URLS.get(bad, "").startswith("errored"),
            dict(agent.FETCHED_URLS))
    r.check("retrying a failed url is refused as a duplicate",
            "duplicate_fetch" in str(
                agent.execute_tool("fetch_article", {"url": bad}).get("error", "")))

    reset()
    agent.execute_tool("fetch_article", {"urls": [LENNY, bad]})
    r.check("in a batch, only the failing member is marked errored",
            agent.FETCHED_URLS.get(LENNY) == "fetched successfully"
            and agent.FETCHED_URLS.get(bad, "").startswith("errored"),
            dict(agent.FETCHED_URLS))

    print("\n--- Gate 10b: read before you cite ---")
    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    reset()
    for u in (LENNY, SVPG, TALK, MTP):
        agent.SEEN_SOURCES[u] = {"title": "T", "source": "S", "date": recent}
    # Read only three of the four cited pieces.
    for u in (LENNY, SVPG, TALK):
        agent.FETCHED_TEXT[u] = "body"
    out = agent.tool_publish_edition(**edition())
    err = str(out.get("error", ""))
    r.check("citing an unread contrarian is refused",
            err.startswith("Refusing to publish") and "not read" in err, err[:200])
    r.check("the error names the unread piece", "'D'" in err, err[:200])
    r.check("the error points at the batch call", "urls=[...]" in err, err[:300])

    print("\n--- and does not fire when everything was read ---")
    agent.FETCHED_TEXT[MTP] = "body"
    out = agent.tool_publish_edition(**edition())
    r.check("no unread-citation rejection once all four are read",
            "not read" not in str(out.get("error", "")), out)
    r.check("and that edition actually publishes",
            not (isinstance(out, dict) and out.get("error")),
            out.get("error") if isinstance(out, dict) else None)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
