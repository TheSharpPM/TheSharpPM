"""Tests for the shared article cache.

The cache exists so Agent Sharp can read on Sunday what the aggregator
fetched during the week. What matters here is that it never serves the
wrong text, never serves stale text, and never takes a pipeline down
when it fails - it is an optimisation, and an optimisation that can
break the daily digest is not worth having.

Run with:  python3 scripts/tests/test_article_cache.py
"""
import atexit
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Results, load_agent_sharp  # noqa: E402

load_agent_sharp()          # stubs requests/feedparser/groq if absent
import article_cache        # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="article_cache_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
article_cache.CACHE_DIR = _TMP

URL = "https://www.svpg.com/the-ai-productivity-paradox/"
TTL = article_cache.CACHE_TTL_SECONDS


def main():
    r = Results()

    print("\n--- round trip ---")
    article_cache.put(URL, "The real problem with the project model...",
                      title="The AI Productivity Paradox", source="SVPG")
    got = article_cache.get(URL)
    r.check("a stored body reads back", got is not None
            and got["text"].startswith("The real problem"), got)
    r.check("title and source are kept for the seen-sources digest",
            got["title"] == "The AI Productivity Paradox" and got["source"] == "SVPG")
    r.check("a URL never stored is a miss",
            article_cache.get("https://svpg.com/never-seen") is None)

    print("\n--- staleness ---")
    article_cache.put(URL, "week-old text", now=time.time() - TTL - 60)
    r.check("an entry past the TTL is a miss", article_cache.get(URL) is None)
    r.check("...and prune deletes it", article_cache.prune() == 1)
    article_cache.put(URL, "fresh", now=time.time() - 60)
    r.check("an entry inside the TTL survives prune",
            article_cache.prune() == 0 and article_cache.get(URL) is not None)

    print("\n--- never serve the wrong article ---")
    # A hand-edited or colliding entry must not surface another URL's
    # text: the agent would quote it and attribute it to the wrong piece.
    path = _TMP / f"{article_cache._key(URL)}.json"
    entry = json.loads(path.read_text())
    entry["url"] = "https://example.com/somewhere-else"
    path.write_text(json.dumps(entry))
    r.check("an entry whose url does not match the key is a miss",
            article_cache.get(URL) is None)

    path.write_text("{ this is not json")
    r.check("unreadable JSON is a miss, not a crash",
            article_cache.get(URL) is None)
    r.check("...and prune removes it", article_cache.prune() == 1)

    print("\n--- writes never break the caller ---")
    r.check("empty text is not stored", article_cache.put(URL, "") is None)
    r.check("whitespace-only text is not stored",
            article_cache.put(URL, "   \n ") is None)
    broken = Path("/proc/nonexistent-dir-for-test/articles")
    real, article_cache.CACHE_DIR = article_cache.CACHE_DIR, broken
    r.check("an unwritable cache dir returns None instead of raising",
            article_cache.put(URL, "text") is None)
    article_cache.CACHE_DIR = real

    print("\n--- truncation ---")
    article_cache.put(URL, "x" * (article_cache.MAX_CACHED_CHARS + 5000))
    r.check("a huge body is capped at MAX_CACHED_CHARS",
            len(article_cache.get(URL)["text"]) == article_cache.MAX_CACHED_CHARS)

    print("\n--- load_all seeds a run in one pass ---")
    article_cache.put("https://medium.com/a", "alpha")
    article_cache.put("https://stratechery.com/b", "beta", now=time.time() - TTL - 1)
    everything = article_cache.load_all()
    r.check("live entries are returned", "https://medium.com/a" in everything)
    r.check("stale entries are not", "https://stratechery.com/b" not in everything)

    print("\n--- the count cap bounds a committed directory ---")
    # The TTL alone stopped being a bound: at three weeks, with the
    # aggregator backfilling the corpus, an active week of feeds could
    # push the committed cache well past where it should sit.
    for stale in _TMP.glob("*.json"):
        stale.unlink()
    now = time.time()
    original_cap = article_cache.MAX_CACHE_ENTRIES
    article_cache.MAX_CACHE_ENTRIES = 5
    try:
        # Eight live entries, each an hour older than the last.
        for i in range(8):
            article_cache.put(f"https://www.svpg.com/p/{i}", f"body {i}",
                              now=now - i * 3600)
        removed = article_cache.prune(now=now)
        survivors = {
            json.loads(path.read_text(encoding="utf-8"))["url"]
            for path in _TMP.glob("*.json")
        }
        r.check("prune drops the overflow", removed == 3, removed)
        r.check("exactly the cap survives", len(survivors) == 5, len(survivors))
        r.check("and it is the newest that survive",
                survivors == {f"https://www.svpg.com/p/{i}" for i in range(5)},
                sorted(survivors))
    finally:
        article_cache.MAX_CACHE_ENTRIES = original_cap

    r.check("the shipped cap leaves room for the recent window",
            article_cache.MAX_CACHE_ENTRIES >= 200,
            article_cache.MAX_CACHE_ENTRIES)

    print("\n--- what the daily run offers the cache ---")
    # Originally only new_items were candidates, so an item was visible
    # to the cache on the one day it entered data.json and never again -
    # the cache held 6% of the citable corpus. The backfill is what
    # closes that gap, so the ordering it promises is worth pinning.
    import aggregate

    def item(url, date):
        return {"url": url, "date": date}

    fresh = item("https://www.svpg.com/new", "2026-10-11")
    corpus = [
        item("https://www.producttalk.org/old", "2026-09-20"),
        item("https://www.lennysnewsletter.com/mid", "2026-10-01"),
        item("https://example-vendor-blog.com/spam", "2026-10-10"),
        item("", "2026-10-09"),
        fresh,
    ]
    ordered = aggregate.cache_candidates([fresh], corpus)
    urls = [i["url"] for i in ordered]
    r.check("a new arrival is offered first", urls[0] == fresh["url"], urls)
    r.check("it is not offered twice", urls.count(fresh["url"]) == 1, urls)
    r.check("the backfill runs newest-first",
            urls[1:] == ["https://www.lennysnewsletter.com/mid",
                         "https://www.producttalk.org/old"], urls)
    r.check("uncacheable domains are never offered",
            not any("vendor-blog" in u for u in urls), urls)

    undated = aggregate.cache_candidates(
        [], [item("https://www.svpg.com/a", ""), item("https://www.svpg.com/b", "2026-10-01")]
    )
    r.check("an undated item sorts last instead of crashing",
            [i["url"] for i in undated]
            == ["https://www.svpg.com/b", "https://www.svpg.com/a"],
            [i["url"] for i in undated])

    print("\n--- SSRF defence is shared, not re-implemented ---")
    for bad in ["http://169.254.169.254/latest/meta-data/",
                "http://localhost:8080/admin",
                "file:///etc/passwd",
                "ftp://example.com/x"]:
        ok, _ = article_cache.url_is_safe_to_fetch(bad)
        r.check(f"blocked: {bad[:44]}", not ok)
    ok, _ = article_cache.url_is_safe_to_fetch("https://www.svpg.com/x")
    r.check("a public https URL is allowed", ok)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
