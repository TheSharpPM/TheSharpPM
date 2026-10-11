"""Tests for the MCP server's data layer.

These run with no dependencies at all - tspm_data.py imports nothing
but stdlib, which is the reason the protocol wiring was kept out of it.
The `mcp` package is NOT needed here, so these run in CI next to the
existing suites instead of needing a job of their own.

Everything is checked against a synthetic archive in a temp directory,
never the real data.json: a test that asserts "17 results for pricing"
starts failing the morning the aggregator finds an eighteenth.

Run with:  python3 scripts/tests/test_mcp_data.py
"""
import atexit
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp"))
from harness import Results  # noqa: E402

import tspm_data  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="mcp_data_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
tspm_data.REPO_ROOT = _TMP

ITEMS = [
    {"title": "Pricing ladders that work", "url": "https://a.com/1",
     "source": "SVPG", "date": "2026-10-10T00:00:00+00:00", "type": "article",
     "tags": ["Product Strategy"], "summary": "On willingness to pay."},
    {"title": "A discovery habit", "url": "https://a.com/2",
     "source": "Product Talk", "date": "2026-09-01T00:00:00+00:00",
     "type": "article", "tags": ["Product Strategy", "Career"],
     "summary": "Pricing gets a passing mention here."},
    {"title": "Senior PM role", "url": "https://a.com/3", "source": "Jobs",
     "date": "2026-08-01T00:00:00+00:00", "type": "job",
     "tags": ["Career"], "summary": "Hiring."},
]

EDITION = {
    "edition": "2026-10-11", "headline_theme": "Pricing is a research question",
    "target_audience": "Staff PM", "hook_source": "SVPG",
    "editorial": "Teams guess at pricing. " * 20,
    "key_takeaways": ["Charge for value"], "pm_homework": ["Interview 5 buyers"],
    "contrarian": {"title": "Against pricing science", "url": "https://a.com/c",
                   "note": "Argues the opposite."},
    "must_reads": [{"title": "Pricing ladders that work", "url": "https://a.com/1",
                    "why": "The only piece that does the math."}],
}
OLDER = {
    "edition": "2026-10-04", "headline_theme": "Discovery beats velocity",
    "target_audience": "Senior PM", "hook_source": "Product Talk",
    "editorial": "Velocity is not progress. " * 20,
    "key_takeaways": ["Talk to users"], "pm_homework": ["Run one interview"],
    "contrarian": {}, "must_reads": [],
}


def setup():
    (_TMP / "agent").mkdir(exist_ok=True)
    (_TMP / "data.json").write_text(json.dumps(
        {"updated": "now", "total": len(ITEMS), "items": ITEMS}), encoding="utf-8")
    (_TMP / "agent" / "index.json").write_text(json.dumps({"editions": [
        {"date": "2026-10-11"}, {"date": "2026-10-04"},
    ]}), encoding="utf-8")
    for doc in (EDITION, OLDER):
        (_TMP / "agent" / f"{doc['edition']}.json").write_text(
            json.dumps(doc), encoding="utf-8")


def main():
    setup()
    r = Results()

    print("--- list_tags: the discovery call ---")
    t = tspm_data.list_tags()
    r.check("counts every item", t["total_items"] == 3, t["total_items"])
    r.check("tags come back with counts, most common first",
            t["tags"][0] == {"tag": "Product Strategy", "count": 2}, t["tags"])
    r.check("types are listed too",
            {x["type"] for x in t["types"]} == {"article", "job"}, t["types"])
    r.check("and the window it covers",
            t["covers"] == {"from": "2026-08-01", "to": "2026-10-10"}, t["covers"])

    print("\n--- search_items ---")
    res = tspm_data.search_items(query="pricing")
    r.check("matches title and summary", res["total_matches"] == 2, res)
    r.check("a title hit outranks a passing mention in a summary",
            res["items"][0]["url"] == "https://a.com/1",
            [i["url"] for i in res["items"]])

    r.check("an unmatched query returns nothing, not everything",
            tspm_data.search_items(query="zzzz")["total_matches"] == 0)
    r.check("no query at all returns the whole archive",
            tspm_data.search_items()["total_matches"] == 3)

    print("\n--- filters combine with AND ---")
    r.check("tag filter",
            tspm_data.search_items(tag="Career")["total_matches"] == 2)
    r.check("tag matching ignores case",
            tspm_data.search_items(tag="career")["total_matches"] == 2)
    r.check("type filter",
            tspm_data.search_items(content_type="job")["total_matches"] == 1)
    r.check("since filter excludes older items",
            tspm_data.search_items(since="2026-09-01")["total_matches"] == 2)
    r.check("filters stack",
            tspm_data.search_items(tag="Career", content_type="job")
            ["total_matches"] == 1)

    print("\n--- results are shaped for a model, not dumped ---")
    long_summary = dict(ITEMS[0])
    long_summary["summary"] = "x" * 5000
    (_TMP / "data.json").write_text(json.dumps(
        {"items": [long_summary]}), encoding="utf-8")
    shaped = tspm_data.search_items()["items"][0]
    r.check("summaries are truncated",
            len(shaped["summary"]) == tspm_data.MAX_SUMMARY_CHARS,
            len(shaped["summary"]))
    r.check("the shape is fixed, not whatever the feed had",
            set(shaped) == {"title", "url", "source", "date", "type",
                            "tags", "summary"}, sorted(shaped))
    setup()

    r.check("limit is capped so one call cannot flood the context",
            tspm_data.search_items(limit=9999)["returned"] <= tspm_data.MAX_LIMIT)
    r.check("a limit of zero still returns something",
            tspm_data.search_items(limit=0)["returned"] >= 1)

    print("\n--- get_edition ---")
    r.check("no date gives the newest",
            tspm_data.get_edition()["edition"] == "2026-10-11")
    r.check("an exact date works",
            tspm_data.get_edition("2026-10-04")["headline_theme"]
            == "Discovery beats velocity")
    missing = tspm_data.get_edition("2001-01-01")
    r.check("a missing date explains itself instead of failing silently",
            "error" in missing and missing["available"], missing)

    print("\n--- search_editorials: the original material ---")
    hits = tspm_data.search_editorials("pricing")
    r.check("finds the edition that argued it", hits["total_matches"] == 1, hits)
    top = hits["editions"][0]
    r.check("carries the homework", top["pm_homework"] == ["Interview 5 buyers"])
    r.check("carries the contrarian pick",
            top["contrarian"]["url"] == "https://a.com/c", top["contrarian"])
    r.check("carries each must-read WITH the reason it was picked",
            top["must_reads"][0]["why"].startswith("The only piece"),
            top["must_reads"])
    r.check("an empty query is refused, not treated as match-all",
            "error" in tspm_data.search_editorials(""))

    print("\n--- a broken repo degrades instead of crashing ---")
    (_TMP / "data.json").write_text("{not json", encoding="utf-8")
    r.check("unparseable data.json returns no items, raises nothing",
            tspm_data.search_items()["total_matches"] == 0)
    r.check("...and list_tags still answers",
            tspm_data.list_tags()["total_items"] == 0)
    (_TMP / "data.json").unlink()
    r.check("a missing data.json is survivable too",
            tspm_data.search_items()["total_matches"] == 0)
    setup()

    (_TMP / "agent" / "index.json").write_text(json.dumps({"editions": [
        {"date": "../../../etc/passwd"}, {"date": "2026-10-11"},
    ]}), encoding="utf-8")
    eds = tspm_data.load_editions()
    r.check("an index entry that is not a plain date is ignored",
            [e["edition"] for e in eds] == ["2026-10-11"],
            [e.get("edition") for e in eds])

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
