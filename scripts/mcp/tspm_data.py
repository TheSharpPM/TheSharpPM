"""Read-only queries over the Sharp PM archive.

This module is deliberately free of any MCP import. The protocol layer
lives in server.py and does nothing but wire these functions up, which
buys three things:

  - these functions are testable with no dependencies at all, so they
    run in the existing CI suites rather than needing a new job;
  - the same logic can be served over stdio now and HTTP later without
    being touched;
  - a bug is either in the data layer or in the wiring, never ambiguous.

Everything is read from the repo's own JSON files. At ~950KB total
(874 links, 26 editions) there is no reason for a database, and no
reason to cache: json.load on this is single-digit milliseconds, and
loading fresh means the aggregator's daily push is picked up without
restarting the server. Revisit if the corpus grows an order of
magnitude.
"""
import json
from pathlib import Path

# scripts/mcp/tspm_data.py -> scripts/mcp -> scripts -> repo root.
# Resolved from __file__, not cwd: an MCP client launches the server
# from wherever it happens to be, usually not the repo.
REPO_ROOT = Path(__file__).resolve().parents[2]

# How much of a summary to hand back. The caller is a language model
# with a context budget, so results are shaped, not dumped - the same
# reason the Groq agent caps feed summaries at 200 chars.
MAX_SUMMARY_CHARS = 300
MAX_EDITORIAL_CHARS = 600
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


def _read_json(relative_path, fallback):
    """Read a repo JSON file, or return `fallback` if it is unusable.

    A missing or half-written file must degrade to an empty result
    rather than kill the server: the aggregator rewrites data.json every
    morning, and a client that crashes when it catches a partial write
    is worse than one that briefly sees nothing.
    """
    path = REPO_ROOT / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def load_items():
    """Every link in the 90-day archive, newest first."""
    data = _read_json("data.json", {})
    items = data.get("items") or []
    return sorted(items, key=lambda i: i.get("date") or "", reverse=True)


def load_editions():
    """Every published edition, newest first, as full documents."""
    index = _read_json("agent/index.json", {})
    editions = []
    for meta in index.get("editions") or []:
        date = str(meta.get("date") or "")
        # The index names its own files; only ever read the expected
        # shape, so a bad entry cannot walk out of agent/.
        if not (len(date) == 10 and date[4] == "-" and date[7] == "-"):
            continue
        doc = _read_json(f"agent/{date}.json", None)
        if isinstance(doc, dict):
            editions.append(doc)
    return editions


def _terms(query):
    return [t for t in str(query or "").lower().split() if t]


def _score(item, terms):
    """Relevance of one item to the query terms.

    Title matches outweigh summary matches because a title is the
    author's own claim about what the piece is; a summary term can be a
    passing mention. Tags count too, since every item here has them.
    """
    if not terms:
        return 1
    title = str(item.get("title") or "").lower()
    summary = str(item.get("summary") or "").lower()
    tags = " ".join(item.get("tags") or []).lower()
    score = 0
    for term in terms:
        if term in title:
            score += 3
        if term in tags:
            score += 2
        if term in summary:
            score += 1
    return score


def _shape_item(item):
    summary = str(item.get("summary") or "")
    return {
        "title": item.get("title"),
        "url": item.get("url"),
        "source": item.get("source"),
        "date": (item.get("date") or "")[:10],
        "type": item.get("type"),
        "tags": item.get("tags") or [],
        "summary": summary[:MAX_SUMMARY_CHARS],
    }


def list_tags():
    """What the archive can be filtered by: tags, types, sources.

    The discovery call. Without it a model has to guess that the tag is
    "Product Strategy" and not "product-strategy", and a wrong guess
    returns nothing with no hint why.
    """
    items = load_items()
    tags, types, sources = {}, {}, {}
    for item in items:
        for tag in item.get("tags") or []:
            tags[tag] = tags.get(tag, 0) + 1
        for field, bucket in (("type", types), ("source", sources)):
            value = item.get(field)
            if value:
                bucket[value] = bucket.get(value, 0) + 1
    dates = sorted((i.get("date") or "")[:10] for i in items if i.get("date"))
    return {
        "total_items": len(items),
        "covers": {"from": dates[0], "to": dates[-1]} if dates else {},
        "tags": [{"tag": t, "count": c}
                 for t, c in sorted(tags.items(), key=lambda kv: -kv[1])],
        "types": [{"type": t, "count": c}
                  for t, c in sorted(types.items(), key=lambda kv: -kv[1])],
        "sources": [{"source": s, "count": c}
                    for s, c in sorted(sources.items(), key=lambda kv: -kv[1])],
    }


def search_items(query=None, tag=None, content_type=None, since=None,
                 limit=DEFAULT_LIMIT):
    """Search the link archive. Filters combine with AND."""
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    terms = _terms(query)
    results = []
    for item in load_items():
        if tag and tag.lower() not in [
            t.lower() for t in (item.get("tags") or [])
        ]:
            continue
        if content_type and str(item.get("type") or "").lower() != content_type.lower():
            continue
        # ISO dates compare correctly as strings, so no parsing needed.
        if since and (item.get("date") or "")[:10] < since[:10]:
            continue
        score = _score(item, terms)
        if score <= 0:
            continue
        results.append((score, item))
    results.sort(key=lambda pair: (-pair[0], pair[1].get("date") or ""))
    return {
        "total_matches": len(results),
        "returned": min(len(results), limit),
        "filters": {"query": query, "tag": tag,
                    "type": content_type, "since": since},
        "items": [_shape_item(i) for _, i in results[:limit]],
    }


def get_edition(date=None):
    """One published edition in full. Omit `date` for the latest."""
    editions = load_editions()
    if not editions:
        return {"error": "no editions found"}
    if date:
        wanted = str(date)[:10]
        for doc in editions:
            if str(doc.get("edition") or "")[:10] == wanted:
                return doc
        return {
            "error": f"no edition published on {wanted}",
            "available": [str(d.get("edition"))[:10] for d in editions[:8]],
        }
    return editions[0]


def search_editorials(query, limit=5):
    """Search what the editor ARGUED, across every past edition.

    The point of difference. The link archive is a commodity - anyone
    can subscribe to the same feeds - but the editorial take, the
    contrarian pick and the homework exist nowhere else.
    """
    limit = max(1, min(int(limit or 5), MAX_LIMIT))
    terms = _terms(query)
    if not terms:
        return {"error": "query is required"}
    hits = []
    for doc in load_editions():
        haystack = " ".join(str(doc.get(f) or "") for f in (
            "headline_theme", "editorial", "target_audience", "hook_source"
        )).lower()
        takeaways = " ".join(doc.get("key_takeaways") or []).lower()
        score = sum(
            3 * str(doc.get("headline_theme") or "").lower().count(t)
            + haystack.count(t)
            + takeaways.count(t)
            for t in terms
        )
        if score <= 0:
            continue
        contrarian = doc.get("contrarian") or {}
        hits.append((score, {
            "edition": doc.get("edition"),
            "headline_theme": doc.get("headline_theme"),
            "target_audience": doc.get("target_audience"),
            "editorial_excerpt": str(doc.get("editorial") or "")[:MAX_EDITORIAL_CHARS],
            "key_takeaways": doc.get("key_takeaways") or [],
            "pm_homework": doc.get("pm_homework") or [],
            "contrarian": {
                "title": contrarian.get("title"),
                "url": contrarian.get("url"),
                "note": contrarian.get("note"),
            } if contrarian else None,
            "must_reads": [
                {"title": mr.get("title"), "url": mr.get("url"),
                 "why": mr.get("why")}
                for mr in (doc.get("must_reads") or [])
            ],
        }))
    hits.sort(key=lambda pair: (-pair[0], pair[1]["edition"] or ""))
    return {
        "total_matches": len(hits),
        "returned": min(len(hits), limit),
        "editions": [h for _, h in hits[:limit]],
    }
