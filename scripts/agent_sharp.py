"""Agent Sharp - weekly editorial dispatch for The Sharp PM.

This agent reads PM/AI feeds, searches the web for context, reviews its own
past editions from memory, and publishes a weekly editorial digest with
opinion and voice. Output lives in agent/<date>.json.

Differs from scripts/aggregate.py: the aggregator is a mechanical pipeline
that summarises every item it sees. This is an editor - it decides what
matters, surfaces themes, writes with voice, and publishes one curated
dispatch per week.
"""

import feedparser
import json
import os
import re
import requests
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import mktime

from groq import Groq


# ── CONFIG ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
MODEL = os.environ.get("AGENT_MODEL", "llama-3.3-70b-versatile")

AGENT_DIR = Path("agent")
INDEX_FILE = AGENT_DIR / "index.json"

MAX_ITERATIONS = 15          # safety cap on agent turns
MAX_ARTICLE_CHARS = 5000     # truncate article fetches
MAX_FEED_ITEMS = 30          # cap feed payload per tool call


# ── FEEDS ─────────────────────────────────────────────────────────────────────

FEEDS = [
    {"url": "https://www.lennysnewsletter.com/feed", "source": "Lenny's Newsletter"},
    {"url": "https://www.reforge.com/blog/rss.xml", "source": "Reforge"},
    {"url": "https://www.svpg.com/articles/rss", "source": "SVPG"},
    {"url": "https://www.mindtheproduct.com/feed/", "source": "Mind the Product"},
    {"url": "https://blackboxofpm.com/feed", "source": "Black Box of PM"},
    {"url": "https://www.producttalk.org/feed/", "source": "Product Talk"},
    {"url": "https://www.ben-evans.com/benedictevans/rss.xml", "source": "Benedict Evans"},
    {"url": "https://stratechery.com/feed/", "source": "Stratechery"},
    {"url": "https://www.exponentialview.co/feed", "source": "Exponential View"},
    {"url": "https://www.firstround.com/review/feed.xml", "source": "First Round Review"},
    {"url": "https://hnrss.org/best?q=product+manager", "source": "Hacker News"},
]


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def edition_date():
    """Edition is always dated today (UTC). Sunday cron -> Sunday edition."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def strip_html(raw):
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────

def tool_fetch_feeds(topics=None, max_per_feed=2):
    """Return recent entries from curated feeds, optionally filtered by topic keywords."""
    max_per_feed = max(1, min(int(max_per_feed or 2), 5))
    topics_lower = [t.lower() for t in (topics or [])]
    items = []

    for feed in FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:max_per_feed]:
                title = entry.get("title", "")
                summary = strip_html(entry.get("summary", ""))[:500]

                if topics_lower:
                    haystack = (title + " " + summary).lower()
                    if not any(t in haystack for t in topics_lower):
                        continue

                date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    date = datetime.fromtimestamp(
                        mktime(entry.published_parsed), tz=timezone.utc
                    ).isoformat()

                items.append({
                    "title": title,
                    "url": entry.get("link", ""),
                    "source": feed["source"],
                    "summary": summary,
                    "date": date,
                })
        except Exception as e:
            print(f"  feed error ({feed['source']}): {e}")

    items = items[:MAX_FEED_ITEMS]
    return {"count": len(items), "items": items}


def tool_web_search(query, max_results=5):
    """Search the web via Tavily. Returns snippets optimised for agent consumption."""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not configured"}
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max(1, min(int(max_results or 5), 10)),
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20,
        )
        data = response.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:1000],
                "score": r.get("score", 0),
            })
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": str(e)}


def tool_fetch_article(url):
    """Fetch the text of a specific URL. Truncated to MAX_ARTICLE_CHARS."""
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Agent Sharp Weekly Editor)"},
        )
        text = strip_html(response.text)
        return {
            "url": url,
            "text": text[:MAX_ARTICLE_CHARS],
            "truncated": len(text) > MAX_ARTICLE_CHARS,
        }
    except Exception as e:
        return {"error": str(e)}


def tool_read_memory(weeks=4):
    """Return summaries of the most recent editions for de-duplication and continuity."""
    weeks = max(1, min(int(weeks or 4), 12))
    if not INDEX_FILE.exists():
        return {"editions": [], "note": "No past editions yet (first run)."}

    try:
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"editions": [], "note": "Could not read index."}

    editions_meta = index.get("editions", [])[:weeks]
    summaries = []
    for meta in editions_meta:
        edition_file = AGENT_DIR / f"{meta['date']}.json"
        if not edition_file.exists():
            continue
        try:
            ed = json.loads(edition_file.read_text(encoding="utf-8"))
            summaries.append({
                "date": ed.get("edition"),
                "headline_theme": ed.get("headline_theme"),
                "editorial_excerpt": (ed.get("editorial") or "")[:400],
                "must_reads": [
                    {"title": mr.get("title"), "why": mr.get("why")}
                    for mr in (ed.get("must_reads") or [])
                ],
            })
        except Exception:
            continue
    return {"count": len(summaries), "editions": summaries}


def tool_publish_edition(headline_theme, editorial, must_reads,
                         contrarian=None, also_worth=None):
    """Save the edition to disk and update the index. Ends the run."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)

    date = edition_date()
    edition = {
        "edition": date,
        "headline_theme": headline_theme,
        "editorial": editorial,
        "must_reads": must_reads or [],
        "contrarian": contrarian,
        "also_worth": also_worth or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": MODEL,
    }

    edition_file = AGENT_DIR / f"{date}.json"
    edition_file.write_text(
        json.dumps(edition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if INDEX_FILE.exists():
        try:
            index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            index = {"editions": []}
    else:
        index = {"editions": []}

    index["editions"] = [e for e in index.get("editions", []) if e.get("date") != date]
    index["editions"].insert(0, {
        "date": date,
        "headline_theme": headline_theme,
        "file": f"agent/{date}.json",
    })
    index["updated_at"] = datetime.now(timezone.utc).isoformat()

    INDEX_FILE.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"status": "published", "file": f"agent/{date}.json"}


# ── TOOL DECLARATIONS (OpenAI / Groq function calling) ────────────────────────

TOOL_DECLARATIONS = [
    {
        "name": "read_memory",
        "description": "Read summaries of past Agent Sharp editions. Call this at the start of every run to avoid repeating themes and to spot continuities.",
        "parameters": {
            "type": "object",
            "properties": {
                "weeks": {
                    "type": "integer",
                    "description": "How many past editions to retrieve. Default 4.",
                },
            },
        },
    },
    {
        "name": "fetch_feeds",
        "description": "Fetch recent entries from curated PM and AI RSS feeds. Returns titles, URLs, summaries, dates. Optionally filter entries by topic keywords (case-insensitive substring match on title + summary).",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to filter entries. Omit to get a broad sample.",
                },
                "max_per_feed": {
                    "type": "integer",
                    "description": "Max entries per feed source. Default 2, keep low (1-3).",
                },
            },
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for context, reactions, or to verify a claim. Returns titles, URLs and content snippets. Use sparingly - 5 to 15 searches per run is normal.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "max_results": {"type": "integer", "description": "Max results. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_article",
        "description": "Fetch the full text of a URL when a feed summary is not enough. Text truncated to about 5000 chars.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The article URL."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "publish_edition",
        "description": "Publish the final weekly edition. Call this ONCE at the end, when you have enough material and opinion. After this call, the run ends.",
        "parameters": {
            "type": "object",
            "properties": {
                "headline_theme": {
                    "type": "string",
                    "description": "Short, provocative headline for the week. 5 to 12 words.",
                },
                "editorial": {
                    "type": "string",
                    "description": "Opening editorial with voice and opinion. 2 or 3 paragraphs, 150 to 300 words. Direct, sharp, no corporate jargon.",
                },
                "must_reads": {
                    "type": "array",
                    "description": "3 to 5 hand-picked articles with an editorial 'why'. Ordered by importance.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                            "why": {
                                "type": "string",
                                "description": "1 or 2 sentences explaining why this matters and for whom. Opinion, not description.",
                            },
                        },
                        "required": ["title", "url", "source", "why"],
                    },
                },
                "contrarian": {
                    "type": "object",
                    "description": "Optional. A pick that pushes back on the week's dominant narrative.",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "note": {"type": "string", "description": "Why this challenges the dominant view."},
                    },
                },
                "also_worth": {
                    "type": "array",
                    "description": "Optional. 3 to 8 secondary picks with lighter treatment.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                        },
                        "required": ["title", "url"],
                    },
                },
            },
            "required": ["headline_theme", "editorial", "must_reads"],
        },
    },
]

# Wrap declarations in the OpenAI / Groq tool envelope.
TOOLS = [{"type": "function", "function": d} for d in TOOL_DECLARATIONS]


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the editor of "Agent Sharp" - a weekly editorial dispatch for Product Managers interested in AI.

Your voice is direct, opinionated, and sharp. You cut through hype and surface what actually matters.

Your weekly routine:

1. Call read_memory first. See what the last few editions covered. Avoid repeating themes from the last 2-3 weeks unless there is genuine news to add.
2. Call fetch_feeds to see what is out there. Start broad, then narrow with topic filters when a theme emerges.
3. When an item looks important, use fetch_article to read the full piece, or web_search to find reactions and context.
4. Identify ONE dominant theme for the week. Not a vague category - a real, opinionated angle.
5. Pick 3 to 5 must-reads with an editorial "why". The "why" must contain opinion, not description.
6. Optionally find a contrarian: something that challenges the week's dominant narrative.
7. Write the editorial: 2 to 3 short paragraphs, sharp voice, no hedging.
8. Call publish_edition when ready. This ends the run.

Editorial principles:
- Opinion beats description. "This matters because X" beats "This article says Y".
- Depth beats volume. 4 great picks beats 15 mediocre ones.
- Push back. If a piece is weak or hype, say so.
- No corporate jargon. Avoid "unlock", "leverage", "dive into", "game-changer", "deep dive".
- Short sentences. Real voice. Use em dashes sparingly - prefer commas or periods.
- Assume readers are PMs with 5 to 15 years of experience. Do not explain basics.
- Write for sharp minds who want a point of view, not a roundup.

Budget:
- Use at most 14 tool calls total before publish_edition.
- Typical spread: 1x read_memory, 2 to 3x fetch_feeds, 3 to 6x web_search, 2 to 4x fetch_article.
- If you hit 12 tool calls without having picked must-reads, stop exploring and start writing.

Stop conditions:
- You MUST call publish_edition exactly once before you stop.
- Do not produce plain-text output outside of tool calls.
"""


# ── AGENT LOOP ────────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "read_memory": tool_read_memory,
    "fetch_feeds": tool_fetch_feeds,
    "web_search": tool_web_search,
    "fetch_article": tool_fetch_article,
    "publish_edition": tool_publish_edition,
}


def execute_tool(name, args):
    print(f"  -> tool: {name}({json.dumps(args, ensure_ascii=False)[:200]})")
    fn = TOOL_DISPATCH.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"tool {name} failed: {e}"}


def run_agent():
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set")
        return 1
    if not TAVILY_API_KEY:
        print("WARNING: TAVILY_API_KEY not set - web_search will return errors.")

    client = Groq(api_key=GROQ_API_KEY)

    date = edition_date()
    user_turn = (
        f"Begin editorial work for the Agent Sharp edition of {date}. "
        f"Start with read_memory, then explore feeds, finish by calling publish_edition."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]

    print(f"Agent Sharp - starting run for {date}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[iter {iteration}]")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
            )
        except Exception as e:
            print(f"  Groq error: {e}")
            return 1

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_msg = {"role": "assistant", "content": message.content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            if message.content:
                print(f"  Agent said: {message.content[:300]}")
            print("  Agent produced no tool calls. Stopping.")
            return 1

        published = False
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_tool(tc.function.name, args)
            except json.JSONDecodeError as e:
                result = {"error": f"could not parse arguments: {e}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

            if (tc.function.name == "publish_edition"
                    and isinstance(result, dict)
                    and result.get("status") == "published"):
                published = True
                print(f"  [published] {result.get('file')}")

        if published:
            print("\nDone.")
            return 0

    print(f"\nERROR: agent hit max iterations ({MAX_ITERATIONS}) without publishing.")
    return 1


if __name__ == "__main__":
    sys.exit(run_agent())
