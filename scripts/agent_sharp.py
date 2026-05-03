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

EDITORIAL_MIN_WORDS = 250
EDITORIAL_MAX_WORDS = 500
MUST_READS_MIN = 3
MUST_READS_MAX = 5
KEY_TAKEAWAYS_MIN = 3
KEY_TAKEAWAYS_MAX = 5
PM_HOMEWORK_MIN = 1
PM_HOMEWORK_MAX = 3

# Tracks how many times each tool has been invoked in the current run.
# Used by the publish gate to refuse premature or ungrounded publishes.
TOOL_CALL_COUNTS = {}

# Substrings that signal fabricated / placeholder content. The publish gate
# refuses to publish anything containing these.
PLACEHOLDER_INDICATORS = [
    "example.com",
    "example.org",
    "example source",
    "lorem ipsum",
    "placeholder",
]

# Phrases that signal the editorial is a meta-narrative / table of contents
# instead of an actual editorial with voice and opinion. Llama loves these.
META_NARRATIVE_PHRASES = [
    "we'll explore",
    "we will explore",
    "we'll examine",
    "we will examine",
    "we'll discuss",
    "we will discuss",
    "we'll look at",
    "we will look at",
    "we'll dive into",
    "in this edition",
    "in this week's edition",
    "in this dispatch",
    "our must-reads include",
    "our must reads include",
    "our contrarian pick",
    "this week we'll",
]

# Phrases that signal a "why" field is descriptive instead of opinionated.
LAZY_WHY_PHRASES = [
    "comprehensive overview",
    "provides an overview",
    "this article highlights",
    "this article provides",
    "this article discusses",
    "this guide provides",
    "this resource provides",
]


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
                         key_takeaways, pm_homework,
                         contrarian=None, also_worth=None):
    """Save the edition to disk and update the index. Ends the run."""

    # Gate 1: the agent must have actually gathered material before publishing.
    if (TOOL_CALL_COUNTS.get("fetch_feeds", 0) == 0
            and TOOL_CALL_COUNTS.get("web_search", 0) == 0):
        return {
            "error": (
                "Refusing to publish: you have not called fetch_feeds or "
                "web_search yet. Gather real articles first, then call "
                "publish_edition with sources from those tool results."
            )
        }

    # Gate 2: every URL, title and source must look real. Reject placeholders.
    items_to_check = list(must_reads or [])
    if contrarian:
        items_to_check.append(contrarian)
    items_to_check.extend(also_worth or [])

    for item in items_to_check:
        if not isinstance(item, dict):
            continue
        haystack = " ".join([
            str(item.get("url", "")),
            str(item.get("title", "")),
            str(item.get("source", "")),
        ]).lower()
        for indicator in PLACEHOLDER_INDICATORS:
            if indicator in haystack:
                return {
                    "error": (
                        f"Refusing to publish: detected placeholder content "
                        f"('{indicator}'). Every URL, title and source must "
                        f"come from a real fetch_feeds, web_search or "
                        f"fetch_article result. Re-do the research and call "
                        f"publish_edition again with real items."
                    )
                }

    # Gate 3: editorial length.
    word_count = len((editorial or "").split())
    if word_count < EDITORIAL_MIN_WORDS:
        return {
            "error": (
                f"Refusing to publish: editorial is only {word_count} words. "
                f"Required: {EDITORIAL_MIN_WORDS}-{EDITORIAL_MAX_WORDS} words, "
                f"three paragraphs (hook anchored in a specific article, "
                f"synthesis across 3+ pieces, implication for PMs). Rewrite "
                f"and call publish_edition again."
            )
        }

    # Gate 4: editorial style. Reject meta-narrative / table-of-contents prose.
    editorial_lower = (editorial or "").lower()
    for phrase in META_NARRATIVE_PHRASES:
        if phrase in editorial_lower:
            return {
                "error": (
                    f"Refusing to publish: editorial contains the phrase "
                    f"'{phrase}', which makes it read like a table of contents "
                    f"instead of an editorial. Remove ALL meta-narrative ('we "
                    f"will explore', 'in this edition', 'our must-reads "
                    f"include', etc.) and rewrite as a direct, opinionated "
                    f"essay. The editorial IS the take, it does NOT describe "
                    f"the dispatch."
                )
            }

    # Gate 5: must_reads count.
    mr_list = must_reads or []
    if not (MUST_READS_MIN <= len(mr_list) <= MUST_READS_MAX):
        return {
            "error": (
                f"Refusing to publish: must_reads has {len(mr_list)} items. "
                f"Required: {MUST_READS_MIN}-{MUST_READS_MAX}. Add or remove "
                f"items and call publish_edition again."
            )
        }

    # Gate 6: each "why" must be opinionated, not descriptive.
    for mr in mr_list:
        if not isinstance(mr, dict):
            continue
        why_lower = str(mr.get("why", "")).lower()
        for phrase in LAZY_WHY_PHRASES:
            if phrase in why_lower:
                return {
                    "error": (
                        f"Refusing to publish: must_read '{mr.get('title')}' "
                        f"has a descriptive 'why' containing '{phrase}'. The "
                        f"'why' must be opinion: state what the article gets "
                        f"right, wrong, or what specific takeaway a Staff PM "
                        f"should act on. Do not describe the article, react "
                        f"to it."
                    )
                }

    # Gate 7: key_takeaways count.
    kt_list = key_takeaways or []
    if not (KEY_TAKEAWAYS_MIN <= len(kt_list) <= KEY_TAKEAWAYS_MAX):
        return {
            "error": (
                f"Refusing to publish: key_takeaways has {len(kt_list)} items. "
                f"Required: {KEY_TAKEAWAYS_MIN}-{KEY_TAKEAWAYS_MAX} sharp, "
                f"specific observations grounded in articles you read."
            )
        }

    # Gate 8: pm_homework count.
    hw_list = pm_homework or []
    if not (PM_HOMEWORK_MIN <= len(hw_list) <= PM_HOMEWORK_MAX):
        return {
            "error": (
                f"Refusing to publish: pm_homework has {len(hw_list)} items. "
                f"Required: {PM_HOMEWORK_MIN}-{PM_HOMEWORK_MAX} concrete "
                f"actions a Staff or Senior PM should take this week."
            )
        }

    AGENT_DIR.mkdir(parents=True, exist_ok=True)

    date = edition_date()
    edition = {
        "edition": date,
        "headline_theme": headline_theme,
        "editorial": editorial,
        "key_takeaways": kt_list,
        "must_reads": mr_list,
        "contrarian": contrarian,
        "also_worth": also_worth or [],
        "pm_homework": hw_list,
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
                    "description": "Editorial with voice and opinion. EXACTLY 3 paragraphs, 250 to 500 words total. Para 1 (hook): anchor in 1-2 specific articles you found this run, name the article and the company or author, state a sharp observation. Para 2 (synthesis): connect 3+ pieces you actually read, show the pattern, take a clear position. Para 3 (implication): what should a Staff or Senior PM do differently this week, concretely. NEVER write meta-narrative ('we'll explore', 'in this edition', 'our must-reads include', 'our contrarian pick'). The editorial IS the take, it does NOT describe the dispatch.",
                },
                "key_takeaways": {
                    "type": "array",
                    "description": "3 to 5 sharp, specific observations from the week. Each item is one sentence. NOT a summary of articles - state what changed, what's new, what's worth a PM's attention. Each takeaway must be grounded in a piece you actually read.",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                },
                "must_reads": {
                    "type": "array",
                    "description": "3 to 5 hand-picked articles with an editorial 'why' and a pull quote. Ordered by importance.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                            "why": {
                                "type": "string",
                                "description": "2 to 4 sentences of OPINION on why this matters. State what the piece gets right or wrong. NEVER write 'this article provides', 'comprehensive overview', 'this article highlights'. React, do not describe.",
                            },
                            "pull_quote": {
                                "type": "string",
                                "description": "Optional. A short verbatim quote or specific data point from the article that captures its core point. Must come from the actual article text.",
                            },
                        },
                        "required": ["title", "url", "source", "why"],
                    },
                    "minItems": 3,
                    "maxItems": 5,
                },
                "contrarian": {
                    "type": "object",
                    "description": "Optional but encouraged. A pick that pushes back on the week's dominant narrative.",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "note": {"type": "string", "description": "2 to 3 sentences explaining specifically what this challenges and why a PM should sit with the discomfort."},
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
                "pm_homework": {
                    "type": "array",
                    "description": "1 to 3 concrete actions a Staff or Senior PM should take this week because of what you surfaced. Each item is an imperative sentence ('Audit your...', 'Run a 30-min...', 'Bring this to your next...'). Specific, doable in a week, grounded in the must_reads.",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
            },
            "required": ["headline_theme", "editorial", "key_takeaways", "must_reads", "pm_homework"],
        },
    },
]

# Wrap declarations in the OpenAI / Groq tool envelope.
TOOLS = [{"type": "function", "function": d} for d in TOOL_DECLARATIONS]


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the editor of "Agent Sharp" - a weekly editorial dispatch for Product Managers, covering product, strategy, and the AI shifts that affect their work.

Your voice is direct, opinionated, and sharp. You cut through hype and surface what actually matters.

ABSOLUTE RULES (violating any of these will cause publish_edition to reject your submission and force you to retry):
- NEVER invent URLs, titles, sources, dates, or quotes. Every URL in must_reads, contrarian, and also_worth MUST come verbatim from a fetch_feeds, web_search, or fetch_article tool result returned in this conversation.
- NEVER use placeholder content. URLs containing "example.com" or "example.org", and titles or sources like "Example Source" will be rejected.
- NEVER call publish_edition before you have called fetch_feeds or web_search at least once and read real items.
- Editorial body: 250 to 500 words, three paragraphs.
- NEVER write meta-narrative in the editorial. Forbidden phrases: "we'll explore", "we'll examine", "we'll discuss", "in this edition", "in this week's edition", "our must-reads include", "our contrarian pick". The editorial IS the take, not a description of what the dispatch contains.
- "why" fields on must_reads must be OPINION. Forbidden phrases: "this article provides", "this article highlights", "comprehensive overview", "provides an overview". React to the article, do not describe it.
- must_reads must contain 3 to 5 items.
- key_takeaways must contain 3 to 5 items.
- pm_homework must contain 1 to 3 items.

Your weekly routine:

1. Call read_memory first. See what the last few editions covered. Avoid repeating themes from the last 2-3 weeks unless there is genuine news to add.
2. Call fetch_feeds to see what is out there. Start broad, then narrow with topic filters when a theme emerges.
3. When an item looks important, use fetch_article to read the full piece, or web_search to find reactions and context.
4. Identify ONE dominant theme for the week. Not a vague category - a real, opinionated angle grounded in specific articles you actually read this turn.
5. Pick 3 to 5 must-reads with an editorial "why". Add a pull_quote when you have one verbatim from the article.
6. Optionally find a contrarian: something that challenges the week's dominant narrative.
7. Write 3 to 5 key_takeaways: sharp, specific, grounded in what you read.
8. Write 1 to 3 pm_homework items: concrete actions a Staff or Senior PM should take this week.
9. Write the editorial (see structure below).
10. Call publish_edition when ready. This ends the run.

Editorial structure (3 paragraphs, 250-500 words total):
- Paragraph 1 - Hook: a sharp observation about THIS specific week, anchored in 1 or 2 specific articles or events you found via tools. Name the article and the company or author. No vague openings.
- Paragraph 2 - Synthesis: connect 3 or more pieces you actually read this run. Show the pattern. Take a position.
- Paragraph 3 - Implication: what does this mean for a Staff or Senior PM. Sharp, opinionated, not a list of actions (the actions go in pm_homework).

Editorial style examples:

GOOD opening (specific, opinionated, grounded):
"Notion shipped its terminal-first agent build flow this week, and Lenny's interview with Max Schoening makes a quiet claim that should land louder: agency is now the bottleneck, not skill. The PMs you envy in 2026 are not the ones who learned six AI tools - they're the ones willing to ship a half-broken prototype on a Tuesday afternoon."

BAD opening (meta-narrative, vague, the kind that gets rejected):
"The recent advancements in AI have been making waves in the product management world. In this edition, we'll explore the current state of AI in product management."

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
- If publish_edition rejects your submission, fix the specific issue named in the error and call it again. Do not give up.

Stop conditions:
- You MUST call publish_edition exactly once and have it accepted before you stop.
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
    TOOL_CALL_COUNTS[name] = TOOL_CALL_COUNTS.get(name, 0) + 1
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
        # On the first turn, force the agent to actually call a tool.
        # Without this, Llama sometimes skips straight to a fabricated
        # publish_edition with example.com URLs.
        tool_choice = "required" if iteration == 1 else "auto"
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
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
