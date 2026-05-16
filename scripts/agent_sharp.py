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
import ipaddress
import json
import os
import re
import requests
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import mktime
from urllib.parse import urlparse

from groq import Groq


# ── CONFIG ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
# Default to qwen/qwen3-32b. History of what was tried and discarded:
#   - llama-3.3-70b-versatile: emits malformed <function=name{args}>
#     tags (missing `>`) on iteration 1, confirmed even after 3 retries
#     with perturbed temperatures - identical bad output every time.
#   - openai/gpt-oss-120b: 8k TPM on free tier, this workload hits ~9k+
#     per request, every run rate-limited.
#   - moonshotai/kimi-k2-instruct: not listed in Groq's available
#     models, returns 404.
# qwen3-32b is on Groq's Preview tier, has solid tool-calling
# reputation in agentic flows, and 32B params should fit within the
# free-tier TPM budget. Override with AGENT_MODEL env var.
MODEL = os.environ.get("AGENT_MODEL", "qwen/qwen3-32b")

AGENT_DIR = Path("agent")
INDEX_FILE = AGENT_DIR / "index.json"

MAX_ITERATIONS = 15          # safety cap on agent turns
MAX_ARTICLE_CHARS = 2500     # truncate article fetches
MAX_FEED_ITEMS = 10          # cap feed payload per tool call
MAX_FEED_SUMMARY_CHARS = 200 # cap each feed item's summary

# How long to sleep on a transient rate_limit_exceeded (TPM) error
# before retrying. TPM windows are 60 seconds, so 65s is enough for
# the oldest tokens to roll out. Tries up to RATE_LIMIT_MAX_RETRIES.
RATE_LIMIT_SLEEP_SECONDS = 65
RATE_LIMIT_MAX_RETRIES = 2

EDITORIAL_MIN_WORDS = 250
EDITORIAL_MAX_WORDS = 500
MUST_READS_MIN = 3
MUST_READS_MAX = 5
KEY_TAKEAWAYS_MIN = 3
KEY_TAKEAWAYS_MAX = 5
PM_HOMEWORK_MIN = 1
PM_HOMEWORK_MAX = 3

# Minimum total of fetch_feeds + web_search + fetch_article calls before
# publish_edition will be accepted. Stops the agent from publishing on
# iteration 3-5 with a single article.
MIN_RESEARCH_CALLS = 6

# Per-field hard caps on what publish_edition will accept. Defence
# against an LLM prompt-injected into emitting absurdly large strings
# that bloat the JSON file and inflate every later read_memory call.
MAX_HEADLINE_CHARS = 200
MAX_MUST_READ_TITLE_CHARS = 300
MAX_MUST_READ_SOURCE_CHARS = 100
MAX_MUST_READ_WHY_CHARS = 1500
MAX_MUST_READ_PULL_QUOTE_CHARS = 500
MAX_KEY_TAKEAWAY_CHARS = 500
MAX_PM_HOMEWORK_CHARS = 500
MAX_CONTRARIAN_NOTE_CHARS = 1000
MAX_ALSO_WORTH_TITLE_CHARS = 300

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
                summary = strip_html(entry.get("summary", ""))[:MAX_FEED_SUMMARY_CHARS]

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
    # Hard cap at 3 results and 400 chars/snippet. With the TPM cap on
    # qwen3-32b free tier (6k), one fat web_search response was enough
    # to push the accumulated context over the line.
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max(1, min(int(max_results or 3), 3)),
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
                "content": (r.get("content") or "")[:400],
                "score": r.get("score", 0),
            })
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": str(e)}


# Hard cap on bytes downloaded by tool_fetch_article. The text is
# truncated anyway, but without this a malicious server could stream
# arbitrary amounts of data before we hit our slice. 1 MB is enough
# headroom for any legitimate article we'd want to extract from.
MAX_FETCH_BYTES = 1_048_576


def _url_is_safe_to_fetch(url):
    """Return (ok, reason). Blocks anything that's not public http/https.

    Defends against SSRF via prompt injection: a feed snippet could
    instruct the LLM to call fetch_article(http://169.254.169.254/...)
    to hit the GHA runner's cloud-metadata endpoint, an internal
    service, or a loopback address. We allow only http(s) and only to
    DNS names that resolve to public IPs.
    """
    if not isinstance(url, str) or not url.strip():
        return False, "empty URL"
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed (http/https only)"
    if not parsed.hostname:
        return False, "URL has no hostname"
    # Resolve every address the hostname maps to. Reject if any is
    # private, loopback, link-local, reserved or multicast. A single
    # public A record alongside a private one is treated as unsafe -
    # DNS rebinding pretext.
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        return False, f"DNS resolution failed: {e}"
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False, f"unresolvable IP: {addr}"
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False, f"hostname resolves to non-public IP {addr}"
    return True, "ok"


MAX_REDIRECTS = 5


def tool_fetch_article(url):
    """Fetch the text of a specific URL. Truncated to MAX_ARTICLE_CHARS."""
    ok, reason = _url_is_safe_to_fetch(url)
    if not ok:
        return {"error": f"refusing to fetch: {reason}"}
    try:
        # Manual redirect handling. Each hop is re-validated by
        # _url_is_safe_to_fetch so a public host cannot 302-redirect
        # us to a private/loopback/metadata target. stream=True +
        # iter_content caps the download at MAX_FETCH_BYTES so a
        # malicious server cannot stream gigabytes before our text
        # slice kicks in.
        current_url = url
        response = None
        for hop in range(MAX_REDIRECTS + 1):
            response = requests.get(
                current_url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (Agent Sharp Weekly Editor)"},
                stream=True,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                next_url = response.headers.get("Location")
                response.close()
                if not next_url:
                    return {"error": "redirect with no Location header"}
                if hop == MAX_REDIRECTS:
                    return {"error": f"too many redirects (> {MAX_REDIRECTS})"}
                # Resolve relative Location against the current URL,
                # then re-validate before the next hop.
                next_url = requests.compat.urljoin(current_url, next_url)
                ok, reason = _url_is_safe_to_fetch(next_url)
                if not ok:
                    return {"error": f"refusing redirect to unsafe target: {reason}"}
                current_url = next_url
                continue
            break
        chunks = []
        bytes_read = 0
        for chunk in response.iter_content(chunk_size=16384, decode_unicode=False):
            if not chunk:
                continue
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read >= MAX_FETCH_BYTES:
                break
        response.close()
        raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        text = strip_html(raw)
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
    # Strict ISO-date regex. index.json is in-repo so a malicious entry
    # would require write access, but a single bogus value like
    # "../../../etc/passwd" would otherwise let read_memory open files
    # outside AGENT_DIR. Cheap belt-and-braces.
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for meta in editions_meta:
        date = str(meta.get("date", ""))
        if not date_re.match(date):
            continue
        edition_file = AGENT_DIR / f"{date}.json"
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

    # Gate 1b: minimum exploration before publishing. Llama 3.3 likes to
    # publish on iteration 3-5 with one article. Force a wider lens.
    research_calls = (
        TOOL_CALL_COUNTS.get("fetch_feeds", 0)
        + TOOL_CALL_COUNTS.get("web_search", 0)
        + TOOL_CALL_COUNTS.get("fetch_article", 0)
    )
    if research_calls < MIN_RESEARCH_CALLS:
        return {
            "error": (
                f"Refusing to publish: only {research_calls} research calls "
                f"made (fetch_feeds + web_search + fetch_article). "
                f"Minimum {MIN_RESEARCH_CALLS}. You cannot pick the week's "
                f"theme from one article. Call fetch_feeds with different "
                f"topic filters, run web_search to find reactions and "
                f"context, and fetch_article on at least 2-3 full pieces "
                f"before deciding what matters."
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

    # Gate 2b: every URL must be http(s). Defense against an LLM
    # hallucinating a javascript:, data:, or file: URL that the
    # frontend then injects into an href attribute.
    for item in items_to_check:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        if not (url.lower().startswith("http://")
                or url.lower().startswith("https://")):
            return {
                "error": (
                    f"Refusing to publish: URL '{url[:80]}' is not http "
                    f"or https. Every URL in must_reads, contrarian and "
                    f"also_worth must be a real article link starting "
                    f"with http:// or https:// from a tool result."
                )
            }

    # Gate 3: editorial length (both ends). Enforcing the upper bound
    # is also a defence against an LLM being prompt-injected into
    # publishing a 50k-word blob that bloats the repo and burns tokens
    # on every subsequent read_memory.
    word_count = len((editorial or "").split())
    if word_count < EDITORIAL_MIN_WORDS or word_count > EDITORIAL_MAX_WORDS:
        return {
            "error": (
                f"Refusing to publish: editorial is {word_count} words. "
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

    # Gate 9: per-field length caps. Truncate-as-you-go (rather than
    # reject) so a slightly verbose model still gets published, but
    # nothing can bloat the JSON file by orders of magnitude. Mutates
    # the dicts in place; the caller has already passed them in.
    def _cap(s, n):
        if not isinstance(s, str):
            return s
        return s if len(s) <= n else s[:n].rstrip() + "..."

    headline_theme = _cap(headline_theme, MAX_HEADLINE_CHARS)
    for mr in mr_list:
        if not isinstance(mr, dict):
            continue
        mr["title"] = _cap(mr.get("title"), MAX_MUST_READ_TITLE_CHARS)
        mr["source"] = _cap(mr.get("source"), MAX_MUST_READ_SOURCE_CHARS)
        mr["why"] = _cap(mr.get("why"), MAX_MUST_READ_WHY_CHARS)
        if mr.get("pull_quote"):
            mr["pull_quote"] = _cap(mr["pull_quote"], MAX_MUST_READ_PULL_QUOTE_CHARS)
    kt_list = [_cap(k, MAX_KEY_TAKEAWAY_CHARS) for k in kt_list]
    hw_list = [_cap(h, MAX_PM_HOMEWORK_CHARS) for h in hw_list]
    if isinstance(contrarian, dict):
        contrarian["title"] = _cap(contrarian.get("title"), MAX_MUST_READ_TITLE_CHARS)
        contrarian["source"] = _cap(contrarian.get("source"), MAX_MUST_READ_SOURCE_CHARS)
        contrarian["note"] = _cap(contrarian.get("note"), MAX_CONTRARIAN_NOTE_CHARS)
    if also_worth:
        for aw in also_worth:
            if not isinstance(aw, dict):
                continue
            aw["title"] = _cap(aw.get("title"), MAX_ALSO_WORTH_TITLE_CHARS)
            if aw.get("source"):
                aw["source"] = _cap(aw["source"], MAX_MUST_READ_SOURCE_CHARS)

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

    # Atomic write: serialise to .tmp, then rename. Prevents a half-written
    # JSON from being committed if the process is killed mid-write.
    edition_file = AGENT_DIR / f"{date}.json"
    edition_tmp = edition_file.with_suffix(".json.tmp")
    edition_tmp.write_text(
        json.dumps(edition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(edition_tmp, edition_file)

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

    index_tmp = INDEX_FILE.with_suffix(".json.tmp")
    index_tmp.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(index_tmp, INDEX_FILE)

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
                    "description": "5-12 word provocative headline.",
                },
                "editorial": {
                    "type": "string",
                    "description": "250-500 words, 3 paragraphs (hook/synthesis/implication). See system prompt.",
                },
                "key_takeaways": {
                    "type": "array",
                    "description": "3-5 sharp one-sentence observations grounded in articles read.",
                    "items": {"type": "string"},
                },
                "must_reads": {
                    "type": "array",
                    "description": "3-5 hand-picked articles ordered by importance.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "source": {"type": "string"},
                            "why": {
                                "type": "string",
                                "description": "2-4 sentences of opinion (react, don't describe).",
                            },
                            "pull_quote": {
                                "type": "string",
                                "description": "Optional verbatim quote from article.",
                            },
                        },
                        "required": ["title", "url", "source", "why"],
                    },
                },
                "contrarian": {
                    "type": "object",
                    "description": "Optional pick that pushes back on the week's narrative.",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "note": {"type": "string", "description": "2-3 sentences on what this challenges."},
                    },
                },
                "also_worth": {
                    "type": "array",
                    "description": "Optional 3-8 secondary picks.",
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
                    "description": "1-3 concrete imperative actions for a Staff/Senior PM this week.",
                    "items": {"type": "string"},
                },
            },
            "required": ["headline_theme", "editorial", "key_takeaways", "must_reads", "pm_homework"],
        },
    },
]

# Wrap declarations in the OpenAI / Groq tool envelope.
TOOLS = [{"type": "function", "function": d} for d in TOOL_DECLARATIONS]


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the editor of Agent Sharp, a weekly editorial dispatch for Product Managers (5-15 yrs experience). Voice: direct, opinionated, sharp. No hype, no corporate jargon (no "unlock", "leverage", "dive into", "game-changer", "deep dive").

HARD RULES (publish_edition will reject and you must retry):
- Every URL/title/source comes verbatim from a fetch_feeds, web_search, or fetch_article result in this conversation. No invented sources, no placeholders.
- Call publish_edition only after >=6 research tool calls (fetch_feeds + web_search + fetch_article combined).
- Editorial: 250-500 words, exactly 3 paragraphs. NEVER meta-narrative ("we'll explore", "in this edition", "our must-reads include"). The editorial IS the take, not a TOC.
- must_reads "why": OPINION. Never "this article provides", "comprehensive overview", "highlights". React, don't describe.
- must_reads: 3-5 items. key_takeaways: 3-5. pm_homework: 1-3.

ROUTINE:
1. read_memory (avoid themes from last 2-3 weeks unless real news).
2. fetch_feeds broadly, then with topic filters as a theme emerges.
3. fetch_article on items that look important; web_search for context.
4. Pick ONE opinionated theme grounded in articles you actually read.
5. Pick 3-5 must_reads, optional contrarian, write 3-5 key_takeaways, 1-3 pm_homework.
6. Write editorial: P1 hook (name a specific article + author/company, sharp observation), P2 synthesis (connect 3+ pieces, take position), P3 implication for Staff/Senior PMs.
7. Call publish_edition. Run ends.

Budget: <=14 tool calls before publish. If publish_edition is rejected, fix the specific issue named and call again.

Stop: publish_edition must be called exactly once and accepted.
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


# Sliding-window cap: keep system + user + this many of the most recent
# messages in full. Older tool message contents are truncated to a short
# stub to stay within the model's per-request token budget. The agent
# does not need to re-read old tool dumps - it builds up understanding
# turn by turn - so truncating older results is structurally safe.
KEEP_RECENT_MESSAGES = 4
OLD_TOOL_RESULT_PREVIEW_CHARS = 150


def trim_message_history(messages):
    """In-place: truncate content of old tool messages to save tokens.

    Leaves system prompt, user turn, and the last KEEP_RECENT_MESSAGES
    untouched. For older tool-role messages, replaces the JSON content
    with a short preview plus a marker so the model knows that detail
    is no longer available.
    """
    if len(messages) <= 2 + KEEP_RECENT_MESSAGES:
        return
    cutoff = len(messages) - KEEP_RECENT_MESSAGES
    for i in range(2, cutoff):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= OLD_TOOL_RESULT_PREVIEW_CHARS + 50:
            continue
        msg["content"] = (
            content[:OLD_TOOL_RESULT_PREVIEW_CHARS]
            + "...[older tool result truncated for token budget; "
            "re-fetch if you need this content again]"
        )


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

    # Perturbed temperatures used on tool_use_failed retries. Cycling
    # through different values nudges the model out of any deterministic
    # bad-output pattern (malformed function tags, repeated invalid
    # arguments).
    RETRY_TEMPERATURES = [1.0, 0.4, 1.1]
    schema_retries = 0
    rate_limit_retries = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[iter {iteration}]")
        # On the first turn, force the agent to actually call a tool.
        # Without this, the model sometimes skips straight to a
        # fabricated publish_edition with example.com URLs.
        tool_choice = "required" if iteration == 1 else "auto"
        # Use perturbed temperature on retries; baseline 0.7 otherwise.
        if schema_retries > 0 and schema_retries <= len(RETRY_TEMPERATURES):
            current_temp = RETRY_TEMPERATURES[schema_retries - 1]
        else:
            current_temp = 0.7
        # Truncate content of older tool results before sending. Stops
        # the messages array from growing past the per-request TPM cap
        # as iterations stack up.
        trim_message_history(messages)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                temperature=current_temp,
            )
        except Exception as e:
            error_str = str(e)
            # Groq does its own validation on tool calls and returns 400
            # / tool_use_failed before the model gets a chance to react.
            # Two flavours: JSON-Schema mismatch on arguments, and
            # malformed function-tag emission from the model. In both
            # cases, feed the error back as a user message and let the
            # agent retry with a perturbed temperature, instead of
            # crashing the whole run.
            if "tool_use_failed" in error_str and schema_retries < 3:
                schema_retries += 1
                print(
                    f"  Groq rejected tool call. Asking agent to retry "
                    f"(retry {schema_retries}/3, temp -> "
                    f"{RETRY_TEMPERATURES[schema_retries - 1]})."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous tool call was rejected by the API. "
                        "It was either malformed (function tag syntax) or "
                        "did not match the tool's JSON schema. Re-read "
                        "the tool definitions carefully (required fields, "
                        "expected counts, types) and try again, emitting "
                        "exactly one well-formed tool call. Error from "
                        f"the API: {error_str[:600]}"
                    ),
                })
                continue
            # Transient TPM rate limit: free tier windows are per-minute,
            # so a short sleep is usually enough for the oldest tokens
            # to roll out. Don't retry forever - if it keeps failing,
            # the payload is structurally too large and a code fix is
            # needed, not a wait.
            if ("rate_limit_exceeded" in error_str
                    and "tokens per minute" in error_str.lower()
                    and rate_limit_retries < RATE_LIMIT_MAX_RETRIES):
                rate_limit_retries += 1
                print(
                    f"  Hit TPM rate limit. Sleeping "
                    f"{RATE_LIMIT_SLEEP_SECONDS}s and retrying "
                    f"(retry {rate_limit_retries}/"
                    f"{RATE_LIMIT_MAX_RETRIES})."
                )
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                continue
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
