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
import random
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
# Default to openai/gpt-oss-120b. History of what was tried and
# discarded:
#   - qwen/qwen3-32b: reliable tool calling but 6k TPM ceiling is too
#     tight for this workload. Typical requests sit at 6.0-6.7k, so
#     every run is one bad iteration away from a 413 even with the
#     aggressive trim below.
#   - llama-3.3-70b-versatile: 12k TPM headroom but tool calling is
#     fundamentally broken on Groq for this model. Emits
#     `<function=name>{json}</function>` instead of OpenAI tool_calls
#     format, AND escapes single quotes as `\'` which is invalid JSON.
#     Perturbed-temperature retries do not fix - deterministic bad
#     output. Also hallucinates URLs (example.com placeholders).
#   - meta-llama/llama-4-scout-17b-16e-instruct: 30k TPM headroom but
#     tool calling also broken. Emits a generic
#     `[{"name":..., "parameters":...}]` list instead of OpenAI
#     tool_calls format, AND has the same `\'` invalid-JSON escape
#     habit as Llama 3. Entire Llama family unusable on Groq for this.
#   - moonshotai/kimi-k2-instruct: not listed in Groq's available
#     models, returns 404.
# gpt-oss-120b has OpenAI-native tool calling format (it is the OpenAI
# Open weights model). 8k TPM is enough now that the trim below caps
# typical requests around 6.7k. Override with AGENT_MODEL env var.
MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-120b")

AGENT_DIR = Path("agent")
INDEX_FILE = AGENT_DIR / "index.json"

MAX_ITERATIONS = 15          # safety cap on agent turns
MAX_ARTICLE_CHARS = 2500     # truncate article fetches
MAX_FEED_ITEMS = 15          # cap feed payload per tool call; needs to be >= len(FEEDS) so every source has a shot when shuffled
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

# Upper bound after which the agent loop starts actively pushing the
# model to publish. Without this, gpt-oss-120b will gladly burn the
# full MAX_ITERATIONS budget on research, especially when the dedup
# cache refuses re-fetches, and never call publish_edition. At this
# threshold we inject a user message demanding publish_edition; on the
# next iteration we also pin tool_choice to publish_edition.
FORCE_PUBLISH_AT_RESEARCH_CALLS = 8

# Allowed values for target_audience on publish_edition. Editorial P3
# is written specifically for the chosen seniority level. The agent
# must rotate: cannot repeat the same value as either of the last two
# editions (enforced by gate). Read order intentional - puts the more
# common Staff/Senior on the list with Lead/Principal so editorial can
# alternate without going off-canon.
TARGET_AUDIENCES = ["Senior PM", "Staff PM", "Lead/Principal PM"]
# How many of the most recent editions' target_audience values are
# blocked from re-use on the next edition. With 3 levels and a lookback
# of 2, the rotation is forced into the third unused level each week.
TARGET_AUDIENCE_LOOKBACK = 2

# How many of the most recent editions' hook_source values are blocked
# from re-use. hook_source is the publisher whose article opens P1 of
# the editorial. Without this rotation Lenny's Newsletter defaults to
# opening every dispatch because it sits at feed index 0 and publishes
# the most content per week. Lookback 2 forces P1 to rotate across at
# least 3 publishers.
HOOK_SOURCE_LOOKBACK = 2

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

# Per-run cache of URLs the agent has already attempted to fetch.
# Maps url -> 1-line note about the previous outcome. Used by
# execute_tool to short-circuit duplicate fetch_article calls so the
# agent cannot burn its iteration budget re-fetching the same URLs
# after the sliding-window trim forgets the earlier result.
FETCHED_URLS = {}

# Per-run log of every tool result that came back, kept verbatim and
# uncapped. Used by the publish gate to check that every percentage or
# multiplier the editorial cites can be found in at least one tool
# result - blocks the model from inventing stats like "40% reduction"
# or "300% headcount" that never appear in the sources. Lives separate
# from the message history so the sliding-window trim does not erase it.
TOOL_RESULTS_LOG = []

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
# Trimmed down to phrases that are UNAMBIGUOUSLY descriptive - "shows
# how" and "demonstrates" on their own can introduce a strong opinion
# ("shows how Apple's antitrust exposure changes product bets"), so
# blocking them causes the agent to loop trying to please the gate.
# Keep only patterns that are almost always Goodreads-summary language.
LAZY_WHY_PHRASES = [
    "comprehensive overview",
    "provides an overview",
    "this article highlights",
    "this article provides",
    "this article discusses",
    "this article shows",
    "this article explains",
    "this article offers",
    "this piece highlights",
    "this piece provides",
    "this piece shows",
    "this guide provides",
    "this resource provides",
    "showcases",
    "a great example",
    "valuable insight",
    "valuable insights",
    "i appreciate",
    "i find",
    "real-world playbook",
    "real world playbook",
]

# Domains accepted as must_read sources. Built from FEEDS plus the two
# major platforms where most PM/strategy writing lives (Medium and
# Substack) and a small set of high-signal publications. Anything
# outside this set is rejected at publish time, so the agent cannot
# elevate a vendor blog or SEO content farm to must_read just because
# web_search returned it.
TRUSTED_SOURCE_DOMAINS = {
    # In-feed sources
    "lennysnewsletter.com",
    "reforge.com",
    "svpg.com",
    "mindtheproduct.com",
    "blackboxofpm.com",
    "producttalk.org",
    "ben-evans.com",
    "stratechery.com",
    "exponentialview.co",
    "firstround.com",
    # Platforms where most PM writing lives
    "medium.com",
    "substack.com",
    # High-signal publications
    "hbr.org",
    "every.to",
    "platformer.news",
    "casey.news",
    "theverge.com",
    "wired.com",
    "techcrunch.com",
    "theinformation.com",
}

# Path patterns that mark a URL as vendor marketing rather than an
# editorial article. Reject must_reads matching these even when the
# domain happens to be in TRUSTED_SOURCE_DOMAINS.
VENDOR_URL_PATH_PATTERNS = [
    "/platform/",
    "/product/",
    "/pricing/",
    "/demo/",
    "/features/",
    "/integrations/",
    "/solutions/",
    "/signup/",
    "/sign-up/",
]


def _is_trusted_source(url):
    """Return True if the URL's hostname is in TRUSTED_SOURCE_DOMAINS,
    treating subdomains (e.g. user.substack.com, user.medium.com) as
    matching their parent domain."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for trusted in TRUSTED_SOURCE_DOMAINS:
        if host == trusted or host.endswith("." + trusted):
            return True
    return False


# Regex for percentages and Nx multipliers the editorial might claim.
# 4-digit years (1900-2099) are explicitly exempted: "in 2026" is a
# date reference, not a stat that needs grounding.
_NUMERIC_CLAIM_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?%|\d+(?:\.\d+)?x|\d+(?:\.\d+)?×)",
    re.IGNORECASE,
)


def _ungrounded_numeric_claims(editorial):
    """Return a list of numeric claims (percentages, Nx multipliers)
    that appear in the editorial but not verbatim in any tool result
    captured during this run. Empty list means everything is grounded.
    """
    if not isinstance(editorial, str) or not editorial:
        return []
    claims = _NUMERIC_CLAIM_RE.findall(editorial)
    if not claims:
        return []
    haystack = "\n".join(TOOL_RESULTS_LOG)
    ungrounded = []
    seen = set()
    for claim in claims:
        norm = claim.strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        # Match case-insensitively; the haystack is mixed-case JSON.
        if norm not in haystack.lower():
            ungrounded.append(claim)
    return ungrounded


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

    # Shuffle the feed order each call so no single publisher (Lenny's
    # sits at index 0) always ends up at the top of what the agent
    # sees. Without this, the P1 hook of the editorial defaults to
    # whoever is highest in FEEDS - hence the "opens with Lenny's
    # again" pattern we saw.
    shuffled_feeds = list(FEEDS)
    random.shuffle(shuffled_feeds)

    items = []

    for feed in shuffled_feeds:
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
                "target_audience": ed.get("target_audience"),
                "hook_source": ed.get("hook_source"),
                "editorial_excerpt": (ed.get("editorial") or "")[:400],
                "must_reads": [
                    {"title": mr.get("title"), "why": mr.get("why")}
                    for mr in (ed.get("must_reads") or [])
                ],
            })
        except Exception:
            continue
    return {"count": len(summaries), "editions": summaries}


def _recent_target_audiences():
    """Return the target_audience values of the most recent editions
    (up to TARGET_AUDIENCE_LOOKBACK). Used by the publish gate to block
    re-using the same audience two editions in a row. Older editions
    that pre-date the field return None and are skipped."""
    if not INDEX_FILE.exists():
        return []
    try:
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    recent = []
    for meta in index.get("editions", [])[:TARGET_AUDIENCE_LOOKBACK]:
        date = str(meta.get("date", ""))
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue
        f = AGENT_DIR / f"{date}.json"
        if not f.exists():
            continue
        try:
            ed = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        ta = ed.get("target_audience")
        if ta:
            recent.append(ta)
    return recent


def _recent_hook_sources():
    """Return the hook_source values (lowercased for case-insensitive
    comparison) of the most recent editions, up to HOOK_SOURCE_LOOKBACK.
    Editions that pre-date the field are silently skipped."""
    if not INDEX_FILE.exists():
        return []
    try:
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    recent = []
    for meta in index.get("editions", [])[:HOOK_SOURCE_LOOKBACK]:
        date = str(meta.get("date", ""))
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            continue
        f = AGENT_DIR / f"{date}.json"
        if not f.exists():
            continue
        try:
            ed = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        hs = ed.get("hook_source")
        if hs:
            recent.append(str(hs).strip().lower())
    return recent


def tool_publish_edition(headline_theme, editorial, must_reads,
                         key_takeaways, pm_homework,
                         contrarian=None, also_worth=None,
                         target_audience=None, hook_source=None):
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

    # Gate 3b: numeric claims in editorial must be grounded in tool
    # results. Catches the LLM fabricating stats like "40% reduction"
    # or "300% headcount increase" that never appear in any source.
    # 4-digit years are exempt via the regex.
    ungrounded = _ungrounded_numeric_claims(editorial)
    if ungrounded:
        return {
            "error": (
                f"Refusing to publish: editorial contains numeric claim(s) "
                f"{ungrounded} that do not appear in any tool result from "
                f"this run. Every percentage and multiplier in the editorial "
                f"must come verbatim from an article you actually fetched or "
                f"a search result you saw. Either (a) remove the unsupported "
                f"number, (b) rewrite the sentence without the stat, or "
                f"(c) fetch_article the source that has it. Do not invent "
                f"statistics. Call publish_edition again."
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

    # Gate 5b: must_read sources must be from the trusted whitelist, and
    # must not be vendor marketing pages even when the domain is trusted.
    # Stops vendor product pages and SEO content farms from being
    # elevated to must_reads.
    for mr in mr_list:
        if not isinstance(mr, dict):
            continue
        url = str(mr.get("url", "")).strip()
        if not url:
            continue
        if not _is_trusted_source(url):
            return {
                "error": (
                    f"Refusing to publish: must_read URL '{url[:120]}' is "
                    f"from a domain outside the trusted source list. "
                    f"must_reads must come only from: "
                    f"{', '.join(sorted(TRUSTED_SOURCE_DOMAINS))}. "
                    f"If web_search returned a piece from elsewhere, you may "
                    f"still use it as context for the editorial, but pick "
                    f"must_reads from trusted domains only. Swap this item "
                    f"for one from a trusted source and call publish_edition "
                    f"again."
                )
            }
        try:
            path = (urlparse(url).path or "").lower()
        except Exception:
            path = ""
        for pattern in VENDOR_URL_PATH_PATTERNS:
            if pattern in path:
                return {
                    "error": (
                        f"Refusing to publish: must_read URL '{url[:120]}' "
                        f"looks like a vendor marketing page (path contains "
                        f"'{pattern}'). Pick an actual editorial article, "
                        f"not a product, platform, or pricing page. Swap "
                        f"this item and call publish_edition again."
                    )
                }

    # Gate 5c: contrarian is required and must not duplicate a must_read.
    # The contrarian's job is to challenge the editorial thesis, so it
    # cannot be the same URL as one of the supporting must_reads.
    if not contrarian or not isinstance(contrarian, dict):
        return {
            "error": (
                "Refusing to publish: contrarian is missing. Every edition "
                "needs one contrarian item: a piece that pushes back on or "
                "complicates the editorial's main thesis. Pick a piece from "
                "your research that disagrees with where you landed, and "
                "include it with title, url, source, and a 2-3 sentence note."
            )
        }
    contrarian_url = str(contrarian.get("url", "")).strip()
    if not contrarian_url:
        return {
            "error": (
                "Refusing to publish: contrarian.url is empty. Provide a "
                "real article URL from your research."
            )
        }
    mr_urls = {
        str(mr.get("url", "")).strip()
        for mr in mr_list if isinstance(mr, dict)
    }
    if contrarian_url in mr_urls:
        return {
            "error": (
                f"Refusing to publish: contrarian.url '{contrarian_url[:120]}' "
                f"is the same as one of your must_reads. The contrarian must "
                f"be a DIFFERENT piece that challenges your editorial thesis. "
                f"Pick another article from your research and call "
                f"publish_edition again."
            )
        }

    # Gate 5d: target_audience must be one of the allowed seniority
    # levels and must NOT match either of the last TARGET_AUDIENCE_LOOKBACK
    # editions. Forces P3 to rotate audience across weeks instead of
    # always landing on Staff/Senior PM.
    if not target_audience or not isinstance(target_audience, str):
        return {
            "error": (
                f"Refusing to publish: target_audience is missing. Pick "
                f"one of: {', '.join(TARGET_AUDIENCES)}. Use read_memory "
                f"to see what the last {TARGET_AUDIENCE_LOOKBACK} editions "
                f"used and pick a different level so the dispatch rotates "
                f"who P3 speaks to."
            )
        }
    if target_audience not in TARGET_AUDIENCES:
        return {
            "error": (
                f"Refusing to publish: target_audience '{target_audience}' "
                f"is not in the allowed set. Must be exactly one of: "
                f"{', '.join(TARGET_AUDIENCES)}."
            )
        }
    recent_ta = _recent_target_audiences()
    if target_audience in recent_ta:
        return {
            "error": (
                f"Refusing to publish: target_audience '{target_audience}' "
                f"was used in one of the last {TARGET_AUDIENCE_LOOKBACK} "
                f"editions ({recent_ta}). Pick a different level so the "
                f"dispatch rotates. Available options not in recent "
                f"history: "
                f"{[ta for ta in TARGET_AUDIENCES if ta not in recent_ta]}."
            )
        }

    # Gate 5e: hook_source must be present, must correspond to a source
    # in must_reads or contrarian (agent can't invent one), and must
    # NOT match any of the last HOOK_SOURCE_LOOKBACK editions. Forces
    # rotation of which publisher opens P1 across weeks.
    if not hook_source or not isinstance(hook_source, str):
        return {
            "error": (
                "Refusing to publish: hook_source is missing. Provide "
                "the source name (e.g. \"Lenny's Newsletter\", "
                "\"Stratechery\") of the article that opens paragraph 1 "
                "of the editorial. It must exactly match the 'source' "
                "field of one of your must_reads or the contrarian."
            )
        }
    hook_source_norm = hook_source.strip().lower()
    known_sources = [
        str(mr.get("source", "")).strip().lower()
        for mr in mr_list if isinstance(mr, dict)
    ]
    if isinstance(contrarian, dict):
        known_sources.append(str(contrarian.get("source", "")).strip().lower())
    if hook_source_norm not in known_sources:
        return {
            "error": (
                f"Refusing to publish: hook_source '{hook_source}' does "
                f"not match any source in must_reads or contrarian. The "
                f"article that opens P1 must be one of the items you "
                f"picked. Set hook_source to match the 'source' string "
                f"of that item exactly. Available sources: "
                f"{sorted(set(s for s in known_sources if s))}."
            )
        }
    recent_hs = _recent_hook_sources()
    if hook_source_norm in recent_hs:
        return {
            "error": (
                f"Refusing to publish: hook_source '{hook_source}' "
                f"opened one of the last {HOOK_SOURCE_LOOKBACK} editions "
                f"({recent_hs}). Anchor P1 in a different publisher "
                f"this week - the dispatch should rotate which source "
                f"gets the opening spot. Pick a must_read or contrarian "
                f"from a different source and rewrite P1 to hook off it."
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
                        f"Refusing to publish: must_read "
                        f"'{mr.get('title')}' has a descriptive 'why' "
                        f"containing '{phrase}'. Rewrite the 'why' field "
                        f"for THIS specific must_read only - do not touch "
                        f"the others. Remove the phrase '{phrase}' and "
                        f"replace with a sharp opinion. \n\n"
                        f"BAD (descriptive): 'This article shows how "
                        f"metrics can mislead teams.'\n"
                        f"GOOD (opinionated): 'The author is right that "
                        f"metric worship is the tax on lazy PM judgment "
                        f"- but wrong to blame data teams. Fix your OKR "
                        f"process before you blame the dashboard.'\n\n"
                        f"React, take a position, name what a Staff PM "
                        f"should DO. Then call publish_edition again."
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
        "target_audience": target_audience,
        "hook_source": hook_source,
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
                "target_audience": {
                    "type": "string",
                    "description": "Required. The PM seniority level paragraph 3 of the editorial speaks to. Must be one of: 'Senior PM', 'Staff PM', 'Lead/Principal PM'. Use read_memory to see what the last two editions used; the gate rejects re-using either.",
                    "enum": ["Senior PM", "Staff PM", "Lead/Principal PM"],
                },
                "hook_source": {
                    "type": "string",
                    "description": "Required. The source name of the article whose hook opens paragraph 1 of the editorial (e.g. 'Lenny\\'s Newsletter', 'Stratechery', 'SVPG'). Must exactly match the 'source' field of one of your must_reads or the contrarian. The gate rejects re-using the same hook_source as either of the last two editions - check read_memory and pick a different publisher this week.",
                },
                "editorial": {
                    "type": "string",
                    "description": "250-500 words, 3 paragraphs (hook/synthesis/implication). P3 is addressed specifically to the target_audience level. See system prompt.",
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
                    "description": "REQUIRED pick that challenges the editorial thesis. Must be a different URL from every must_read.",
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
- must_reads ONLY from these trusted domains: lennysnewsletter.com, reforge.com, svpg.com, mindtheproduct.com, blackboxofpm.com, producttalk.org, ben-evans.com, stratechery.com, exponentialview.co, firstround.com, medium.com, substack.com, hbr.org, every.to, platformer.news, casey.news, theverge.com, wired.com, techcrunch.com, theinformation.com. Subdomains count (e.g. user.substack.com).
- Vendor product/marketing pages are NEVER must_reads, even on trusted domains. Reject URLs whose path contains /platform/, /product/, /pricing/, /demo/, /features/, /integrations/, /solutions/, /signup/.
- If web_search returns a piece from an untrusted domain, you may use it as context for the editorial, but DO NOT cite it as a must_read.
- contrarian is REQUIRED, not optional. Pick a piece that challenges your editorial thesis. Its URL must be DIFFERENT from every must_read.
- Call publish_edition only after >=6 research tool calls (fetch_feeds + web_search + fetch_article combined).
- Editorial: 250-500 words, exactly 3 paragraphs. NEVER meta-narrative ("we'll explore", "in this edition", "our must-reads include"). The editorial IS the take, not a TOC.
- target_audience is REQUIRED. Pick one of: "Senior PM", "Staff PM", "Lead/Principal PM". Paragraph 3 of the editorial speaks SPECIFICALLY to that seniority level (its leverage points, its stakeholders, its decisions). The gate REJECTS re-using the same target_audience as either of the last two editions - check read_memory output and pick a level not used recently. Rotate so the dispatch hits different audiences across weeks.
- hook_source is REQUIRED. It is the exact 'source' string of the article that opens P1 of the editorial. Must appear in your must_reads or contrarian. The gate REJECTS re-using the same hook_source as either of the last two editions - Lenny's Newsletter cannot open three weeks running. Check read_memory and deliberately pick a P1 anchor from a source that has NOT opened recently. If your best hook naturally comes from a repeat source, promote a different article to the opener slot even if you keep the original as must_read #2 or #3.
- Every percentage or Nx multiplier in the editorial MUST appear verbatim in a tool result you received this run. Do NOT invent stats like "40% reduction" or "300% headcount" - the gate will reject and you will have to rewrite. If you do not have a real number, write the sentence without one.
- must_reads "why": OPINION, not description. NEVER use: "this article provides/shows/highlights/discusses", "shows how", "demonstrates", "showcases", "a great example", "valuable insight", "I appreciate/think/find", "real-world playbook". React to what the piece argues - agree, disagree, or call out what a Staff PM should DO about it.
- must_reads: 3-5 items. key_takeaways: 3-5. pm_homework: 1-3.

ROUTINE:
1. read_memory (avoid themes from last 2-3 weeks unless real news).
2. fetch_feeds broadly, then with topic filters as a theme emerges.
3. fetch_article on items that look important; web_search for context.
4. Pick ONE opinionated theme grounded in articles you actually read.
5. Pick 3-5 must_reads (trusted domains only), one contrarian (different URL), write 3-5 key_takeaways, 1-3 pm_homework.
6. Write editorial: P1 hook (name a specific article + author/company, sharp observation), P2 synthesis (connect 3+ pieces, take position), P3 implication for the chosen target_audience - speak to their specific leverage, stakeholders, and decisions (a Senior PM's daily reality is different from a Lead/Principal's).
7. Call publish_edition. Run ends.

Budget: <=14 tool calls before publish. If publish_edition is rejected, fix the specific issue named and call again. Never re-fetch a URL you already fetched in this run: the dispatcher will refuse with "duplicate_fetch". Pick a different URL or proceed to publish with what you have.

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
    # Short-circuit duplicate fetch_article calls. The sliding-window
    # trim eventually replaces older tool results with stubs, which
    # causes the agent to forget it already fetched a URL and try it
    # again. Returning a sharp refusal here is cheaper than letting it
    # burn an iteration on a duplicate fetch.
    if name == "fetch_article":
        url = args.get("url", "")
        if url and url in FETCHED_URLS:
            return {
                "error": (
                    "duplicate_fetch: you have already fetched this "
                    "URL in this run. Do not fetch it again. Use the "
                    "earlier result, or pick a different URL, or "
                    "proceed to publish_edition with what you have."
                ),
                "previous_outcome": FETCHED_URLS[url],
                "url": url,
            }
    try:
        result = fn(**args)
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"tool {name} failed: {e}"}
    # Record the URL after a fetch attempt so future calls to the same
    # URL short-circuit. Mark errors distinctly so the model knows not
    # to keep retrying a broken URL either.
    if name == "fetch_article":
        url = args.get("url", "")
        if url:
            if isinstance(result, dict) and result.get("error"):
                FETCHED_URLS[url] = f"errored: {str(result.get('error'))[:120]}"
            else:
                FETCHED_URLS[url] = "fetched successfully"
    # Log every tool result verbatim into TOOL_RESULTS_LOG so the
    # publish-time numeric grounding gate can check that any
    # percentage/multiplier in the editorial actually appears in the
    # research the agent did. Skip publish_edition's own result -
    # circular and pointless.
    if name != "publish_edition":
        try:
            TOOL_RESULTS_LOG.append(json.dumps(result, ensure_ascii=False))
        except Exception:
            pass
    return result


# Sliding-window cap: keep system + user + this many of the most recent
# messages in full. Older tool message contents are truncated to a short
# stub to stay within the model's per-request token budget. The agent
# does not need to re-read old tool dumps - it builds up understanding
# turn by turn - so truncating older results is structurally safe.
KEEP_RECENT_MESSAGES = 4
OLD_TOOL_RESULT_PREVIEW_CHARS = 200
# Even the most recent tool results get capped, because a single
# fetch_article or web_search dump can otherwise dominate the request.
# 3000 chars is roughly the working memory the agent needs to keep
# track of what it just fetched without re-fetching the same URLs.
RECENT_TOOL_RESULT_CAP_CHARS = 3000

# Tighter window applied after a 413 TPM error. Used to shrink the
# payload below the per-request token cap before the next retry, since
# the standard trim is by definition not aggressive enough if we hit
# the ceiling.
KEEP_RECENT_MESSAGES_ON_413 = 2
OLD_TOOL_RESULT_PREVIEW_CHARS_ON_413 = 80
RECENT_TOOL_RESULT_PREVIEW_CHARS_ON_413 = 500


def trim_message_history(messages):
    """In-place: truncate content of tool messages to save tokens.

    Leaves system prompt and user turn untouched. For older tool-role
    messages (anything older than KEEP_RECENT_MESSAGES), replaces the
    JSON content with a short preview plus a marker. For recent tool
    messages, applies a softer cap so a single oversized tool dump
    cannot single-handedly breach the per-request TPM ceiling.
    """
    if len(messages) <= 2:
        return
    cutoff = max(2, len(messages) - KEEP_RECENT_MESSAGES)
    # Older tool messages: stub them down hard.
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
    # Recent tool messages: cap them, but more generously.
    for i in range(cutoff, len(messages)):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= RECENT_TOOL_RESULT_CAP_CHARS + 50:
            continue
        msg["content"] = (
            content[:RECENT_TOOL_RESULT_CAP_CHARS]
            + "...[recent tool result capped for token budget]"
        )


def trim_message_history_aggressive(messages):
    """In-place: emergency shrink after a 413 TPM breach.

    The standard trim is a no-op when the payload is already over
    budget. This pass truncates ALL tool messages (including the most
    recent ones) and uses a tighter recent-window, so the next request
    can fit under the per-minute token ceiling. Called only on
    rate_limit_exceeded - normal iterations keep the gentler trim.
    """
    if len(messages) <= 2:
        return
    cutoff = max(2, len(messages) - KEEP_RECENT_MESSAGES_ON_413)
    # Older tool messages: stub them down to ~80 chars.
    for i in range(2, cutoff):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) <= OLD_TOOL_RESULT_PREVIEW_CHARS_ON_413 + 50:
            continue
        msg["content"] = (
            content[:OLD_TOOL_RESULT_PREVIEW_CHARS_ON_413]
            + "...[truncated under TPM pressure]"
        )
    # Recent tool messages: keep more context but still cap them.
    for i in range(cutoff, len(messages)):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) <= RECENT_TOOL_RESULT_PREVIEW_CHARS_ON_413 + 50:
            continue
        msg["content"] = (
            content[:RECENT_TOOL_RESULT_PREVIEW_CHARS_ON_413]
            + "...[truncated under TPM pressure]"
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
    # If the model returns assistant text without any tool call, we nudge
    # it back into the tool loop once. Cap at 1 to avoid an infinite
    # ping-pong of "you must call a tool" -> "I am a language model".
    NO_TOOL_CALL_MAX_NUDGES = 1
    no_tool_call_nudges = 0
    # Set to True after we inject the "stop researching, publish now"
    # nudge. Used to force tool_choice to publish_edition on the next
    # iteration so the model cannot keep stalling on more research.
    force_publish_next = False
    force_publish_nudged = False
    # Bound on how many times publish_edition can be rejected by gates
    # before we give up. Each quality gate (whitelist, contrarian,
    # numeric grounding, lazy why, etc.) returns a "Refusing to publish"
    # error and asks for a retry. If calibration is wrong they could
    # loop forever. Capping at 4 lets the model fix 3 distinct issues
    # in sequence before we abort the run.
    MAX_PUBLISH_REJECTS = 4
    publish_rejects = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[iter {iteration}]")
        # On the first turn, force the agent to actually call a tool.
        # Without this, the model sometimes skips straight to a
        # fabricated publish_edition with example.com URLs.
        if iteration == 1:
            tool_choice = "required"
        elif force_publish_next:
            # Pin the next tool call to publish_edition. Resets after
            # the call so a single rejected publish still has retries.
            tool_choice = {
                "type": "function",
                "function": {"name": "publish_edition"},
            }
            force_publish_next = False
        else:
            tool_choice = "auto"
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
            # Groq validation errors that we treat as recoverable:
            # - tool_use_failed: schema mismatch on arguments or
            #   malformed function-tag emission
            # - output_parse_failed: the model wrote raw chain-of-thought
            #   text instead of emitting a tool call at all (gpt-oss-120b
            #   sometimes leaks reasoning as prose)
            # Both get the same treatment: feed the error back as a
            # user message and let the agent retry with a perturbed
            # temperature, instead of crashing the whole run.
            recoverable_errors = ("tool_use_failed", "output_parse_failed")
            if (any(k in error_str for k in recoverable_errors)
                    and schema_retries < 3):
                schema_retries += 1
                print(
                    f"  Groq rejected model output. Asking agent to retry "
                    f"(retry {schema_retries}/3, temp -> "
                    f"{RETRY_TEMPERATURES[schema_retries - 1]})."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was rejected by the API. "
                        "Either your tool call was malformed / did not "
                        "match the tool's JSON schema, OR you wrote "
                        "reasoning as prose instead of emitting a tool "
                        "call. Do NOT think out loud - emit exactly one "
                        "well-formed tool call now, with no surrounding "
                        "text. Re-read the tool definitions if needed "
                        "(required fields, expected counts, types). "
                        f"Error from the API: {error_str[:600]}"
                    ),
                })
                continue
            # Per-request TPM breach: Groq compares the single request's
            # token count against the per-minute ceiling, so when this
            # fires the payload itself is over budget and sleeping alone
            # does nothing - the next request would be the same size.
            # First retry: trim aggressively to shrink the payload.
            # Second retry: sleep so the per-minute window rolls over,
            # in case the aggressive trim wasn't enough on its own.
            if ("rate_limit_exceeded" in error_str
                    and "tokens per minute" in error_str.lower()
                    and rate_limit_retries < RATE_LIMIT_MAX_RETRIES):
                rate_limit_retries += 1
                if rate_limit_retries == 1:
                    print(
                        f"  Hit TPM rate limit. Aggressive trim and "
                        f"retrying (retry {rate_limit_retries}/"
                        f"{RATE_LIMIT_MAX_RETRIES})."
                    )
                    trim_message_history_aggressive(messages)
                else:
                    print(
                        f"  Hit TPM rate limit again. Sleeping "
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
            # The agent sometimes drops into prose mode just shy of
            # publishing - "Here's a summary of what I found..." - which
            # leaves a perfectly good run unpublished. Nudge it back
            # into the tool loop once before giving up.
            if no_tool_call_nudges < NO_TOOL_CALL_MAX_NUDGES:
                no_tool_call_nudges += 1
                print(
                    f"  No tool call. Nudging the agent back to "
                    f"publish_edition (nudge "
                    f"{no_tool_call_nudges}/{NO_TOOL_CALL_MAX_NUDGES})."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "You returned text but no tool call. Your only "
                        "way to finish the run is to call publish_edition "
                        "with the fields populated from the research you "
                        "have already done. Do not respond with prose - "
                        "emit exactly one publish_edition tool call now."
                    ),
                })
                continue
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
            # Count publish_edition rejections from quality gates so we
            # can bail out instead of looping forever if calibration is
            # off. The gates all use "Refusing to publish" prefix. Also
            # re-pin tool_choice to publish_edition on the next
            # iteration so the agent cannot drift back to research
            # after a rejection - it must fix the publish call.
            # MAX_PUBLISH_REJECTS bounds the loop.
            if (tc.function.name == "publish_edition"
                    and isinstance(result, dict)
                    and str(result.get("error", "")).startswith("Refusing to publish")):
                publish_rejects += 1
                print(
                    f"  Publish rejected by quality gate "
                    f"({publish_rejects}/{MAX_PUBLISH_REJECTS})."
                )
                if publish_rejects >= MAX_PUBLISH_REJECTS:
                    print(
                        f"\nERROR: publish_edition rejected "
                        f"{publish_rejects} times. Aborting to avoid an "
                        f"infinite retry loop. Last error: "
                        f"{str(result.get('error',''))[:300]}"
                    )
                    return 1
                force_publish_next = True

        if published:
            print("\nDone.")
            return 0

        # If the agent has done enough research but is still spinning,
        # inject a user message demanding publish_edition next and pin
        # tool_choice so the next request must produce that call. Fires
        # once per run.
        research_calls = (
            TOOL_CALL_COUNTS.get("fetch_feeds", 0)
            + TOOL_CALL_COUNTS.get("web_search", 0)
            + TOOL_CALL_COUNTS.get("fetch_article", 0)
        )
        if (not force_publish_nudged
                and research_calls >= FORCE_PUBLISH_AT_RESEARCH_CALLS):
            force_publish_nudged = True
            force_publish_next = True
            print(
                f"  Research budget reached ({research_calls} calls). "
                f"Forcing publish_edition on next iteration."
            )
            messages.append({
                "role": "user",
                "content": (
                    f"You have made {research_calls} research tool calls "
                    f"(min required: {MIN_RESEARCH_CALLS}). Stop researching. "
                    f"Do NOT call fetch_feeds, web_search, fetch_article, "
                    f"or read_memory again. Your next and only tool call "
                    f"must be publish_edition, populated from what you "
                    f"have already fetched."
                ),
            })

    print(f"\nERROR: agent hit max iterations ({MAX_ITERATIONS}) without publishing.")
    return 1


if __name__ == "__main__":
    sys.exit(run_agent())
