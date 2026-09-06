"""Shared article fetching and a short-lived body cache.

Two pipelines want the text of the same articles. The daily aggregator
sees every item hours before Agent Sharp runs on Sunday, so the weekly
agent was re-fetching pages the aggregator had already had in hand -
and paying an iteration out of its budget of 15 for each one.

This module holds the fetch primitives both pipelines use, plus a cache
the aggregator fills and the agent reads.

What the cache is NOT for: saving tokens. A cached body costs exactly
what a freshly fetched one costs once it enters the model's context.
What it actually buys:

  - iterations, which is the binding constraint. The 2026-09-06 run used
    13 of 15, four of them on fetch_article.
  - better text per token. That run's SVPG fetch spent roughly 300 of
    its 2,500 characters on nav chrome ("Workshops Product Masterclass
    Transformed Loved Services Transformation Engagements ...") before
    the article began. Cleaning happens once, offline, here.
  - a citation gate that is affordable. Refusing to cite an unread piece
    is the fix for a sources_read_ratio of 0.4, and it only works if
    reading is close to free.

Bodies deliberately do NOT go in data.json: index.html fetches that file
on every page load, and it is already ~500KB.

The fetch primitives live here rather than in agent_sharp.py so the
daily pipeline cannot end up with a weaker fetcher than the weekly one.
Both get the same SSRF checks.
"""
import hashlib
import ipaddress
import json
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

CACHE_DIR = Path("cache/articles")
# One week. Long enough that Sunday's run sees everything the aggregator
# collected since the previous edition, short enough that the cache does
# not become an archive - agent/*.json is the archive.
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
# Stored longer than the agent's MAX_ARTICLE_CHARS (2500) so the agent
# slices from a fuller body rather than a body already cut to its own
# limit. Raising the agent's limit later then needs no cache rebuild.
MAX_CACHED_CHARS = 6000

MAX_REDIRECTS = 5
MAX_FETCH_BYTES = 1_048_576
FETCH_TIMEOUT_SECONDS = 15
USER_AGENT = "Mozilla/5.0 (The Sharp PM article fetcher)"


# ── FETCH PRIMITIVES ──────────────────────────────────────────────────────────

def strip_html(raw):
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def url_is_safe_to_fetch(url):
    """Return (ok, reason). Blocks anything that's not public http/https.

    Defends against SSRF via prompt injection: a feed snippet could
    instruct the LLM to call fetch_article(http://169.254.169.254/...)
    to hit the GHA runner's cloud-metadata endpoint, an internal
    service, or a loopback address. We allow only http(s) and only to
    DNS names that resolve to public IPs.

    The aggregator needs this as much as the agent does - it follows
    links out of third-party RSS feeds, which anyone can publish to.
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


def fetch_text(url):
    """Fetch a URL and return (text, error). Exactly one is None.

    Redirects are followed by hand so every hop is re-validated: a
    public host must not be able to 302 us onto a private target.
    stream=True plus a byte counter caps the download so a hostile
    server cannot stream gigabytes before the text slice applies.
    """
    ok, reason = url_is_safe_to_fetch(url)
    if not ok:
        return None, f"refusing to fetch: {reason}"
    try:
        current_url = url
        response = None
        for hop in range(MAX_REDIRECTS + 1):
            response = requests.get(
                current_url,
                timeout=FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
                stream=True,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                next_url = response.headers.get("Location")
                response.close()
                if not next_url:
                    return None, "redirect with no Location header"
                if hop == MAX_REDIRECTS:
                    return None, f"too many redirects (> {MAX_REDIRECTS})"
                next_url = requests.compat.urljoin(current_url, next_url)
                ok, reason = url_is_safe_to_fetch(next_url)
                if not ok:
                    return None, f"refusing redirect to unsafe target: {reason}"
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
        return strip_html(raw), None
    except Exception as e:
        return None, str(e)


# ── CACHE ─────────────────────────────────────────────────────────────────────

def _key(url):
    """Filename for a URL. sha1 of the exact string - no normalisation,
    so a cache hit means the agent is reading the same URL it would have
    fetched, not a near-miss."""
    return hashlib.sha1(str(url).encode("utf-8")).hexdigest()


def _entry_path(url):
    return CACHE_DIR / f"{_key(url)}.json"


def get(url, now=None):
    """Cached body for a URL, or None when absent, unreadable or stale."""
    path = _entry_path(url)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if entry.get("url") != url:
        # sha1 collision, or a hand-edited file. Treat as a miss rather
        # than serving one article's text under another's URL.
        return None
    fetched_at = entry.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return None
    if (now or time.time()) - fetched_at > CACHE_TTL_SECONDS:
        return None
    return entry


def put(url, text, title=None, source=None, now=None):
    """Store a body. Returns the entry, or None if there was nothing
    worth storing. Never raises - a cache write failing must not take
    down the pipeline that was only trying to be helpful."""
    if not text or not str(text).strip():
        return None
    entry = {
        "url": url,
        "title": title or "",
        "source": source or "",
        "fetched_at": float(now or time.time()),
        "text": str(text)[:MAX_CACHED_CHARS],
    }
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _entry_path(url)
        # Atomic write, matching how editions and data.json are written:
        # a half-written entry must never be committed.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        print(f"  cache write failed for {url}: {e}")
        return None
    return entry


def load_all(now=None):
    """Every live entry, as {url: entry}. Used to seed a run in one pass
    instead of stat-ing the cache once per candidate URL."""
    if not CACHE_DIR.exists():
        return {}
    now = now or time.time()
    entries = {}
    for path in CACHE_DIR.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = entry.get("url")
        fetched_at = entry.get("fetched_at")
        if not url or not isinstance(fetched_at, (int, float)):
            continue
        if now - fetched_at > CACHE_TTL_SECONDS:
            continue
        entries[url] = entry
    return entries


def prune(now=None):
    """Delete entries past the TTL. Returns how many were removed.

    Called by the aggregator each day. Without it the cache is an
    ever-growing committed directory; with it the working set stays at
    roughly one week of citable articles.
    """
    if not CACHE_DIR.exists():
        return 0
    now = now or time.time()
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = entry.get("fetched_at")
            stale = (not isinstance(fetched_at, (int, float))
                     or now - fetched_at > CACHE_TTL_SECONDS)
        except Exception:
            stale = True   # unreadable entry is dead weight either way
        if stale:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed
