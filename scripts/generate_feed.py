"""Generate agent/feed.xml (RSS 2.0) from agent/index.json and edition
files. Called after each Agent Sharp run so external services (Zapier,
Publer, Buffer) can watch the feed and cross-post to LinkedIn.

The feed lists the last MAX_ITEMS_IN_FEED editions in reverse chronological
order. Each item carries:
- title: headline_theme
- link + guid: permalink to the edition page
- pubDate: RFC-822 from generated_at, falling back to the edition date
- description: CDATA HTML with the editorial plus the must_reads list, so
  the automation vendor has everything it needs to compose a LinkedIn post
  without having to fetch the site.

Safe to re-run: it rebuilds feed.xml from scratch each call. If no editions
exist yet, writes an empty (but valid) channel.
"""

import json
import re
import sys
from datetime import datetime, timezone
from email.utils import format_datetime
from html import escape
from pathlib import Path

AGENT_DIR = Path("agent")
INDEX_FILE = AGENT_DIR / "index.json"
FEED_FILE = AGENT_DIR / "feed.xml"

SITE_URL = "https://thesharppm.com"
FEED_URL = f"{SITE_URL}/agent/feed.xml"
CHANNEL_TITLE = "The Sharp PM — Agent Sharp"
CHANNEL_LINK = f"{SITE_URL}/agent.html"
CHANNEL_DESCRIPTION = (
    "Weekly editorial dispatch for Product Managers. AI-curated must-reads "
    "and a contrarian pick every Sunday. EXPERIMENTAL: written and published "
    "by an AI agent with no human review. It misreads sources and states weak "
    "conclusions confidently - read the linked originals, do not cite this."
)

# Prepended to every item body. The feed is where this content leaves the
# site: a reader in an RSS client, or a LinkedIn post composed from the
# feed, never sees the banner on the page. The warning has to travel with
# the item or it does not travel at all.
ITEM_DISCLAIMER = (
    "<p><em><strong>Experimental / AI-generated.</strong> Written and "
    "published automatically by an AI agent, with no human review. It "
    "misreads sources and recycles picks. Read the linked originals - "
    "do not treat this as fact, and do not cite it.</em></p>"
)
CHANNEL_LANGUAGE = "en"
MAX_ITEMS_IN_FEED = 20


def _edition_url(date):
    return f"{SITE_URL}/agent-edition.html?date={date}"


def _safe_url(url):
    """Return the URL if it is http(s), else '#'.

    escape() neutralises quote-breakout but not the scheme, so without
    this a javascript: or data: URL that reached an edition JSON would
    go verbatim into an <a href> in the feed description. Mirrors the
    safeUrl() helper the site's pages already use."""
    trimmed = str(url or "").strip()
    if re.match(r"^https?://", trimmed, re.IGNORECASE):
        return trimmed
    return "#"


def _pub_date(edition):
    """Best-effort RFC-822 pubDate. Prefer generated_at, fall back to the
    edition date at noon UTC, then to now()."""
    gen = edition.get("generated_at")
    if gen:
        try:
            return format_datetime(
                datetime.fromisoformat(gen.replace("Z", "+00:00"))
            )
        except Exception:
            pass
    date = edition.get("edition") or ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        try:
            return format_datetime(
                datetime.fromisoformat(f"{date}T12:00:00+00:00")
            )
        except Exception:
            pass
    return format_datetime(datetime.now(timezone.utc))


def _build_description(edition):
    """CDATA HTML body of the RSS description. Includes the editorial and
    a must_reads list so the LinkedIn post can be composed without a
    second HTTP fetch."""
    parts = [ITEM_DISCLAIMER]
    editorial = str(edition.get("editorial") or "").strip()
    if editorial:
        # Preserve paragraphs. The editorial has \n\n between paragraphs.
        for para in re.split(r"\n\s*\n", editorial):
            para = para.strip()
            if para:
                parts.append(f"<p>{escape(para)}</p>")
    must_reads = edition.get("must_reads") or []
    if must_reads:
        parts.append("<h3>Must reads</h3><ul>")
        for mr in must_reads[:5]:
            if not isinstance(mr, dict):
                continue
            title = escape(str(mr.get("title") or ""))
            url = escape(_safe_url(mr.get("url")), quote=True)
            source = escape(str(mr.get("source") or ""))
            parts.append(f'<li><a href="{url}">{title}</a> — {source}</li>')
        parts.append("</ul>")
    contrarian = edition.get("contrarian")
    if isinstance(contrarian, dict) and contrarian.get("url"):
        parts.append("<h3>Contrarian</h3>")
        title = escape(str(contrarian.get("title") or ""))
        url = escape(_safe_url(contrarian.get("url")), quote=True)
        source = escape(str(contrarian.get("source") or ""))
        parts.append(f'<p><a href="{url}">{title}</a> — {source}</p>')
    link = _edition_url(edition.get("edition") or "")
    parts.append(
        f'<p><a href="{escape(link, quote=True)}">Read the full edition</a></p>'
    )
    return "".join(parts)


def _build_item(edition):
    date = str(edition.get("edition") or "")
    title = str(edition.get("headline_theme") or f"Agent Sharp — {date}")
    link = _edition_url(date)
    pub_date = _pub_date(edition)
    description = _build_description(edition)
    return (
        "    <item>\n"
        f"      <title>{escape(title)}</title>\n"
        f"      <link>{escape(link, quote=True)}</link>\n"
        f'      <guid isPermaLink="true">{escape(link, quote=True)}</guid>\n'
        f"      <pubDate>{pub_date}</pubDate>\n"
        f"      <description><![CDATA[{description}]]></description>\n"
        "    </item>\n"
    )


def main():
    editions_meta = []
    if INDEX_FILE.exists():
        try:
            index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            editions_meta = index.get("editions", [])[:MAX_ITEMS_IN_FEED]
        except Exception as e:
            print(f"WARNING: could not parse {INDEX_FILE}: {e}")

    items_xml = []
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for meta in editions_meta:
        date = str(meta.get("date", ""))
        if not date_re.match(date):
            continue
        ef = AGENT_DIR / f"{date}.json"
        if not ef.exists():
            continue
        try:
            ed = json.loads(ef.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {date}: {e}")
            continue
        items_xml.append(_build_item(ed))

    last_build = format_datetime(datetime.now(timezone.utc))
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        '  <channel>\n'
        f"    <title>{escape(CHANNEL_TITLE)}</title>\n"
        f'    <link>{escape(CHANNEL_LINK, quote=True)}</link>\n'
        f"    <description>{escape(CHANNEL_DESCRIPTION)}</description>\n"
        f"    <language>{CHANNEL_LANGUAGE}</language>\n"
        f"    <lastBuildDate>{last_build}</lastBuildDate>\n"
        f'    <atom:link href="{escape(FEED_URL, quote=True)}" rel="self" '
        'type="application/rss+xml" />\n'
        + "".join(items_xml)
        + "  </channel>\n</rss>\n"
    )

    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic write to avoid a half-written feed being committed on error.
    tmp = FEED_FILE.with_suffix(".xml.tmp")
    tmp.write_text(feed, encoding="utf-8")
    tmp.replace(FEED_FILE)
    print(f"Wrote {FEED_FILE} with {len(items_xml)} item(s).")


if __name__ == "__main__":
    sys.exit(main() or 0)
