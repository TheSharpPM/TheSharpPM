import feedparser
import html
import json
import os
import re
import requests
import sys
import unicodedata
from datetime import datetime, timezone, timedelta
from time import mktime, sleep

from tag_inference import infer_tags

# ── CONFIG ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")
FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_FALLBACK_MODELS",
        "openai/gpt-oss-120b:free,"
        "meta-llama/llama-3.3-70b-instruct:free,"
        "nvidia/nemotron-nano-9b-v2:free,"
        "minimax/minimax-m2.5:free",
    ).split(",") if m.strip()
]
MAX_ITEMS_PER_FEED = int(os.environ.get("MAX_ITEMS_PER_FEED", "2"))
OUTPUT_FILE = "data.json"
MAX_DAYS = 90
REQUEST_DELAY = 1.5  # seconds to sleep after each successful API call
RETRY_BACKOFFS = [15, 30]  # seconds to wait before retrying on provider error
 
FEEDS = [
    # Articles
    {"url": "https://www.lennysnewsletter.com/feed", "source": "Lenny's Newsletter", "type": "article"},
    {"url": "https://www.reforge.com/blog/rss.xml", "source": "Reforge", "type": "article"},
    {"url": "https://www.svpg.com/articles/rss", "source": "SVPG", "type": "article"},
    {"url": "https://productcoalition.com/feed", "source": "Product Coalition", "type": "article"},
    {"url": "https://www.mindtheproduct.com/feed/", "source": "Mind the Product", "type": "article"},
    {"url": "https://blackboxofpm.com/feed", "source": "Black Box of PM", "type": "article"},
    {"url": "https://www.producttalk.org/feed/", "source": "Product Talk", "type": "article"},
 
    # AI Trends
    {"url": "https://www.ben-evans.com/benedictevans/rss.xml", "source": "Benedict Evans", "type": "trend"},
    {"url": "https://stratechery.com/feed/", "source": "Stratechery", "type": "trend"},
    {"url": "https://www.exponentialview.co/feed", "source": "Exponential View", "type": "trend"},
 
    # Articles (Medium)
    {"url": "https://medium.com/feed/tag/product-management", "source": "Medium PM", "type": "article"},
    {"url": "https://medium.com/feed/tag/product-strategy", "source": "Medium Strategy", "type": "article"},
 
    # Jobs (remote)
    {"url": "https://remoteok.com/remote-product-manager-jobs.rss", "source": "Remote OK", "type": "job"},
    {"url": "https://weworkremotely.com/categories/remote-product-jobs.rss", "source": "We Work Remotely", "type": "job"},
    {"url": "https://www.workatastartup.com/jobs.rss?role=product", "source": "Work at a Startup", "type": "job"},
 
    # Additional Articles
    {"url": "https://www.firstround.com/review/feed.xml", "source": "First Round Review", "type": "article"},
    {"url": "https://www.producthunt.com/feed", "source": "Product Hunt", "type": "trend"},
    {"url": "https://hnrss.org/best?q=product+manager", "source": "Hacker News", "type": "trend"},
 
    # Podcasts
    {"url": "https://feeds.transistor.fm/lenny-s-podcast", "source": "Lenny's Podcast", "type": "podcast"},
    {"url": "https://www.mindtheproduct.com/feed/podcast/", "source": "Mind the Product Podcast", "type": "podcast"},
    {"url": "https://feeds.simplecast.com/4MvgQ73R", "source": "Masters of Scale", "type": "podcast"},
    {"url": "https://rss.art19.com/how-i-built-this", "source": "How I Built This", "type": "podcast"},

    # Additional podcasts (broaden the pool - the base 4 above have
    # been producing near-zero items; several of these are on more
    # active PM/tech circuits).
    {"url": "https://feeds.simplecast.com/qc7NJdlL", "source": "Product Thinking", "type": "podcast"},
    {"url": "https://feeds.transistor.fm/acquired", "source": "Acquired", "type": "podcast"},
    {"url": "https://feeds.megaphone.fm/thetwentyminutevc", "source": "20VC", "type": "podcast"},
    {"url": "https://feeds.blubrry.com/feeds/thisisproductmanagement.xml", "source": "This Is Product Management", "type": "podcast"},
    {"url": "https://feeds.simplecast.com/JGE3yC0V", "source": "a16z Podcast", "type": "podcast"},
    {"url": "https://rss.art19.com/tim-ferriss-show", "source": "The Tim Ferriss Show", "type": "podcast"},
 
    # Videos (YouTube RSS)
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCJXGnMEFEpKKBqAVBpTSoQQ", "source": "Y Combinator", "type": "video"},
    {"url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC4r3BHFUBxfXGzKxb5HFmRg", "source": "Product School", "type": "video"},
 
    # Events
    {"url": "https://www.mindtheproduct.com/feed/events/", "source": "Mind the Product Events", "type": "event"},
    {"url": "https://productledalliance.com/feed/", "source": "Product-Led Alliance", "type": "event"},
]
 
# ── LANGUAGE FILTER ───────────────────────────────────────────────────────────
 
def is_english(text):
    """Reject titles that look non-English. Count only Letter-category
    non-ASCII chars (accented Latin, Cyrillic, CJK, etc.) and treat
    emoji, dashes, curly quotes and other typographic symbols as
    neutral. PT/ES/FR/DE titles typically carry multiple diacritics
    and get rejected; English titles with a single emoji or a foreign
    name (Müller, Søren) still pass."""
    letter_non_ascii = sum(
        1 for c in (text or "")
        if ord(c) > 127 and unicodedata.category(c).startswith("L")
    )
    return letter_non_ascii <= 1
 
# ── ANALYSE WITH AI ───────────────────────────────────────────────────────────

# Models that have been marked as exhausted (rate-limited/unavailable) this run.
# Once every candidate model is exhausted, analyse() returns (None, None) to
# signal the caller to stop making further API calls.
_exhausted_models = set()


def _call_model(model, prompt):
    """Call a single model once. Returns (summary, tags) on success, or
    ("__RATE_LIMIT__", None) if the provider reports a rate limit / upstream
    error. Raises on transport/parse errors."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + OPENROUTER_API_KEY,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://thesharppm.com",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    data = response.json()
    if "choices" not in data:
        msg = data.get("error", {}).get("message", str(data))
        print("  Provider error on " + model + ": " + msg)
        return "__RATE_LIMIT__", None
    text = data["choices"][0]["message"]["content"]
    if not text:
        raise ValueError("Empty response content")
    text = text.strip().replace("```json", "").replace("```", "").strip()
    # Extract JSON object even if model adds text around it
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    parsed = json.loads(text)
    return parsed.get("summary", "Summary not available."), parsed.get("tags", [])


def analyse(title, content):
    prompt = (
        "Return ONLY valid JSON with two fields:\n"
        "- summary: 2-3 sentences about this article for a Product Manager\n"
        "- tags: array of 2-3 tags from: AI Strategy, AI Tools, Product Strategy, "
        "Leadership, Metrics, User Research, Growth, Career, Prioritization, "
        "Stakeholders, Product Design, Retention, Data, Communication, Productivity, "
        "Competitive Analysis, Platform Strategy, Product Culture\n\n"
        "Title: " + title + "\nContent: " + content[:800]
    )
 
    candidates = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]

    for model in candidates:
        if model in _exhausted_models:
            continue

        # Try this model up to len(RETRY_BACKOFFS)+1 times (initial + backoffs)
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            try:
                summary, tags = _call_model(model, prompt)
            except Exception as e:
                print("  Warning API error on " + model + ": " + str(e))
                summary, tags = "__RATE_LIMIT__", None

            if summary != "__RATE_LIMIT__":
                # Success — pace the next request and return
                sleep(REQUEST_DELAY)
                return summary, tags

            # Retry with backoff if we have retries left
            if attempt < len(RETRY_BACKOFFS):
                wait = RETRY_BACKOFFS[attempt]
                print("  Retrying " + model + " in " + str(wait) + "s...")
                sleep(wait)

        # All retries failed for this model — mark exhausted and try next
        print("  Model exhausted: " + model)
        _exhausted_models.add(model)

    # Every candidate model is exhausted — signal caller to stop
    return None, None
 
# ── FETCH FEEDS ───────────────────────────────────────────────────────────────
 
def _fallback_summary(content):
    """Used when every LLM in the fallback chain is rate-limited or
    errored. Builds a passable summary from the RSS entry's own text
    so the item still ends up in data.json instead of being dropped
    on the floor. A bad provider day should not mean a silent
    zero-content commit. No marker text appended - we used to but it
    showed up on the live site and looked like noise."""
    snippet = (content or "").strip()
    if not snippet:
        return "Summary unavailable."
    snippet = snippet[:280].rstrip()
    if len(snippet) == 280:
        snippet += "..."
    return snippet


def fetch_feed(feed_config, existing_urls):
    items = []
    try:
        parsed = feedparser.parse(feed_config["url"])
        entries = parsed.entries[:MAX_ITEMS_PER_FEED]

        for entry in entries:
            title = entry.get("title", "No title")

            if not is_english(title):
                print("  Skipped (non-English): " + title[:40])
                continue

            url = entry.get("link", "#")

            if url in existing_urls:
                print("  Skipped (already exists): " + title[:40])
                continue

            date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                date = datetime.fromtimestamp(
                    mktime(entry.published_parsed), tz=timezone.utc
                ).isoformat()

            content = ""
            if hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "content"):
                content = entry.content[0].value

            # Decode HTML entities (&apos;, &#8217;, &#x2014;, &nbsp;, ...)
            # BEFORE stripping tags - some RSS feeds escape both. Without
            # this step the fallback summaries that go straight to the
            # site keep the raw entity garbage.
            content = html.unescape(content)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            print("  -> " + title[:60] + "...")
            result = analyse(title, content)

            # If every LLM in the fallback chain is rate-limited, do
            # NOT drop the item. Add it with a fallback summary derived
            # from the RSS-provided text so the site keeps moving on
            # bad provider days. Tag with keyword inference so the item
            # still lands in useful tag buckets on the site, and mark
            # is_fallback=True so a future job can re-run LLM analysis
            # over these when quotas reset.
            if result[0] is None:
                print("  LLM exhausted - using RSS fallback summary + keyword tags.")
                summary = _fallback_summary(content)
                tags = infer_tags(title, content, feed_config["type"])
                is_fallback = True
            else:
                summary, tags = result
                is_fallback = False

            items.append({
                "title": title,
                "url": url,
                "source": feed_config["source"],
                "type": feed_config["type"],
                "date": date,
                "summary": summary,
                "tags": tags,
                "is_fallback": is_fallback,
            })

    except Exception as e:
        print("  Error reading " + feed_config["source"] + ": " + str(e))

    return items
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    print("The Sharp PM - Aggregator running\n")

    # Fail loudly if the API key is missing, instead of letting every
    # provider call return "Bearer None" and silently producing an
    # empty / partial data.json.
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set. Aborting.")
        sys.exit(1)

    # Load existing articles
    existing_items = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_items = existing_data.get("items", [])
 
    # Existing URLs to avoid duplicates
    existing_urls = {item["url"] for item in existing_items}
 
    # Fetch new articles. Even if every LLM model in the fallback
    # chain is exhausted, we keep iterating feeds: items get added
    # with a fallback RSS-derived summary rather than being dropped.
    # This stops the silent zero-content commit pattern we saw when
    # OpenRouter's free tier was globally rate-limited.
    new_items = []
    for feed in FEEDS:
        print("Feed: " + feed["source"])
        items = fetch_feed(feed, existing_urls)
        new_items.extend(items)
        print("   " + str(len(items)) + " new articles\n")
 
    # Merge new with existing
    all_items = new_items + existing_items
    all_items.sort(key=lambda x: x.get("date") or "", reverse=True)
 
    # Remove articles older than MAX_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_DAYS)
    all_items = [i for i in all_items if not i.get("date") or datetime.fromisoformat(i["date"]) >= cutoff]
 
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total": len(all_items),
        "items": all_items,
    }
 
    # Atomic write: serialise to a temp file, then rename. Prevents a half-
    # written data.json from being committed if the process dies mid-write.
    tmp_file = OUTPUT_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, OUTPUT_FILE)

    print("Done! " + str(len(all_items)) + " total articles saved.")
 
 
if __name__ == "__main__":
    main()
 
