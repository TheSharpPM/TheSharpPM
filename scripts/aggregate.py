import feedparser
import json
import os
import re
import requests
from datetime import datetime, timezone
from time import mktime
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_ITEMS_PER_FEED = 3  # articles per source
OUTPUT_FILE = "data.json"
 
# RSS sources per category
FEEDS = [
    # PM Articles
    {"url": "https://www.lennysnewsletter.com/feed", "source": "Lenny's Newsletter", "type": "article"},
    {"url": "https://www.reforge.com/blog/rss.xml", "source": "Reforge", "type": "article"},
    {"url": "https://www.svpg.com/articles/rss", "source": "SVPG", "type": "article"},
    {"url": "https://productcoalition.com/feed", "source": "Product Coalition", "type": "article"},
    {"url": "https://www.mindtheproduct.com/feed/", "source": "Mind the Product", "type": "article"},
 
    # AI Trends
    {"url": "https://www.ben-evans.com/benedictevans/rss.xml", "source": "Benedict Evans", "type": "trend"},
    {"url": "https://stratechery.com/feed/", "source": "Stratechery", "type": "trend"},
 
    # Frameworks & templates (via Medium tags)
    {"url": "https://medium.com/feed/tag/product-management", "source": "Medium PM", "type": "framework"},
]

# ── LANGUAGE FILTER ───────────────────────────────────────────────────────────
 
def is_english(text):
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii <= 3
 
# ── SUMMARISE WITH AI ─────────────────────────────────────────────────────────
 
def summarise(title, content):
    if not OPENROUTER_API_KEY:
        return "Summary not available."

    prompt = (
        "You are an assistant for Product Managers. Summarize the following article in 2-3 concise sentences focusing on the key insight and what is actionable for a PM. Reply ONLY with the summary, no introduction or commentary.\n\n"
        "Title: " + title + "\n"
        "Content: " + content[:1500]
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://thesharppm.github.io",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            timeout=30,
        )
        data = response.json()
        print(f"  API response: {data}")
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ API error: {e}")
        return "Summary not available."
 
 
# ── FETCH FEEDS ───────────────────────────────────────────────────────────────
 
def fetch_feed(feed_config):
    items = []
    try:
        parsed = feedparser.parse(feed_config["url"])
        entries = parsed.entries[:MAX_ITEMS_PER_FEED]
 
        for entry in entries:
            title = entry.get("title", "No title")
 
            if not is_english(title):
                print("  Skipped (non-English): " + title[:40])
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
 
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
 
            url = entry.get("link", "#")
 
            print("  -> " + title[:60] + "...")
            summary = summarise(title, content)
 
            items.append({
                "title": title,
                "url": url,
                "source": feed_config["source"],
                "type": feed_config["type"],
                "date": date,
                "summary": summary,
            })
 
    except Exception as e:
        print("  Error reading " + feed_config["source"] + ": " + str(e))
 
    return items
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    print("🚀 The Sharp PM — Aggregator running\n")
    all_items = []
 
    for feed in FEEDS:
        print(f"📡 {feed['source']}")
        items = fetch_feed(feed)
        all_items.extend(items)
        print(f"   ✓ {len(items)} articles collected\n")
 
    # Sort by data (most recent first)
    all_items.sort(key=lambda x: x.get("date") or "", reverse=True)
 
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total": len(all_items),
        "items": all_items,
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print(f"✅ All done! {len(all_items)} articles saved on {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()
