import feedparser
import json
import os
import re
import requests
from datetime import datetime, timezone
from time import mktime
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MAX_ITEMS_PER_FEED = 3
OUTPUT_FILE = "data.json"
 
FEEDS = [
    {"url": "https://www.lennysnewsletter.com/feed", "source": "Lenny's Newsletter", "type": "article"},
    {"url": "https://www.reforge.com/blog/rss.xml", "source": "Reforge", "type": "article"},
    {"url": "https://www.svpg.com/articles/rss", "source": "SVPG", "type": "article"},
    {"url": "https://productcoalition.com/feed", "source": "Product Coalition", "type": "article"},
    {"url": "https://www.mindtheproduct.com/feed/", "source": "Mind the Product", "type": "article"},
    {"url": "https://www.ben-evans.com/benedictevans/rss.xml", "source": "Benedict Evans", "type": "trend"},
    {"url": "https://stratechery.com/feed/", "source": "Stratechery", "type": "trend"},
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
        "Summarize this article in exactly 2-3 sentences in English. "
        "Be concise and focus on the key insight for a Product Manager. "
        "Only output the summary, nothing else.\n\n"
        "Title: " + title + "\n"
        "Content: " + content[:1500]
    )
 
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + OPENROUTER_API_KEY,
                "Content-Type": "application/json",
                "HTTP-Referer": "https://thesharppm.github.io",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            },
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("  Warning API error: " + str(e))
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
    print("The Sharp PM - Aggregator running\n")
    all_items = []
 
    for feed in FEEDS:
        print("Feed: " + feed["source"])
        items = fetch_feed(feed)
        all_items.extend(items)
        print("   " + str(len(items)) + " articles collected\n")
 
    all_items.sort(key=lambda x: x.get("date") or "", reverse=True)
 
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total": len(all_items),
        "items": all_items,
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print("Done! " + str(len(all_items)) + " articles saved to " + OUTPUT_FILE)
 
 
if __name__ == "__main__":
    main()
