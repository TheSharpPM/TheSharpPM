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
]
 
# ── LANGUAGE FILTER ───────────────────────────────────────────────────────────
 
def is_english(text):
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii <= 3
 
# ── API CHECK ─────────────────────────────────────────────────────────────────
 
def check_api_available():
    if not OPENROUTER_API_KEY:
        return False
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
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        data = response.json()
        return "choices" in data
    except Exception:
        return False
 
# ── ANALYSE WITH AI ───────────────────────────────────────────────────────────
 
def analyse(title, content):
    prompt = (
        "Analyse this article and return a JSON object with two fields:\n"
        "1. 'summary': 2-3 sentences in English summarizing the key insight for a Product Manager.\n"
        "2. 'tags': a list of 2-4 short tags that best describe the topics of this article. "
        "Each tag should be 1-2 words, capitalized (e.g. 'Product Strategy', 'User Research', 'AI Tools'). "
        "Generate tags that are specific and meaningful for a Product Manager audience.\n\n"
        "Return ONLY valid JSON, no explanation.\n\n"
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
        text = data["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed.get("summary", "Summary not available."), parsed.get("tags", [])
    except Exception as e:
        print("  Warning API error: " + str(e))
        return "Summary not available.", []
 
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
            summary, tags = analyse(title, content)
 
            items.append({
                "title": title,
                "url": url,
                "source": feed_config["source"],
                "type": feed_config["type"],
                "date": date,
                "summary": summary,
                "tags": tags,
            })
 
    except Exception as e:
        print("  Error reading " + feed_config["source"] + ": " + str(e))
 
    return items
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    print("The Sharp PM - Aggregator running\n")
 
    print("Checking OpenRouter API...")
    if not check_api_available():
        print("API not available or out of credits. Skipping run.")
        return
    print("API available. Proceeding...\n")
 
    # Load existing articles
    existing_items = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_items = existing_data.get("items", [])
 
    # Existing URLs to avoid duplicates
    existing_urls = {item["url"] for item in existing_items}
 
    # Fetch new articles
    new_items = []
    for feed in FEEDS:
        print("Feed: " + feed["source"])
        items = fetch_feed(feed)
        fresh = [i for i in items if i["url"] not in existing_urls]
        new_items.extend(fresh)
        print("   " + str(len(fresh)) + " new articles\n")
 
    # Merge new with existing
    all_items = new_items + existing_items
    all_items.sort(key=lambda x: x.get("date") or "", reverse=True)
 
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total": len(all_items),
        "items": all_items,
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print("Done! " + str(len(all_items)) + " total articles saved.")
 
 
if __name__ == "__main__":
    main()
