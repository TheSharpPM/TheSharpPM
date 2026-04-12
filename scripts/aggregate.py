import feedparser
import json
import os
import requests
from datetime import datetime, timezone
from time import mktime
 
# ── CONFIG ────────────────────────────────────────────────────────────────────
 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3.1-8b-instruct:free"
MAX_ITEMS_PER_FEED = 3  # artigos por fonte
OUTPUT_FILE = "data.json"
 
# Fontes RSS por categoria
FEEDS = [
    # Artigos de PM
    {"url": "https://www.lennysnewsletter.com/feed", "source": "Lenny's Newsletter", "type": "article"},
    {"url": "https://www.reforge.com/blog/rss.xml", "source": "Reforge", "type": "article"},
    {"url": "https://www.svpg.com/articles/rss", "source": "SVPG", "type": "article"},
    {"url": "https://productcoalition.com/feed", "source": "Product Coalition", "type": "article"},
    {"url": "https://www.mindtheproduct.com/feed/", "source": "Mind the Product", "type": "article"},
 
    # Tendências AI
    {"url": "https://www.ben-evans.com/benedictevans/rss.xml", "source": "Benedict Evans", "type": "trend"},
    {"url": "https://stratechery.com/feed/", "source": "Stratechery", "type": "trend"},
 
    # Frameworks & templates (via Medium tags)
    {"url": "https://medium.com/feed/tag/product-management", "source": "Medium PM", "type": "framework"},
]
 
# ── SUMMARISE WITH AI ─────────────────────────────────────────────────────────
 
def summarise(title, content):
    """Pede ao LLM um resumo em português de 2-3 frases."""
    if not OPENROUTER_API_KEY:
        return "Resumo não disponível (API key não configurada)."
 
    prompt = f"""És um assistente para Product Managers. Resume o seguinte artigo em português europeu em 2-3 frases concisas e úteis para um PM. Foca-te no insight principal e no que é accionável.
 
Título: {title}
Conteúdo: {content[:1500]}
 
Responde APENAS com o resumo, sem introduções nem comentários."""
 
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
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  ⚠ Erro na API: {e}")
        return "Resumo não disponível."
 
 
# ── FETCH FEEDS ───────────────────────────────────────────────────────────────
 
def fetch_feed(feed_config):
    """Vai buscar os artigos mais recentes de um RSS feed."""
    items = []
    try:
        parsed = feedparser.parse(feed_config["url"])
        entries = parsed.entries[:MAX_ITEMS_PER_FEED]
 
        for entry in entries:
            # Data
            date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                date = datetime.fromtimestamp(
                    mktime(entry.published_parsed), tz=timezone.utc
                ).isoformat()
 
            # Conteúdo para resumir
            content = ""
            if hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "content"):
                content = entry.content[0].value
 
            # Remove HTML básico
            import re
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
 
            title = entry.get("title", "Sem título")

            # Filtrar artigos não ingleses usando caracteres não-ASCII como indicador
            non_ascii = sum(1 for c in title if ord(c) > 127)
            if non_ascii > 3:
                print(f"  ⏭ Ignorado (não inglês): {title[:40]}")
                continue
            url = entry.get("link", "#")
 
            print(f"  → {title[:60]}...")
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
        print(f"  ⚠ Erro a ler {feed_config['source']}: {e}")
 
    return items
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
    print("🚀 The Sharp PM — Aggregator a correr\n")
    all_items = []
 
    for feed in FEEDS:
        print(f"📡 {feed['source']}")
        items = fetch_feed(feed)
        all_items.extend(items)
        print(f"   ✓ {len(items)} artigos recolhidos\n")
 
    # Ordena por data (mais recentes primeiro)
    all_items.sort(key=lambda x: x.get("date") or "", reverse=True)
 
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total": len(all_items),
        "items": all_items,
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
 
    print(f"✅ Concluído! {len(all_items)} artigos guardados em {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()
