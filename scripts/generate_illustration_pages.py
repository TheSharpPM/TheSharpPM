"""Generate static per-illustration HTML pages from illustrations/index.json.

Each entry in the JSON produces one file at the repo root named
`illustration-<slug>.html`. Each file contains:
- Open Graph + Twitter Card meta tags so LinkedIn / X / Slack / etc
  render a rich preview when the URL is shared.
- The illustration itself displayed large.
- Title, description, date, credit.
- A copy-link button.

Why static per-illustration files (instead of a dynamic
illustration.html?slug=xxx): social scrapers do NOT execute JavaScript.
The OG tags need to be present in the initial HTML for LinkedIn to
render the preview. A dynamic page would show a generic preview.

Safe to re-run: rebuilds all files from scratch each call. Removes
orphaned illustration-*.html files whose slug is no longer in the JSON.
"""

import glob
import json
import os
import re
import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INDEX_FILE = REPO_ROOT / "illustrations" / "index.json"
SITE_URL = "https://thesharppm.com"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-P5RKML0BZJ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-P5RKML0BZJ');
    </script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' https://www.googletagmanager.com https://www.google-analytics.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' https://www.google-analytics.com https://stats.g.doubleclick.net; frame-ancestors 'none'; base-uri 'self'; form-action 'self';">
    <meta name="referrer" content="strict-origin-when-cross-origin">
    <title>{title} - The Sharp PM</title>

    <!-- Open Graph / LinkedIn / Facebook -->
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="The Sharp PM">
    <meta property="og:title" content="{title}">
    <meta property="og:description" content="{description}">
    <meta property="og:image" content="{image_absolute_url}">
    <meta property="og:url" content="{page_absolute_url}">

    <!-- Twitter / X -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{title}">
    <meta name="twitter:description" content="{description}">
    <meta name="twitter:image" content="{image_absolute_url}">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --ink: #0f0e0d;
            --paper: #f5f2ed;
            --accent: #c8391a;
            --muted: #8a8278;
            --card: #ffffff;
            --border: #e0dbd4;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: var(--paper);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
            font-weight: 300;
            line-height: 1.6;
        }}
        header {{
            border-bottom: 2px solid var(--ink);
            padding: 0 2rem;
            background: var(--paper);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .header-inner {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 0;
        }}
        .logo {{ text-decoration: none; display: flex; align-items: center; }}
        .logo img {{ height: 36px; width: auto; }}
        .header-right {{ display: flex; align-items: center; gap: 1.5rem; }}
        .nav-link {{
            font-size: 0.75rem;
            color: var(--muted);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            text-decoration: none;
            padding-bottom: 2px;
        }}
        .nav-link:hover {{ color: var(--ink); }}
        .nav-link.active {{ color: var(--ink); }}
        .nav-link.active::after {{
            content: '';
            display: block;
            height: 2px;
            background: var(--accent);
            margin-top: 2px;
        }}
        main {{
            max-width: 900px;
            margin: 0 auto;
            padding: 3rem 2rem 4rem;
        }}
        .back-link {{
            display: inline-block;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            text-decoration: none;
            margin-bottom: 1.5rem;
        }}
        .back-link:hover {{ color: var(--ink); }}
        .meta-row {{
            display: flex;
            gap: 1rem;
            align-items: center;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.75rem;
        }}
        h1 {{
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 2.5rem;
            line-height: 1.15;
            margin-bottom: 1rem;
        }}
        .description {{
            font-size: 1.05rem;
            color: var(--ink);
            margin-bottom: 2rem;
        }}
        .illustration-img {{
            display: block;
            width: 100%;
            height: auto;
            border: 1px solid var(--border);
            background: var(--card);
            margin-bottom: 1.5rem;
        }}
        .credit {{
            font-size: 0.85rem;
            color: var(--muted);
            font-style: italic;
            margin-bottom: 2rem;
        }}
        .actions {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}
        .btn {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0.6rem 1rem;
            background: var(--ink);
            color: var(--paper);
            text-decoration: none;
            border: 1px solid var(--ink);
            cursor: pointer;
            transition: background 0.15s;
        }}
        .btn:hover {{ background: var(--accent); border-color: var(--accent); }}
        .btn-secondary {{
            background: transparent;
            color: var(--ink);
        }}
        .btn-secondary:hover {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
        footer {{
            border-top: 2px solid var(--ink);
            padding: 2rem;
            text-align: center;
            font-size: 0.75rem;
            color: var(--muted);
        }}
        footer a {{ color: var(--accent); text-decoration: none; }}
        @media (max-width: 600px) {{
            main {{ padding: 1.5rem; }}
            h1 {{ font-size: 1.75rem; }}
            .header-right {{ gap: 1rem; }}
        }}
    </style>
</head>
<body>

<header>
    <div class="header-inner">
        <a href="index.html" class="logo">
            <img src="The_Sharp_PM_Header3.png" alt="The Sharp PM">
        </a>
        <div class="header-right">
            <a href="index.html" class="nav-link">Daily Digest</a>
            <a href="agent.html" class="nav-link">Agent Sharp</a>
            <a href="illustrations.html" class="nav-link active">Illustrations</a>
            <a href="about.html" class="nav-link">About</a>
        </div>
    </div>
</header>

<main>
    <a href="illustrations.html" class="back-link">&larr; All illustrations</a>

    <div class="meta-row">
        <span>{date_human}</span>
    </div>

    <h1>{title}</h1>
    <p class="description">{description}</p>

    <img src="{image_url}" alt="{title}" class="illustration-img">

    <p class="credit">{credit}</p>

    <div class="actions">
        <a class="btn" href="https://www.linkedin.com/sharing/share-offsite/?url={page_absolute_url_encoded}" target="_blank" rel="noopener">Share on LinkedIn</a>
        <button class="btn btn-secondary" id="copy-link">Copy link</button>
    </div>
</main>

<footer>
    <p>The Sharp PM - <a href="https://github.com/TheSharpPM/TheSharpPM" target="_blank">GitHub</a></p>
</footer>

<script>
    document.getElementById('copy-link').addEventListener('click', function() {{
        navigator.clipboard.writeText(window.location.href).then(function() {{
            var btn = document.getElementById('copy-link');
            var original = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(function() {{ btn.textContent = original; }}, 1500);
        }});
    }});
</script>

</body>
</html>
"""


def _slug_safe(slug):
    """Only allow safe chars in slugs to avoid path-traversal in the
    output filename."""
    return re.sub(r"[^a-z0-9\-]", "", (slug or "").lower())


def _date_human(iso_date):
    """Turn '2026-08-09' into 'Aug 9, 2026' for display."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%b %-d, %Y")
    except Exception:
        return iso_date or ""


def _render(entry):
    slug = _slug_safe(entry.get("slug", ""))
    if not slug:
        return None, None
    title = entry.get("title", "Untitled")
    description = entry.get("description", "")
    date = entry.get("date", "")
    credit = entry.get("credit", "")
    image_rel = entry.get("image", "")
    # image path is relative to illustrations/ folder
    image_url = f"illustrations/{image_rel}"
    image_absolute_url = f"{SITE_URL}/{image_url}"
    page_filename = f"illustration-{slug}.html"
    page_absolute_url = f"{SITE_URL}/{page_filename}"
    from urllib.parse import quote
    page_absolute_url_encoded = quote(page_absolute_url, safe="")
    html_out = TEMPLATE.format(
        title=escape(title),
        description=escape(description),
        credit=escape(credit) if credit else "",
        date_human=escape(_date_human(date)),
        image_url=escape(image_url, quote=True),
        image_absolute_url=escape(image_absolute_url, quote=True),
        page_absolute_url=escape(page_absolute_url, quote=True),
        page_absolute_url_encoded=page_absolute_url_encoded,
    )
    return page_filename, html_out


def main():
    if not INDEX_FILE.exists():
        print(f"ERROR: {INDEX_FILE} not found.")
        return 1

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("illustrations", [])
    print(f"Found {len(entries)} illustration(s) in index.")

    # Track which files we generate so we can prune orphans.
    wanted_files = set()
    for entry in entries:
        filename, html_out = _render(entry)
        if not filename:
            print(f"  skip: no slug in entry {entry.get('title')}")
            continue
        out_path = REPO_ROOT / filename
        wanted_files.add(filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        print(f"  wrote {filename}")

    # Prune orphaned illustration-*.html files (entries removed from
    # the index should not leave stale pages behind).
    existing = [
        os.path.basename(p) for p in
        glob.glob(str(REPO_ROOT / "illustration-*.html"))
    ]
    for f in existing:
        if f not in wanted_files:
            (REPO_ROOT / f).unlink()
            print(f"  removed orphan {f}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
