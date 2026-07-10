"""Keyword-based tag inference for aggregator items.

Called by aggregate.py as a fallback when the LLM summariser is
rate-limited or errored, so items still get useful tags in data.json
instead of empty lists. Also called by the backfill helper to
retroactively tag items that were added before this existed.

Design notes:
- Case-insensitive substring match (not word-boundary) so 'copilot',
  'CoPilot', 'GitHub Copilot' all hit the same keyword.
- Order of TAG_KEYWORDS does not matter - we score by hit count, then
  sort by count desc, then alphabetically for determinism.
- max_tags caps the noise (usually 3, matching the aggregator's
  historical style of 2-3 tags per item).
- If NOTHING matches, we return the type-based default so items are
  never left with empty tags (which breaks tag filters in index.html).
- Jobs are always tagged Career on top of any other matches.
"""

TAG_KEYWORDS = {
    "AI Tools": [
        "chatgpt", "claude ", "gemini", "gpt-", "gpt4", "gpt5", "gpt-5",
        "llm ", "large language model", "copilot", "codex", "cursor ",
        "windsurf", "openai", "anthropic", "mistral", "perplexity",
        "sonnet", "opus 4", "haiku", "ai agent", "coding agent", "glm-",
    ],
    "AI Strategy": [
        "ai strategy", "ai roadmap", "ai transformation", "ai adoption",
        "ai-first", "ai first", "artificial intelligence",
        "machine learning", "ai product", "ai integration",
        " ai ", "genai", "gen ai", "generative ai", " with ai",
        " using ai", "leverage ai", " ai-", " ai.",
    ],
    "Automation": [
        "automation", "no-code", "low-code", "zapier", "n8n",
        "workflow automation", "auto-generated",
    ],
    "Growth": [
        "growth", "acquisition", "activation", "referral", "viral",
        "growth loop", "aarrr", "funnel", "conversion rate",
    ],
    "Retention": [
        "retention", "churn", "engagement rate", "lifetime value",
        " ltv", "cohort analysis",
    ],
    "Pricing": [
        "pricing", "monetization", "monetisation", "monetize", "monetise",
        "revenue model", "billing", "subscription", "paywall", "freemium",
    ],
    "User Research": [
        "user research", "user interview", "usability", "customer discovery",
        "user testing", "ux research", "qualitative research", "personas",
        "customer research",
    ],
    "Metrics": [
        "metric", " kpi", " okr", "north star", "analytics", "measurement",
        "dashboard",
    ],
    "Product Strategy": [
        "product strategy", "roadmap", "product vision",
        "product-market fit", " pmf", "strategic bets", "product bets",
    ],
    "Leadership": [
        "leadership", "vp of product", "vp product", "chief product",
        "director of product", " cpo", "team lead", "product leader",
    ],
    "Career": [
        "career", "hiring", "interview", "resume", "compensation",
        "salary", "senior pm", "staff pm", "principal pm", "junior pm",
        "product owner", "job description",
    ],
    "Product Design": [
        "product design", "ux design", "ui design", "user experience",
        "design system", "figma", "designer", "wireframe", "prototype",
    ],
    "Prioritization": [
        "prioritization", "prioritisation", "prioritize", "prioritise",
        "rice framework", "moscow", "backlog", "trade-off", "tradeoff",
    ],
    "Stakeholders": [
        "stakeholder", "cross-functional", "alignment", "buy-in",
        "executive team", "sales alignment", "engineering alignment",
    ],
    "Communication": [
        "communication", "writing", "presentation", "narrative",
        "storytelling", "documentation", "product doc", "one-pager",
    ],
    "Data": [
        "data pipeline", " sql", "data warehouse", " etl",
        "instrumentation", "data engineer", "data analyst",
    ],
    "Security": [
        "security", "vulnerability", "breach", "encryption",
        "authentication", " auth ", "package hack", "supply chain",
        "malware",
    ],
    "Privacy": [
        "privacy", " gdpr", " ccpa", "consent", "cookie policy",
    ],
    "Compliance": [
        "compliance", "regulation", " fda", " hipaa", " sox", "audit",
        "regulatory",
    ],
    "Competitive Analysis": [
        "competitor", "competitive", "market analysis", "positioning",
        "landscape",
    ],
    "Platform Strategy": [
        "platform", "ecosystem", "marketplace", "api strategy", " sdk",
        "developer platform",
    ],
    "Productivity": [
        "productivity", "efficiency", "focus time", "flow state",
        "deep work",
    ],
    "Product Culture": [
        "product culture", "team culture", "product principles", " values",
        "product mindset",
    ],
    "Risk Management": [
        "risk management", "risk mitigation", "governance",
        "risk register",
    ],
    "Engineering Velocity": [
        "velocity", "shipping fast", "release cadence", "ci/cd",
        "developer experience", " devx",
    ],
    "Community": [
        "community wisdom", "user community", "forum",
    ],
    "Hardware Differentiation": [
        "hardware", "device", " chip", "processor", "silicon",
    ],
}

# Default tag per feed type, used when no keyword matches. Prevents
# items from being returned with empty tags (which the front-end tag
# filters do not handle gracefully).
DEFAULT_TAGS_BY_TYPE = {
    "article": "Product Strategy",
    "job": "Career",
    "podcast": "Product Culture",
    "video": "Product Strategy",
    "event": "Product Culture",
    # 'trend' was defaulting to 'AI Strategy' which put every Product
    # Hunt launch that failed keyword matching into an AI bucket
    # incorrectly. 'Product Strategy' is generic enough to not mislead.
    "trend": "Product Strategy",
    "course": "Career",
}


def infer_tags(title, summary, feed_type, max_tags=3):
    """Return 1-max_tags tags inferred from title + summary keywords.

    Falls back to a type-based default when nothing matches. Job items
    always include 'Career' first."""
    haystack = f"{title or ''} {summary or ''}".lower()
    # Pad so leading/trailing space-anchored patterns like ' sql' still
    # match when the haystack starts or ends with the term.
    padded = f" {haystack} "

    scores = {}
    for tag, keywords in TAG_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            if kw in padded:
                hits += 1
        if hits:
            scores[tag] = hits

    if not scores:
        default = DEFAULT_TAGS_BY_TYPE.get(feed_type, "Product Strategy")
        return [default]

    # Sort by hits desc, then tag name for determinism when ties.
    sorted_tags = [t for t, _ in sorted(
        scores.items(), key=lambda kv: (-kv[1], kv[0])
    )]

    result = sorted_tags[:max_tags]

    # Jobs always include Career - if it did not surface from keywords
    # (e.g. very short job listing text), inject it in first position.
    if feed_type == "job" and "Career" not in result:
        result = ["Career"] + result[: max_tags - 1]

    return result
