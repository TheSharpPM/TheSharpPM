"""MCP server exposing the Sharp PM archive over stdio.

Run it directly to serve:   python3 scripts/mcp/server.py

This file is intentionally almost empty of logic. Every query lives in
tspm_data.py; all that happens here is the protocol wiring. If a result
looks wrong, the bug is in tspm_data.py. If the client cannot see the
tools at all, the bug is here.

TWO THINGS THAT CATCH EVERYONE on stdio transport:

1. stdout IS the protocol channel. A stray print() writes garbage into
   the JSON-RPC stream and the client fails with a confusing parse
   error that points nowhere near the print. Diagnostics go to stderr -
   there is a `log()` helper below, use it instead of print.

2. A tool's docstring becomes the description the MODEL reads when it
   decides whether to call it. It is prompt engineering, not API docs.
   Vague descriptions are the most common reason a working server still
   "doesn't do anything" - the model never picks the tool. The same
   care the Groq agent's TOOL_DECLARATIONS get applies here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tspm_data  # noqa: E402

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as e:  # pragma: no cover - setup help
    # Note the import path: this is the mcp 2.x API, where what used to
    # be called FastMCP was renamed MCPServer. Any v1 example you find
    # online will import mcp.server.fastmcp and fail here.
    sys.stderr.write(
        f"Cannot start: {e}\n"
        "Install the SDK first:  pip install 'mcp>=2'\n"
    )
    raise SystemExit(1)


def log(message):
    """Diagnostics to stderr. NEVER print() in a stdio server."""
    sys.stderr.write(f"[tspm-mcp] {message}\n")
    sys.stderr.flush()


server = MCPServer(
    name="sharp-pm-archive",
    instructions=(
        "Read-only access to The Sharp PM's archive: ~870 curated links "
        "from the last 90 days, and every published weekly edition with "
        "its editorial, contrarian pick and team homework.\n\n"
        "Call list_tags first when you need to filter - it returns the "
        "exact tag, type and source values, which are capitalised "
        "phrases like 'Product Strategy', not slugs.\n\n"
        "For what the editor ARGUED about a topic, use search_editorials, "
        "not search_items: the links are a feed anyone could assemble, "
        "while the editorials, contrarian picks and homework are "
        "original and exist nowhere else. Always cite the url you used."
    ),
)


@server.tool()
def list_tags() -> dict:
    """List every tag, content type and source in the archive, with counts.

    Call this BEFORE filtering with search_items. Tags are capitalised
    phrases ('AI Strategy', 'Product Strategy'), so a guessed slug like
    'ai-strategy' silently matches nothing.
    """
    return tspm_data.list_tags()


@server.tool()
def search_items(query: str | None = None, tag: str | None = None,
                 content_type: str | None = None, since: str | None = None,
                 limit: int = 10) -> dict:
    """Search ~870 curated PM links from the last 90 days.

    Filters combine with AND. Use for "what has been published about X",
    reading recommendations, or tracking a topic over time.

    query: words matched against title, tags and summary.
    tag: exact tag from list_tags, e.g. 'Product Strategy'.
    content_type: one of article, trend, podcast, job, event.
    since: ISO date, e.g. '2026-09-01'. Only items on or after it.
    limit: how many to return, 1-50, default 10.
    """
    return tspm_data.search_items(query=query, tag=tag,
                                  content_type=content_type, since=since,
                                  limit=limit)


@server.tool()
def search_editorials(query: str, limit: int = 5) -> dict:
    """Search what the editor ARGUED, across every published edition.

    Returns the editorial take, key takeaways, team homework, the
    contrarian pick and each must-read with the reason it was chosen.

    This is the original material - use it for "has The Sharp PM covered
    X", "what is the contrarian view on X", or "give my team homework on
    X". Prefer it over search_items whenever the question is about an
    opinion rather than about what exists.
    """
    return tspm_data.search_editorials(query, limit=limit)


@server.tool()
def get_edition(date: str | None = None) -> dict:
    """Fetch one published edition in full.

    date: ISO date like '2026-10-11'. Omit it for the most recent
    edition. If no edition was published that day, the reply lists the
    dates that are available rather than failing silently.
    """
    return tspm_data.get_edition(date)


if __name__ == "__main__":
    # stdio is the default transport and the right one for a local
    # server: the client launches this process and talks over the pipe,
    # so there is no port, no auth and nothing to deploy. Switching to
    # transport="streamable-http" is the whole change needed to serve it
    # over HTTP later.
    log(f"serving {tspm_data.REPO_ROOT}")
    server.run(transport="stdio")
