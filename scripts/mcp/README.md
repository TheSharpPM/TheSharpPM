# Sharp PM archive — MCP server

A local MCP server exposing the archive as four tools. Read-only, no
database, no hosting: it reads this repo's own JSON files.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r scripts/mcp/requirements.txt
```

`.mcp.json` in the repo root already points Claude Code at
`.venv/bin/python scripts/mcp/server.py`, so once the venv exists the
tools appear the next time you start Claude Code in this directory.
Check with `/mcp`.

The venv is gitignored. These deps are kept out of the top-level
`requirements.txt` deliberately — the aggregator and the weekly agent
should not grow a dependency they never import, and the test suites
need to keep running without it.

## The tools

| Tool | Use it for |
|---|---|
| `list_tags` | The 27 tags, 5 types and 15 sources, with counts. Call first when filtering. |
| `search_items` | The ~870 links. Filters: `query`, `tag`, `content_type`, `since`, `limit`. |
| `search_editorials` | What the editor *argued*: takeaways, homework, contrarian pick, must-reads with reasons. |
| `get_edition` | One edition in full. Omit `date` for the latest. |

Try it: *"what tags are in the archive?"*, then *"has The Sharp PM
covered platform gatekeeping?"*

## Why it is split in two files

`tspm_data.py` holds every query and imports nothing but the standard
library. `server.py` holds the protocol wiring and almost no logic.

That split is the main design decision here, and it pays three ways:

1. The queries are testable with **zero dependencies**, so
   `scripts/tests/test_mcp_data.py` runs in the existing CI alongside
   the other suites instead of needing its own job.
2. The same logic serves over stdio now and HTTP later, untouched.
3. A bug is never ambiguous. Wrong results → `tspm_data.py`. Client
   can't see the tools → `server.py`.

## Two things that catch everyone

**1. On stdio, stdout *is* the protocol channel.** A stray `print()`
writes garbage into the JSON-RPC stream, and the client fails with a
parse error that points nowhere near the `print`. Diagnostics go to
stderr — `server.py` has a `log()` helper for exactly this.

**2. A docstring is prompt engineering, not API docs.** The tool
description is what the *model* reads when deciding whether to call
your tool. A working server whose descriptions are vague looks broken,
because the model never picks the tool. Note how `search_editorials`
says *when to prefer it* over `search_items` — the same thing the Groq
agent's `TOOL_DECLARATIONS` do in `agent_sharp.py`.

A third, specific to this SDK: **mcp 2.x renamed `FastMCP` to
`MCPServer`**. Any tutorial importing `mcp.server.fastmcp` is 1.x and
will fail against the pinned version here.

## Testing it without a client

`list_tools()` and `call_tool()` are plain async methods, so the wiring
can be exercised directly:

```bash
./.venv/bin/python -c "
import asyncio, sys; sys.path.insert(0, 'scripts/mcp')
import server as S
print([t.name for t in asyncio.run(S.server.list_tools())])
print(asyncio.run(S.server.call_tool('list_tags', {})).content[0].text[:200])
"
```

For a real end-to-end check, spawn it and speak JSON-RPC over the pipe:
`initialize` → `notifications/initialized` → `tools/list` →
`tools/call`. That is what a client does, and it catches the stdout
mistake above, which in-process calls cannot.

## Next step: HTTP

```python
server.run(transport="streamable-http")
```

That one line is the whole transport change — the tools and the data
layer are untouched. Running it on localhost teaches the HTTP lifecycle
with nothing to sign up for. Only publishing it for other people needs
a host, since GitHub Pages serves static files and cannot run a server.
