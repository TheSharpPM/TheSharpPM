# Agent Sharp tests

Offline tests for `scripts/agent_sharp.py`. No network, no API keys, no
Groq quota - they exercise pure logic only.

## Running

From the repository root:

```
python3 scripts/tests/test_publish_gates.py
python3 scripts/tests/test_tool_spans.py
```

Each prints a pass/fail line per case and exits non-zero on failure, so
they drop into CI as-is.

There is no pytest dependency on purpose: `requirements.txt` pins only
what a real run needs, and these are plain scripts. `harness.py` stubs
any runtime dependency that is not installed, so both files run on a
bare checkout and against the real pinned packages after
`pip install -r requirements.txt`.

## What is covered

| File | Covers |
| --- | --- |
| `test_publish_gates.py` | The sourcing gates in `tool_publish_edition`: citation provenance, recency, verbatim pull quotes, headline theme repetition |
| `test_tool_spans.py` | `span_output()` payload shaping, and the tool-span lifecycle including every tracing-failure path |

`test_publish_gates.py` redirects `AGENT_DIR` and `INDEX_FILE` to a
temporary directory, because `tool_publish_edition` writes the edition
and rewrites `index.json` on success. It restores a pristine index
before each case: a case that publishes would otherwise trip the
`target_audience` and `hook_source` rotation gates for every case after
it.

## Known limitation

`test_tool_spans.py` drives a **mirror** of the per-tool-call span block
in `run()`, not the block itself, which lives inline in a function that
needs a Groq client and a full agent loop to reach. `span_output()` is
imported and tested for real. If the block in `run()` changes, update
`run_tool_span_block` to match - otherwise these tests pass against code
that no longer ships.
