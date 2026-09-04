"""Tests for the Langfuse tool-call instrumentation.

Two things are checked:

  span_output()  the real function, imported from agent_sharp - keeps
                 span payloads small across every tool return shape
  the span block a mirror of the per-tool-call block in run(), driven
                 against a fake Langfuse client

The block is a mirror rather than the real code because it lives inline
in run(), which needs a Groq client and a full agent loop to reach. The
invariant it exists to protect is that tracing never breaks the run:
whatever Langfuse does - refuse to open a span, throw mid-update - the
tool result must pass through untouched and the span must still end.

KEEP IN SYNC: if the block in run() changes, update run_tool_span_block
below to match, or these tests will quietly pass against dead code.

Run with:  python3 scripts/tests/test_tool_spans.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Results, load_agent_sharp  # noqa: E402

agent = load_agent_sharp()
span_output = agent.span_output


class FakeSpan:
    def __init__(self, name, input, metadata, log, fail_update=False):
        self.name = name
        self.log = log
        self.fail_update = fail_update
        self.ended = False
        self.level = None
        log.append(("start", name, json.dumps(input)[:80]))

    def update(self, **kwargs):
        if self.fail_update:
            raise RuntimeError("langfuse network blip")
        if "level" in kwargs:
            self.level = kwargs["level"]
        self.log.append(("update", self.name, sorted(kwargs)))

    def end(self):
        self.ended = True
        self.log.append(("end", self.name))


class FakeRootSpan:
    def __init__(self, log, fail_start=False, fail_update=False):
        self.log = log
        self.fail_start = fail_start
        self.fail_update = fail_update
        self.spans = []

    def start_span(self, name, input=None, metadata=None):
        if self.fail_start:
            raise RuntimeError("langfuse unreachable")
        span = FakeSpan(name, input, metadata, self.log, self.fail_update)
        self.spans.append(span)
        return span


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = type(
            "Function", (), {"name": name, "arguments": arguments}
        )()
        self.id = "call_test"


def run_tool_span_block(tc, root_span, execute_tool, iteration=3):
    """Mirror of the per-tool-call span block in agent_sharp.run()."""
    args = {}
    parse_error = None
    try:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except json.JSONDecodeError as e:
        parse_error = f"could not parse arguments: {e}"

    tool_span = None
    if root_span:
        try:
            tool_span = root_span.start_span(
                name=f"tool_{tc.function.name}",
                input=(
                    args if parse_error is None
                    else {"raw_arguments": (tc.function.arguments or "")[:500]}
                ),
                metadata={"iteration": iteration},
            )
        except Exception as e:
            print(f"  langfuse tool span open failed (continuing): {e}")
            tool_span = None

    if parse_error:
        result = {"error": parse_error}
    else:
        try:
            result = execute_tool(tc.function.name, args)
        except json.JSONDecodeError as e:
            result = {"error": f"could not parse arguments: {e}"}

    if tool_span:
        try:
            if isinstance(result, dict) and result.get("error"):
                tool_span.update(
                    level="ERROR",
                    status_message=str(result["error"])[:500],
                )
            tool_span.update(output=span_output(result))
        except Exception as e:
            print(f"  langfuse tool span close failed (continuing): {e}")
        finally:
            try:
                tool_span.end()
            except Exception:
                pass
    return result


OK_RESULT = {"count": 3, "items": [1, 2, 3]}
tool_ok = lambda name, args: dict(OK_RESULT)
tool_error = lambda name, args: {"error": "Tavily timeout"}


def tool_raises(name, args):
    raise json.JSONDecodeError("bad", "{}", 0)


def main():
    r = Results()

    print("--- span_output keeps payloads small ---")
    shapes = {
        "fetch_article": {"url": "https://x.test/a", "text": "y" * 50000,
                          "truncated": True},
        "fetch_feeds": {"count": 12, "items": [{"t": 1}] * 12},
        "web_search": {"count": 5, "results": [{"r": 1}] * 5},
        "read_memory": {"editions": [{"e": 1}] * 4, "count": 4},
        "read_memory (empty)": {"editions": [],
                                "note": "No past editions yet (first run)."},
        "publish_edition": {"status": "published",
                            "file": "agent/2026-09-04.json"},
        "tool error": {"error": "x" * 900},
        "duplicate fetch": {"error": "duplicate_fetch: ...",
                            "previous_outcome": "fetched successfully",
                            "url": "https://x.test/a"},
    }
    for label, shape in shapes.items():
        rendered = json.dumps(span_output(shape))
        r.check(f"{label} span payload stays under 700 chars "
                f"({len(rendered)})", len(rendered) < 700, rendered)

    big = json.dumps(shapes["fetch_article"])
    small = json.dumps(span_output(shapes["fetch_article"]))
    print(f"  article body {len(big)} chars -> {len(small)} chars on the span")

    r.check("a non-dict result is still rendered",
            span_output("weird") == {"result": "weird"})
    r.check("an unrecognised shape falls back to its keys",
            span_output({"odd": 1}) == {"keys": ["odd"]})

    print("\n--- span lifecycle ---")
    log = []
    root = FakeRootSpan(log)
    out = run_tool_span_block(
        FakeToolCall("fetch_feeds", '{"max_per_feed": 5}'), root, tool_ok)
    r.check("a successful call ends its span with no error level",
            root.spans[0].ended and root.spans[0].level is None)
    r.check("the tool result passes through untouched", out == OK_RESULT)

    log = []
    root = FakeRootSpan(log)
    run_tool_span_block(
        FakeToolCall("web_search", '{"query": "x"}'), root, tool_error)
    r.check("a tool that returns an error dict marks the span ERROR",
            root.spans[0].level == "ERROR" and root.spans[0].ended)

    log = []
    root = FakeRootSpan(log)
    out = run_tool_span_block(
        FakeToolCall("fetch_article", '{"url": broken'), root, tool_ok)
    r.check("malformed arguments are reported, not raised",
            "could not parse arguments" in out["error"])
    r.check("unparseable arguments are logged raw on the span",
            "raw_arguments" in log[0][2], log[0][2])

    log = []
    root = FakeRootSpan(log)
    out = run_tool_span_block(
        FakeToolCall("fetch_feeds", "{}"), root, tool_raises)
    r.check("a JSONDecodeError from execute_tool does not propagate",
            "could not parse arguments" in out["error"])
    r.check("the span still ends after that error", root.spans[0].ended)

    print("\n--- tracing failures must never break the run ---")
    root = FakeRootSpan([], fail_start=True)
    out = run_tool_span_block(FakeToolCall("fetch_feeds", "{}"), root, tool_ok)
    r.check("a span that cannot open leaves the tool result intact",
            out == OK_RESULT)

    root = FakeRootSpan([], fail_update=True)
    out = run_tool_span_block(FakeToolCall("fetch_feeds", "{}"), root, tool_ok)
    r.check("a span that throws mid-update is still ended",
            root.spans[0].ended)
    r.check("a throwing span leaves the tool result intact", out == OK_RESULT)

    out = run_tool_span_block(FakeToolCall("fetch_feeds", "{}"), None, tool_ok)
    r.check("the untraced path (no root span) works", out == OK_RESULT)

    return r.report()


if __name__ == "__main__":
    sys.exit(main())
