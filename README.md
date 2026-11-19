# indian-stocks-mcp-apps

**Two consumers of [`indian-stocks-mcp`](https://github.com/Shubhampm18/indian-stocks-mcp)** — the point-in-time NSE/BSE server. Both install the published package; neither reaches into a checkout.

```
agent/        LangChain agent (create_agent) + durable SQLite memory + stdio client
dashboard/    5-page Streamlit UI (Price / Patterns / Indicators / Returns / Peers)
```

## Why this repo exists

The server's guarantee is easy to claim and easy to break. These two apps exercise it end to end, through different paths:

- **`agent/`** talks to the server over **MCP stdio**, exactly as Claude Desktop or Cursor would. If the handshake, tool schemas, or `isError` envelope regress, this breaks.
- **`dashboard/`** imports the **data layer directly** (no MCP hop) and renders it. If the `as_of` funnel or the adjusted-price rebuild regresses, the chart and the tables show it.

Same functions, two transports — that's the point.

## Quick start

```bash
uv sync --all-packages                      # workspace: agent + dashboard
cd agent && uv run python client.py --demo      # no LLM key needed
uv run python chat.py                            # needs LLM_API_KEY
cd ../dashboard && uv run streamlit run app.py
```

`uv run` installs the `dev` group, which includes the server package — needed for the schema-contract test and for spawning the server from the same interpreter. Use `uv run --no-dev` only if you don't need the agent tests.

## Agent

The loop **is** `create_agent` (LangChain 1.x): one model, one flat tool
list, no sub-agents, no checkpointers, no vector store. Server tool schemas
come from the server itself via `load_mcp_tools` — nothing re-declared
client-side, so a server tool addition can't drift from what the model sees.
Memory tools are local `@tool` functions (`tools_memory.py`) over SQLite
(`memory.db`, gitignored): watchlist, query history, preferences.

```python
# app_agent.py — the whole loop
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=KEY,
                   timeout=TIMEOUT, max_retries=2)
tools = await load_mcp_tools(session) + build_memory_tools(db_path)
graph = create_agent(model=model, tools=tools,
                     system_prompt=STATIC_PROMPT,
                     middleware=[MemoryContext(db_path),
                                 PreferenceDefaults(db_path),
                                 GroundingGuard(model, db_path)])
result = await graph.ainvoke({"messages": [...]},
                             config={"recursion_limit": 30})
```

Middleware carries the behavior that used to be hand-rolled loop logic:

- **MemoryContext** — appends a compact durable-memory summary (watchlist,
  preferences, last 3 questions) to the system message for the current turn
  only; the static prompt never changes.
- **PreferenceDefaults** — fills `get_peer_comparison.metric` /
  `get_risk_metrics.benchmark` from saved preferences when the model omits
  them, noting `[default applied: ...]` in the tool result.
- **GroundingGuard** — every quoted figure (decimal point, thousands
  separator, currency symbol, %) must appear in that turn's server-tool
  output (0.5% relative or 2-decimal tolerance); one repair call, then an
  explicit `(unverified: N figures)` prefix if it still fails. Turns with no
  server tool call are not guarded.

### Examples

```bash
export LLM_API_KEY=sk-...        # or OPENAI_API_KEY
export LLM_MODEL=gpt-4o-mini     # optional
export LLM_BASE_URL=...          # optional, OpenAI-compatible
export LLM_TIMEOUT_SECONDS=60    # optional: a hung provider fails fast

uv run python client.py "Did Reliance beat TCS over 20 sessions as of 31 Dec 2025?"
uv run python client.py "Any doji on RELIANCE as of 2025-12-31?"
uv run python chat.py            # /watchlist, /history, quit — follow-ups keep context
```

Without a key, `client.py --demo` still proves the server works (7/7 tools,
demo peer comparison / indicators / future-date error path).

## Dashboard

Sidebar: symbol, `as_of` date, lookback. Pages: candlestick with pattern markers and support/resistance, pattern table, indicators with an honest trend badge, returns, peer ranking. Future `as_of` → error and stop; a row dated after `as_of` → explicit error, not a silent chart.

## Honest limits

- **Price data only** — no fundamentals, filings, or corporate actions.
- **Single source**: yfinance, via the server. If yfinance is wrong or
  stale, both apps repeat the same wrong numbers.
- **One agent, one flat tool list** — no sub-agents, no task decomposition.
- **The guard scans server payloads only (strict mixed-turn policy)**: in a
  turn that calls server tools and memory tools, only the server payloads
  validate quoted figures. A legitimate figure that exists solely in a
  memory payload costs one repair call — the accepted price for never
  passing recalled context off as fresh tool data.
- **Memory is context, never authority**: the watchlist/history is shown to
  the model as context; figures quoted about market data must come from tool
  output in the current turn (the GroundingGuard enforces this), not from
  recalled history. A turn with no tool call at all that still quotes
  figures is prefixed `(unverified: N figures)`. A turn where only memory
  tools ran is checked against its own memory payloads: figures that appear
  in your stored record pass (they are your history, so `grounded` stays
  NULL), figures that appear nowhere are flagged the same way.
- **Matching is unit-blind**: the guard compares numbers, not units. A
  "1570.40x" ratio matches a payload price of 1570.4 numerically, and
  "1.57 lakh" matches only a payload that itself writes "1.57 lakh", not one
  holding 157000.0. Sign is ignored ("fell -19.8%" grounds against an
  unsigned drawdown 19.8) because payloads store drawdowns unsigned.
- **Chart patterns**: the detector scans only the last `window` bars
  (default 20) — the prompt tells the model to widen `window` before
  declaring a pattern absent for older dates.
- **Live proof blocked**: no `LLM_API_KEY`/`OPENAI_API_KEY` was configured in
  this environment, so the NVIDIA endpoint's acceptance of ChatOpenAI's
  standard message shape (since `_normalise_messages` was removed) is
  unverified — run the two live transcripts in Phase L before trusting it.

## Cost of the LangChain move

- `time uv run python -c "import app_agent"` → **~2.1–2.6 s** wall (Windows,
  Git Bash, warm venv, 3 runs: 2.47 / 2.55 / 2.12 s). The cost is the import
  chain the hand-rolled loop never had — `langchain_core.messages` pulls in
  pydantic and friends; no langgraph module is imported until the graph is
  built, so the middleware hooks cost nothing at startup.
- Dependency count (uv.lock entries): **73 → 96** packages (langchain,
  langchain-core, langchain-openai, langchain-mcp-adapters, langgraph×4,
  langsmith, plus their transitive deps). The `mcp==1.12.4` pin is
  unchanged; `langchain-mcp-adapters` is capped `<0.3` because 0.3.x imports
  symbols that don't exist in mcp 1.12.4 despite declaring `mcp>=1.9.2`.

## Pinning the server

`pyproject.toml` pins the server by git tag:

```toml
[tool.uv.sources]
indian-stocks-mcp = { git = "https://github.com/Shubhampm18/indian-stocks-mcp", tag = "v0.3.0" }
```

Once the package is on PyPI, replace the tag pin with `indian-stocks-mcp>=0.3` and delete `[tool.uv.sources]`.

## License

MIT — see `LICENSE`.
