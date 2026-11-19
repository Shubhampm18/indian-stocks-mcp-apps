# indian-stocks-mcp-apps

**Two consumers of [`indian-stocks-mcp`](https://github.com/Shubhampm18/indian-stocks-mcp)** — the point-in-time NSE/BSE server. Both install the published package; neither reaches into a checkout.

```
agent/        LLM tool-calling agent + durable SQLite memory + stdio client
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

`uv run` installs the `dev` group, which includes the server package — needed for the normaliser parity test and for spawning the server from the same interpreter. Use `uv run --no-dev` only if you don't need the agent tests.

## Agent

Free-text questions → the LLM picks tools → answers with `data_through` attached. Memory is SQLite (`memory.db`, gitignored): watchlist, query history, preferences. Memory tools are `add_to_watchlist`, `remove_from_watchlist`, `list_watchlist`, `recall_history`, `set_preference`, `get_preference`.

```bash
export LLM_API_KEY=sk-...        # or OPENAI_API_KEY
export LLM_MODEL=gpt-4o-mini     # optional
export LLM_BASE_URL=...          # optional, OpenAI-compatible
export LLM_TIMEOUT_SECONDS=60    # optional: a hung provider fails fast
uv run python chat.py            # /watchlist, /history, quit
```

Without a key, `client.py --demo` still proves the server works.

## Dashboard

Sidebar: symbol, `as_of` date, lookback. Pages: candlestick with pattern markers and support/resistance, pattern table, indicators with an honest trend badge, returns, peer ranking. Future `as_of` → error and stop; a row dated after `as_of` → explicit error, not a silent chart.

## Pinning the server

`pyproject.toml` pins the server by git tag:

```toml
[tool.uv.sources]
indian-stocks-mcp = { git = "https://github.com/Shubhampm18/indian-stocks-mcp", tag = "v0.3.0" }
```

Once the package is on PyPI, replace the tag pin with `indian-stocks-mcp>=0.3` and delete `[tool.uv.sources]`.

## License

MIT — see `LICENSE`.
