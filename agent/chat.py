"""Agentic chat for indian-stocks-mcp (free-text LLM agent + SQLite memory).

Type natural-language questions about NSE/BSE stocks; the LLM picks the
MCP tools, reasons across calls, and answers. The agent remembers your
watchlist and past queries across restarts (SQLite via memory.py).

Commands:
  /watchlist   show your durable watchlist
  /history     show recent past questions
  quit / exit  leave

Requires LLM_API_KEY / OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import memory as mem
from agent import Agent, make_client
from client import server_params
from mcp import ClientSession
from mcp.client.stdio import stdio_client


async def main() -> int:
    try:
        client = make_client()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    params = server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = Agent(session, client)

            watchlist = mem.list_watchlist()
            recent = mem.recent_queries(3)
            print("Stock Agent (MCP + LLM + memory). Type 'quit' to exit.")
            print(f"Watchlist on file: {watchlist or '[]'}")
            if recent:
                print("Recent topics:")
                for q in recent:
                    print(f"  - {q['question']}")
            print()

            while True:
                question = input("you: ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit"):
                    break
                if question.startswith("/"):
                    cmd = question.lower()
                    if cmd == "/watchlist":
                        w = mem.list_watchlist()
                        print(f"Watchlist: {w or '[]'}")
                    elif cmd == "/history":
                        for item in mem.recent_queries(10):
                            print(f"  {item['ts']}  {item['question']}")
                    else:
                        print("Unknown command. Try /watchlist or /history.")
                    continue
                try:
                    answer = await agent.ask(question)
                except Exception as exc:  # keep the loop alive on LLM errors
                    print(f"Error: {exc}")
                    continue
                print(f"agent: {answer}\n")
    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(__import__("anyio").run(main))
