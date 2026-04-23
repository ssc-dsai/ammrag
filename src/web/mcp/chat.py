"""
MCP Chat Interface

A standalone Gradio chat interface that communicates with the AMMRAG MCP server.
Uses Ollama directly for LLM inference with MCP tools for document retrieval.
No AMMRAG source code is imported — configuration is read from environment variables.
"""

import os
import json
import asyncio
import sys
from typing import Any

import httpx
import gradio as gr
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

# ─── Configuration (mirrors AMMRAG .env variables) ───────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "gpt-oss:latest"
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
MCP_URL = os.getenv("MCP_URL", f"http://{MCP_HOST}:{MCP_PORT}/mcp/")

# ─── MCP helpers ─────────────────────────────────────────────────────────────


async def _list_tools() -> list[dict]:
    """Return available MCP tools in Ollama tool-call format."""
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in result.tools
            ]


async def _call_tool(name: str, arguments: dict) -> str:
    """Execute a tool on the MCP server and return its text content."""
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return (
                "\n".join(c.text for c in result.content if hasattr(c, "text"))
                or "No result returned."
            )


# ─── Chat logic ──────────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a document assistant. "
    "Always call query_documents before answering any question about documents. "
    "Use synthesis=true when the user needs a formatted, cited answer. "
    "Use synthesis=false when you need raw data points to reason over before answering. "
    "Never answer from memory or prior knowledge alone. "
    "If a result is used, include the link to the source in markdown. "
    "If an image is included, render the image using markdown."
)


async def _chat(message: str, history: list) -> str:
    """Handle a chat turn: call Ollama with MCP tools, execute any tool calls."""
    # Build message history.
    # Gradio 5+ passes history as a flat list of {"role": ..., "content": ...} dicts.
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item.get("content") or ""})
        else:
            # Fallback for old Gradio tuple format
            user_msg, assistant_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Fetch MCP tools; proceed without tools if the server is unreachable
    try:
        tools = await _list_tools()
    except Exception:
        tools = []

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        # ── First Ollama call ──────────────────────────────────────────────
        payload: dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": OLLAMA_TEMPERATURE},
        }
        if tools:
            payload["tools"] = tools

        print(f"\n>>> Ollama request:\n{json.dumps(payload, indent=2)}", file=sys.stderr, flush=True)

        try:
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.ConnectError:
            return f"**Error:** Cannot connect to Ollama at `{OLLAMA_HOST}`."
        except httpx.HTTPStatusError as exc:
            return f"**Error:** Ollama returned {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"**Error:** {exc}"

        print(f"\n<<< Ollama response:\n{json.dumps(r.json(), indent=2)}", file=sys.stderr, flush=True)

        msg = r.json().get("message", {})
        tool_calls: list[dict] = msg.get("tool_calls") or []

        # No tool calls → return the direct reply
        if not tool_calls:
            return msg.get("content") or "No response generated."

        # ── Tool call loop — keep executing until model returns plain content ─
        MAX_ROUNDS = 5
        for round_num in range(MAX_ROUNDS):
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name: str = fn.get("name", "")
                raw_args = fn.get("arguments") or {}
                # Ollama sometimes returns arguments as a JSON string rather than a dict
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                tool_args: dict = raw_args

                print(f"\n>>> Tool call: {tool_name}\n{json.dumps(tool_args, indent=2)}", file=sys.stderr, flush=True)

                try:
                    tool_result = await _call_tool(tool_name, tool_args)
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"

                print(f"\n<<< Tool result: {tool_name}\n{tool_result}", file=sys.stderr, flush=True)

                messages.append({"role": "tool", "content": tool_result})

            next_payload: dict[str, Any] = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": OLLAMA_TEMPERATURE},
                "tools": tools,
            }

            print(f"\n>>> Ollama request (round {round_num + 2}):\n{json.dumps(next_payload, indent=2)}", file=sys.stderr, flush=True)

            try:
                r2 = await client.post(f"{OLLAMA_HOST}/api/chat", json=next_payload)
                r2.raise_for_status()
            except Exception as exc:
                return f"**Error on synthesis:** {exc}"

            print(f"\n<<< Ollama response (round {round_num + 2}):\n{json.dumps(r2.json(), indent=2)}", file=sys.stderr, flush=True)

            msg = r2.json().get("message", {})
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                return msg.get("content") or "No response generated."

        return msg.get("content") or "No response generated."


# ─── Gradio UI ───────────────────────────────────────────────────────────────

demo = gr.ChatInterface(
    fn=_chat,
    title="ammrag · MCP chat",
    description=(
        f"Chat powered by **{OLLAMA_MODEL}** via Ollama "
        f"with document retrieval from the AMMRAG MCP server (`{MCP_URL}`)."
    ),
    chatbot=gr.Chatbot(height="75vh", type="messages"),
)


def main():
    print(f"MCP server : {MCP_URL}")
    print(f"Ollama     : {OLLAMA_HOST}  model={OLLAMA_MODEL}")
    demo.launch()


if __name__ == "__main__":
    main()
