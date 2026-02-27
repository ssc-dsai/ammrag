#!/usr/bin/env python3
"""
Standalone MCP client for testing the AMMRAG MCP server.

Usage:
    python test_mcp_client.py --name <catalog_name> "Your question here"
    python test_mcp_client.py --catalog-id <id> "Your question here"
    python test_mcp_client.py --list-tools

Examples:
    python test_mcp_client.py --name nss "What documents are available?"
    python test_mcp_client.py --catalog-id 1 "Show me the sales data"
    python test_mcp_client.py --list-tools
"""

import argparse
import asyncio
import json
import logging
import sys
import os
import time

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Client-side logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_client")

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
PYTHON_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")


async def run(args: argparse.Namespace) -> None:
    server_params = StdioServerParameters(
        command=PYTHON_BIN,
        args=[SERVER_SCRIPT],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    logger.info("Connecting to MCP server: %s %s", PYTHON_BIN, SERVER_SCRIPT)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            logger.info("Initializing MCP session...")
            await session.initialize()
            logger.info("MCP session initialized successfully")

            if args.list_tools:
                result = await session.list_tools()
                print("Available tools:\n")
                for tool in result.tools:
                    print(f"  {tool.name}")
                    if tool.description:
                        desc = tool.description.split("\n")[0]
                        print(f"    {desc}")
                    print()
                return

            # Build arguments for query_documents
            # FastMCP expects args nested under the function parameter name
            inner: dict = {
                "question": args.question,
                "response_format": args.format,
            }
            if args.name:
                inner["name"] = args.name
            if args.catalog_id is not None:
                inner["catalog_id"] = args.catalog_id

            tool_args = {"params": inner}

            print(f"Calling query_documents with:")
            print(f"  question:    {args.question}")
            if args.name:
                print(f"  name:        {args.name}")
            if args.catalog_id is not None:
                print(f"  catalog_id:  {args.catalog_id}")
            print(f"  format:      {args.format}")
            print("-" * 60)

            logger.info("Sending query_documents call...")
            t0 = time.monotonic()
            result = await session.call_tool("query_documents", tool_args)
            elapsed = time.monotonic() - t0
            logger.info("query_documents returned in %.1fs, %d content block(s)", elapsed, len(result.content))

            if result.isError:
                logger.error("Tool returned an error")

            for block in result.content:
                if hasattr(block, "text"):
                    print(block.text)
                else:
                    print(block)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test client for the AMMRAG MCP server",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural language question to ask",
    )
    parser.add_argument(
        "--name",
        help="Catalog name to query",
    )
    parser.add_argument(
        "--catalog-id",
        type=int,
        dest="catalog_id",
        help="Catalog ID to query",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Response format (default: json)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available MCP tools and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.list_tools:
        if not args.question:
            parser.error("a question is required unless --list-tools is used")
        if not args.name and args.catalog_id is None:
            parser.error("at least one of --name or --catalog-id is required")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
