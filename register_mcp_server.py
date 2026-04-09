#!/usr/bin/env python3
"""
Register the AMMRAG HTTP MCP server with the canchat-v2 application.

Usage:
    python register_mcp_server.py --token <admin-token>
    python register_mcp_server.py --mcp-url http://172.22.0.3:8001/mcp/ --token <token>
    python register_mcp_server.py --dry-run
"""

import argparse
import json
import os
import sys
try:
    import requests
except ImportError:
    sys.exit("Error: 'requests' is not installed. Run: pip install requests")


DEFAULT_MCP_URL = "http://172.22.0.3:8001/mcp/"


def load_mcp_url() -> str:
    return DEFAULT_MCP_URL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register the AMMRAG HTTP MCP server with canchat-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("CANCHAT_SERVER_URL", "http://172.17.0.3:8080"),
        help="Base URL of the canchat-v2 app (default: http://172.17.0.3:8080)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CANCHAT_ADMIN_TOKEN", ""),
        help="Admin bearer token (or set CANCHAT_ADMIN_TOKEN env var)",
    )
    parser.add_argument(
        "--mcp-url",
        default=load_mcp_url(),
        help="URL of the AMMRAG MCP server (default from mcp_config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request without sending it",
    )
    return parser.parse_args()


def register(base_url: str, token: str, mcp_url: str, dry_run: bool):
    endpoint = f"{base_url.rstrip('/')}/api/v1/mcp/config/update"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    current_urls = []
    current_api_configs = {}
    if not dry_run:
        cfg = requests.get(
            f"{base_url.rstrip('/')}/api/v1/mcp/config",
            headers=headers,
            timeout=15,
        )
        if cfg.ok:
            data = cfg.json()
            current_urls = data.get("MCP_BASE_URLS", []) or []
            current_api_configs = data.get("MCP_API_CONFIGS", {}) or {}

    if mcp_url not in current_urls:
        current_urls.append(mcp_url)

    payload = {
        "ENABLE_MCP_API": True,
        "MCP_BASE_URLS": current_urls,
        "MCP_API_CONFIGS": current_api_configs,
    }

    print(f"\n{'='*50}")
    print("Registering AMMRAG HTTP MCP server")
    print(f"{'='*50}")
    print(f"  Endpoint : {endpoint}")
    print(f"  MCP URL  : {mcp_url}")

    if dry_run:
        print("\n[dry-run] Payload:")
        print(json.dumps(payload, indent=2))
        return

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=15)
    if not resp.ok:
        print(f"\nError {resp.status_code}: {resp.text}")
        sys.exit(1)

    urls = resp.json().get("MCP_BASE_URLS", [])
    print(f"\n  Registered. MCP_BASE_URLS is now: {urls}")

    # Verify
    print(f"\n{'='*50}")
    print("Verifying MCP connection")
    print(f"{'='*50}")
    verify = requests.post(
        f"{base_url.rstrip('/')}/api/v1/mcp/verify",
        headers=headers,
        json={"url": mcp_url},
        timeout=15,
    )
    if not verify.ok:
        print(f"  Warning: verification returned {verify.status_code}: {verify.text}")
        return
    data = verify.json()
    print(f"  Status      : {data.get('status', 'unknown')}")
    print(f"  Tools found : {data.get('tools_count', '?')}")
    if data.get("status") != "connected":
        print(f"  Warning: not connected. Is AMMRAG running at {mcp_url}?")


def main():
    args = parse_args()
    if not args.token and not args.dry_run:
        sys.exit(
            "Error: admin token required.\n"
            "  Set CANCHAT_ADMIN_TOKEN env var or pass --token <token>"
        )
    register(args.server_url, args.token, args.mcp_url, args.dry_run)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
