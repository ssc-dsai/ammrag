#!/usr/bin/env python3
"""
Unified launcher for AMMRAG services
Starts both FastAPI and MCP server automatically
"""

import socket
import subprocess
import sys
import time
import signal
import os
import urllib.request
import urllib.error
from pathlib import Path

# Track subprocess PIDs for cleanup
processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("✅ All services stopped")
    sys.exit(0)


def _wait_for_health(url: str, proc: subprocess.Popen, timeout: int = 60, interval: float = 0.5) -> bool:
    """Poll *url* until HTTP 200 is received, the process dies, or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # process exited; caller checks stderr
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: int = 60, interval: float = 0.5) -> bool:
    """Poll TCP *port* on localhost until it accepts connections, the process dies, or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            pass
        time.sleep(interval)
    return False


def main():
    """Start both FastAPI and MCP servers"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get the project root directory
    project_root = Path(__file__).parent

    print("\n" + "="*60)
    print("🌟 Starting AMMRAG Services")
    print("="*60)

    # Start FastAPI server
    print("🚀 Starting FastAPI service...")
    fastapi_proc = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=project_root,
    )
    processes.append(fastapi_proc)

    fastapi_port = int(os.getenv("PORT", 8000))
    health_url = f"http://localhost:{fastapi_port}/health"
    print(f"   Waiting for health check at {health_url} ...")

    if not _wait_for_health(health_url, fastapi_proc):
        if fastapi_proc.poll() is not None:
            print("❌ FastAPI failed to start (see output above)")
            return 1
        else:
            print("❌ FastAPI health check timed out (60 s) — service may still be starting")
            return 1

    print(f"✅ FastAPI service: http://localhost:{fastapi_port}")
    print(f"📚 API Documentation: http://localhost:{fastapi_port}/docs")

    # Start MCP server
    print("🚀 Starting MCP server...")
    mcp_proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        cwd=project_root,
    )
    processes.append(mcp_proc)

    mcp_port = int(os.getenv("MCP_PORT", 8001))
    print(f"   Waiting for MCP server on port {mcp_port} ...")

    if not _wait_for_port(mcp_port, mcp_proc):
        if mcp_proc.poll() is not None:
            print("❌ MCP server failed to start")
        else:
            print("⚠️  MCP server timed out — it may still be starting")

    print(f"✅ MCP server: http://localhost:{mcp_port}/mcp/")
    print("="*60)
    print("\n📝 Note: Both services are now running")
    print("   Press Ctrl+C to stop all services")
    print("="*60 + "\n")

    # Keep the script running and monitor processes
    try:
        while True:
            # Check if FastAPI is still running
            if fastapi_proc.poll() is not None:
                print("❌ FastAPI service stopped unexpectedly")
                break

            # Check if MCP server is still running
            if mcp_proc.poll() is not None:
                print("❌ MCP server stopped unexpectedly")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    # Cleanup
    signal_handler(None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
