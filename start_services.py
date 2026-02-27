#!/usr/bin/env python3
"""
Unified launcher for AMMRAG services
Starts both FastAPI and MCP server automatically
"""

import subprocess
import sys
import time
import signal
import os
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
        stdout=subprocess.DEVNULL,  # Suppress output for cleaner display
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(fastapi_proc)
    time.sleep(3)  # Give FastAPI time to start

    # Check if FastAPI started successfully
    if fastapi_proc.poll() is not None:
        stderr_output = fastapi_proc.stderr.read()
        if "Address already in use" in stderr_output:
            print("⚠️  FastAPI port already in use - a service may already be running")
            print("   Continuing anyway... (the existing service will handle requests)")
        else:
            print("❌ FastAPI failed to start")
            print(stderr_output)
            return 1

    print("✅ FastAPI service: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")

    # Start MCP server
    print("🚀 Starting MCP server...")
    mcp_proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(mcp_proc)

    print("✅ MCP server: Running on stdio transport")
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
