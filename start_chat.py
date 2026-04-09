#!/usr/bin/env python3
"""
Launcher: starts AMMRAG services (FastAPI + MCP), waits for readiness,
then opens the Gradio chat interface in a separate process.
"""

import subprocess
import sys
import time
import signal
from pathlib import Path

import httpx

project_root = Path(__file__).parent
processes: list[subprocess.Popen] = []


def _shutdown(sig, frame):
    print("\nShutting down...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    sys.exit(0)


def _wait_for_api(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError):
            pass  # Server not up yet — keep polling
        time.sleep(1)
    return False


def main() -> int:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("Starting AMMRAG services...")
    services_proc = subprocess.Popen(
        [sys.executable, "start_services.py"],
        cwd=project_root,
    )
    processes.append(services_proc)

    print("Waiting for FastAPI to be ready...", flush=True)
    if not _wait_for_api("http://localhost:8000/health"):
        print("ERROR: FastAPI did not become ready in time.")
        _shutdown(None, None)
        return 1

    print("FastAPI ready. Starting chat interface...")
    chat_proc = subprocess.Popen(
        [sys.executable, "src/web/chat.py"],
        cwd=project_root,
    )
    processes.append(chat_proc)

    # Block until the chat window exits (or Ctrl+C)
    try:
        chat_proc.wait()
    except KeyboardInterrupt:
        pass

    _shutdown(None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
