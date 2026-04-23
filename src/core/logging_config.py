"""
Central logging configuration for AMMRAG.

Call setup_logging() once at application startup (main.py, mcp_server.py).
All other modules simply use logging.getLogger(__name__).
"""

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path("logs")

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Reduced to WARNING level
_QUIET_AT_WARNING = [
    "httpx",
    "httpcore",
    "asyncio",
    "urllib3",
    "sentence_transformers",
    "transformers",
    "torch",
    "llama_index",
    "fastembed",
    "openai",
    "anthropic",
    "PIL",
    "filelock",
    "huggingface_hub",
    # crewai internals — agent traces still show via crewai.crew / crewai.agent
    "crewai.utilities",
    "crewai.telemetry",
    "crewai.memory",
]

# Completely silenced (CRITICAL only — effectively no output)
_SILENT = [
    "litellm",
    "litellm.utils",
    "litellm.main",
    "LiteLLM",
]


def setup_logging(level: int = logging.INFO, mcp_mode: bool = False) -> None:
    """Configure the root logger.

    Args:
        level: Minimum log level for application code (default INFO).
        mcp_mode: When True, logs only to stderr (stdout must stay clean
                  for the MCP JSON-RPC stdio transport).
    """
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # root captures everything; handlers filter
    root.handlers.clear()          # remove any handlers added by basicConfig

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ──────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr if mcp_mode else sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── Rolling file handler (not in MCP mode — file I/O could block stdio) ──
    if not mcp_mode:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "ammrag.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # ── Suppress noisy third-party libraries ─────────────────────────────────
    for name in _QUIET_AT_WARNING:
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in _SILENT:
        logging.getLogger(name).setLevel(logging.CRITICAL)
