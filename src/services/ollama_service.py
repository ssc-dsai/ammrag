"""
Ollama service for LLM interactions

This service handles communication with the Ollama API for generating
text completions and chat responses.
"""

import csv
import io
import logging
import time
from pathlib import Path
from typing import List, Optional

import httpx
import ollama
import yaml
from fastapi import HTTPException

from src.core.config import settings
from src.models.ollama_models import OllamaGenerateResponse, ParsedTable, ParsedTableList


# ── CSV pre-processing helpers ────────────────────────────────────────────────

def _collapse_newlines(value: str) -> str:
    return " ".join(value.split())


def _parse_csv(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text))
    return [[_collapse_newlines(cell) for cell in row] for row in reader]


def _split_sections(rows: list[list[str]]) -> list[list[list[str]]]:
    """Split rows into sections on blank rows."""
    sections: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            if current:
                sections.append(current)
                current = []
        else:
            current.append(row)
    if current:
        sections.append(current)
    return sections


def _drop_empty_cols(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    num_cols = max(len(r) for r in rows)
    padded = [r + [""] * (num_cols - len(r)) for r in rows]
    keep = [c for c in range(num_cols) if any(padded[r][c].strip() for r in range(len(padded)))]
    return [[row[c] for c in keep] for row in padded]


def _forward_fill_col0(rows: list[list[str]]) -> list[list[str]]:
    """Forward-fill sparse section labels in column 0."""
    if len(rows) < 2:
        return rows
    result = [rows[0]]
    last = rows[0][0].strip() if rows[0] else ""
    for row in rows[1:]:
        val = row[0].strip() if row else ""
        if val:
            last = val
        else:
            row = [last] + row[1:]
        result.append(row)
    return result


def _normalize_grouped_header(rows: list[list[str]]) -> list[list[str]]:
    """Rename col-0/col-1 to 'Section'/'Field' when the table has ≥3 columns."""
    if not rows or len(rows[0]) < 2:
        return rows
    header = list(rows[0])
    header[0] = "Section"
    header[1] = "Field"
    return [header] + rows[1:]


def _rows_to_csv_text(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, quoting=csv.QUOTE_MINIMAL).writerows(rows)
    return buf.getvalue().strip()


def preprocess_csv(csv_content: str) -> str:
    """
    Clean a raw CSV export and return labelled section blocks ready for the LLM.

    Steps:
    - Collapse multiline cells to a single line
    - Drop trailing blank rows
    - Split on blank rows into sections
    - Per section: drop wholly-empty columns, forward-fill the grouping column,
      and rename col-0/col-1 to Section/Field for multi-column tables
    """
    rows = _parse_csv(csv_content)
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()

    parts: list[str] = []
    for i, section in enumerate(_split_sections(rows), 1):
        cleaned = _drop_empty_cols(section)
        if cleaned and max(len(r) for r in cleaned) >= 3:
            cleaned = _forward_fill_col0(cleaned)
            cleaned = _normalize_grouped_header(cleaned)
        parts.append(f"--- Section {i} ---\n{_rows_to_csv_text(cleaned)}")

    return "\n\n".join(parts)

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for interacting with Ollama LLM."""

    def __init__(
        self,
        base_url: str = settings.ollama_host.rstrip('/'),
        model: str = settings.ollama_model,
        timeout: int = settings.ollama_timeout,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.retries = settings.ollama_retries

    def _client(self, timeout: int) -> ollama.AsyncClient:
        return ollama.AsyncClient(host=self.base_url, timeout=timeout)

    async def generate(
        self,
        prompt: str,
        system: str = "",
        images: Optional[List[str]] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        model: str = settings.ollama_model,
        timeout: Optional[int] = None
    ) -> OllamaGenerateResponse:

        model = model or self.model
        request_timeout = timeout or self.timeout

        logger.info(
            "Ollama generate request: model=%s, timeout=%ds, temperature=%.2f",
            model, request_timeout, temperature,
        )

        options = ollama.Options(
            temperature=temperature,
            # num_predict=max_tokens or settings.ollama_num_ctx,
        )

        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            start = time.monotonic()
            try:
                chunk: ollama.GenerateResponse = await self._client(request_timeout).generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    images=images,
                    options=options,
                    stream=False
                ) # type: ignore

                result = OllamaGenerateResponse.model_validate({
                    **chunk.model_dump(),
                    "model": model,
                })

                elapsed = time.monotonic() - start
                logger.info(
                    "Ollama generate completed: model=%s, elapsed=%.1fs, response_length=%d chars",
                    model, elapsed, len(result.response),
                )
                return result

            except (httpx.ConnectError, ConnectionError):
                logger.error("Cannot connect to Ollama server at %s", self.base_url)
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to Ollama server at {self.base_url}. "
                           f"Please ensure Ollama is running.",
                )
            except httpx.TimeoutException:
                elapsed = time.monotonic() - start
                if attempt <= self.retries:
                    logger.warning(
                        "Ollama request timed out (attempt %d/%d): model=%s, timeout=%ds, elapsed=%.1fs — retrying",
                        attempt, attempts, model, request_timeout, elapsed,
                    )
                    continue
                logger.error(
                    "Ollama request timed out after %d attempt(s): model=%s, timeout=%ds, elapsed=%.1fs",
                    attempts, model, request_timeout, elapsed,
                )
                raise HTTPException(
                    status_code=504,
                    detail=f"Request to Ollama timed out after {request_timeout}s and {attempts} attempt(s) (model={model})",
                )
            except ollama.ResponseError as e:
                logger.error(
                    "Ollama API error: model=%s, status=%d, error=%s",
                    model, e.status_code, e.error,
                )
                raise HTTPException(
                    status_code=e.status_code if e.status_code > 0 else 500,
                    detail=f"Ollama API error: {e.error}",
                )
            except Exception as e:
                logger.exception("Unexpected error during Ollama generate: model=%s", model)
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error communicating with Ollama: {str(e)}",
                )

    async def parse_csv_tables(
        self,
        csv_content: str,
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[ParsedTable]:
        """
        Pre-process a raw CSV string and use the LLM to split it into clean,
        structured tables.

        Returns a list of ParsedTable objects, each with a title, headers, and
        rows ready for insertion into a database.
        """
        model = model or self.model
        request_timeout = timeout or self.timeout

        cleaned = preprocess_csv(csv_content)
        prompt_cfg = get_prompt_config("parse_csv_tables", csv_content=cleaned)

        request_timeout = prompt_cfg["timeout"] or timeout or self.timeout

        logger.info(
            "Ollama parse_csv_tables: model=%s, timeout=%ds",
            model, request_timeout,
        )

        try:
            response = await self._client(request_timeout).chat(
                model=model,
                messages=[{"role": "user", "content": prompt_cfg["prompt"]}],
                format=ParsedTableList.model_json_schema(),
                keep_alive=0,
            )
        except (httpx.ConnectError, ConnectionError):
            logger.error("Cannot connect to Ollama server at %s", self.base_url)
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama server at {self.base_url}.",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Ollama timed out after {request_timeout}s (model={model})",
            )
        except ollama.ResponseError as e:
            raise HTTPException(
                status_code=e.status_code if e.status_code > 0 else 500,
                detail=f"Ollama API error: {e.error}",
            )

        result = ParsedTableList.model_validate_json(response.message.content)
        logger.info("parse_csv_tables: %d table(s) extracted", len(result.tables))
        return result.tables


def _load_prompts() -> dict:
    path = Path(__file__).parent.parent.parent / "config" / "prompts" / "ollama.yml"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompts", {})


_PROMPTS: dict = _load_prompts()


def get_prompt(name: str, **kwargs) -> str:
    """Return the named prompt template with variables substituted."""
    template = _PROMPTS[name]["template"]
    return template.format(**kwargs) if kwargs else template


def get_prompt_config(name: str, **kwargs) -> dict:
    """Return a dict of generate() kwargs for the named prompt.

    Includes rendered ``prompt``, ``temperature``, ``max_tokens`` (if set), and
    ``model`` when explicitly specified in the YAML (e.g. ``model: llava``).
    Omitting ``model`` or setting it to ``"default"`` uses the service default.
    """
    cfg = _PROMPTS[name]
    result: dict = {
        "prompt": cfg["template"].format(**kwargs) if kwargs else cfg["template"],
    }
    if "temperature" in cfg:
        result["temperature"] = cfg["temperature"]
    if "max_tokens" in cfg:
        result["max_tokens"] = cfg["max_tokens"]
    if "timeout" in cfg:
        result["timeout"] = cfg["timeout"]
    model = cfg.get("model")
    if model and model != "default":
        result["model"] = model
    return result

ollama_service = OllamaService()
