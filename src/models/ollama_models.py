"""
Pydantic models for Ollama API requests and responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


# ── Shared ───────────────────────────────────────────────────────────────────

class OllamaOptions(BaseModel):
    """Sampling options shared by generate and chat requests."""

    temperature: Optional[float] = None


# ── Generate ─────────────────────────────────────────────────────────────────

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Optional[OllamaOptions] = None
    system: Optional[str] = None
    images: Optional[List[str]] = None


class OllamaGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    response: str
    done: bool
    thinking: Optional[str] = None
    created_at: Optional[str] = None
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# ── Chat ─────────────────────────────────────────────────────────────────────

class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: bool = False
    options: Optional[OllamaOptions] = None


class OllamaChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    message: OllamaChatMessage
    done: bool
    created_at: Optional[str] = None
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


# ── Models list / show ────────────────────────────────────────────────────────

class OllamaModelDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


class OllamaModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    model: Optional[str] = None
    modified_at: Optional[str] = None
    size: Optional[int] = None
    digest: Optional[str] = None
    details: Optional[OllamaModelDetails] = None


class OllamaModelsResponse(BaseModel):
    models: List[OllamaModelInfo]


class OllamaShowResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    modelfile: Optional[str] = None
    parameters: Optional[str] = None
    template: Optional[str] = None
    details: Optional[OllamaModelDetails] = None
    model_info: Optional[Dict[str, Any]] = None


# ── Structured CSV parsing ────────────────────────────────────────────────────

class ParsedTable(BaseModel):
    """One logical table extracted from a CSV file."""

    title: str
    headers: List[str]
    rows: List[List[str]]


class ParsedTableList(BaseModel):
    """Collection of tables returned by the CSV-parsing prompt."""

    tables: List[ParsedTable]
