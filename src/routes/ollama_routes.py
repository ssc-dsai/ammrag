"""
Ollama routes for LLM interactions
"""

from fastapi import APIRouter, HTTPException
from typing import Dict

from src.schemas.ollama_schemas import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    ModelListResponse,
    OllamaHealthResponse
)
from src.services.ollama_service import ollama_service
from src.core.config import settings

router = APIRouter(
    prefix="/ollama",
    tags=["ollama"],
    responses={
        503: {"description": "Ollama service unavailable"},
        504: {"description": "Request timeout"}
    }
)


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate text from prompt",
    description="Generate text completion from a prompt using Ollama"
)
async def generate_text(request: GenerateRequest) -> Dict:
    """
    Generate text completion from a prompt

    This endpoint sends a prompt to Ollama and returns the generated text.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        GenerateResponse with generated text and metadata

    Raises:
        HTTPException: If Ollama service is unavailable or request fails
    """
    result = await ollama_service.generate(
        prompt=request.prompt,
        system=request.system,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return result


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat completion",
    description="Send chat messages and get a completion from Ollama"
)
async def chat_completion(request: ChatRequest) -> Dict:
    """
    Generate a chat completion

    This endpoint handles multi-turn conversations with Ollama.

    Args:
        request: ChatRequest with message history and generation parameters

    Returns:
        ChatResponse with assistant's reply and metadata

    Raises:
        HTTPException: If Ollama service is unavailable or request fails
    """
    # Convert Pydantic models to dicts for the service
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    result = await ollama_service.chat(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return result


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    description="Get list of models available on the Ollama server"
)
async def list_models() -> Dict:
    """
    List all available Ollama models

    Returns:
        ModelListResponse with list of available models

    Raises:
        HTTPException: If Ollama service is unavailable
    """
    return await ollama_service.list_models()


@router.get(
    "/models/{model_name}",
    summary="Get model information",
    description="Get detailed information about a specific model"
)
async def get_model_info(model_name: str) -> Dict:
    """
    Get information about a specific model

    Args:
        model_name: Name of the model to query

    Returns:
        Model information dictionary

    Raises:
        HTTPException: If model not found or Ollama unavailable
    """
    return await ollama_service.show_model_info(model_name)


@router.get(
    "/health",
    response_model=OllamaHealthResponse,
    summary="Check Ollama service health",
    description="Check if Ollama service is reachable and list available models"
)
async def check_ollama_health() -> OllamaHealthResponse:
    """
    Check Ollama service health and availability

    Returns:
        OllamaHealthResponse with service status and configuration

    Note:
        This endpoint will always return 200 OK with an 'available' field
        indicating whether Ollama is reachable
    """
    try:
        models = await ollama_service.list_models()
        models_count = len(models.get("models", []))
        available = True
    except Exception:
        models_count = None
        available = False

    return OllamaHealthResponse(
        status="healthy" if available else "degraded",
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
        available=available,
        models_count=models_count
    )
