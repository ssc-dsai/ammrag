"""
Ollama service for LLM interactions

This service handles communication with the Ollama API for generating
text completions and chat responses.
"""

import logging
import time
from typing import List, Dict, Optional, AsyncGenerator
import httpx
from fastapi import HTTPException

from src.core.config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Service for interacting with Ollama LLM

    Handles both completion and chat API calls to Ollama server
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: int = None
    ):
        """
        Initialize Ollama service

        Args:
            base_url: Ollama server URL (defaults to settings.ollama_host)
            model: Model name (defaults to settings.ollama_model)
            timeout: Request timeout in seconds (defaults to settings.ollama_timeout)
        """
        self.base_url = (base_url or settings.ollama_host).rstrip('/')
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Generate a completion from a prompt

        Args:
            prompt: The prompt to send to the model
            system: Optional system message to set context
            images: Optional list of base64-encoded images for vision models
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Response dictionary with generated text

        Raises:
            HTTPException: If Ollama request fails
        """
        is_vision = bool(images)
        model = settings.ollama_vision_model if is_vision else self.model
        request_timeout = settings.ollama_vision_timeout if is_vision else self.timeout

        logger.info(
            "Ollama generate request: model=%s, vision=%s, timeout=%ds, temperature=%.2f",
            model, is_vision, request_timeout, temperature,
        )
        if is_vision:
            logger.info(
                "Vision request: %d image(s), prompt length=%d chars",
                len(images), len(prompt),
            )

        start = time.monotonic()
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                }
            }

            if system:
                payload["system"] = system

            if images:
                payload["images"] = images

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

            elapsed = time.monotonic() - start
            resp_text = result.get("response", "")
            logger.info(
                "Ollama generate completed: model=%s, elapsed=%.1fs, response_length=%d chars",
                model, elapsed, len(resp_text),
            )
            return result

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama server at %s", self.base_url)
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama server at {self.base_url}. "
                       f"Please ensure Ollama is running."
            )
        except httpx.TimeoutException:
            elapsed = time.monotonic() - start
            logger.error(
                "Ollama request timed out: model=%s, timeout=%ds, elapsed=%.1fs",
                model, request_timeout, elapsed,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request to Ollama timed out after {request_timeout}s (model={model})"
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "Ollama API error: model=%s, status=%d, response=%s",
                model, e.response.status_code, e.response.text,
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            logger.exception("Unexpected error during Ollama generate: model=%s", model)
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error communicating with Ollama: {str(e)}"
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Send a chat completion request

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Response dictionary with chat completion

        Raises:
            HTTPException: If Ollama request fails
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                }
            }

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                return response.json()

        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama server at {self.base_url}. "
                       f"Please ensure Ollama is running."
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Request to Ollama timed out after {self.timeout} seconds"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error communicating with Ollama: {str(e)}"
            )

    async def list_models(self) -> Dict:
        """
        List available models on the Ollama server

        Returns:
            Dictionary containing list of available models

        Raises:
            HTTPException: If Ollama request fails
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json()

        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama server at {self.base_url}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error listing models: {str(e)}"
            )

    async def show_model_info(self, model_name: Optional[str] = None) -> Dict:
        """
        Get information about a specific model

        Args:
            model_name: Name of the model (defaults to configured model)

        Returns:
            Model information dictionary

        Raises:
            HTTPException: If Ollama request fails
        """
        try:
            model = model_name or self.model
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model}
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting model info: {str(e)}"
            )


# Global service instance
ollama_service = OllamaService()
