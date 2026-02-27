"""
Pydantic schemas for Ollama API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="Content of the message")


class GenerateRequest(BaseModel):
    """Request schema for text generation"""
    prompt: str = Field(..., description="The prompt to generate from")
    system: Optional[str] = Field(None, description="System message to set context")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")


class ChatRequest(BaseModel):
    """Request schema for chat completion"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")


class GenerateResponse(BaseModel):
    """Response schema for text generation"""
    model: str = Field(..., description="Model used for generation")
    response: str = Field(..., description="Generated text")
    done: bool = Field(..., description="Whether generation is complete")
    context: Optional[List[int]] = Field(None, description="Context tokens")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Model load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in prompt")
    eval_count: Optional[int] = Field(None, description="Number of tokens in response")
    eval_duration: Optional[int] = Field(None, description="Generation duration in nanoseconds")


class ChatResponse(BaseModel):
    """Response schema for chat completion"""
    model: str = Field(..., description="Model used for chat")
    message: ChatMessage = Field(..., description="Assistant's response message")
    done: bool = Field(..., description="Whether chat is complete")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Model load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in prompt")
    eval_count: Optional[int] = Field(None, description="Number of tokens in response")
    eval_duration: Optional[int] = Field(None, description="Generation duration in nanoseconds")


class ModelInfo(BaseModel):
    """Information about an Ollama model"""
    name: str = Field(..., description="Model name")
    modified_at: str = Field(..., description="Last modification timestamp")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest/hash")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional model details")


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: List[ModelInfo] = Field(..., description="List of available models")


class OllamaHealthResponse(BaseModel):
    """Health check response for Ollama service"""
    status: str = Field(..., description="Service status")
    ollama_host: str = Field(..., description="Configured Ollama host")
    ollama_model: str = Field(..., description="Configured model")
    available: bool = Field(..., description="Whether Ollama is reachable")
    models_count: Optional[int] = Field(None, description="Number of available models")
