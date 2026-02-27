"""Test Ollama MCP tools"""
import asyncio
import json
from mcp_server import (
    ollama_generate,
    ollama_chat,
    check_ollama_health,
    OllamaGenerateInput,
    OllamaChatInput,
    OllamaChatMessage,
    QueryFilesInput,
    ResponseFormat
)


async def main():
    print("=== Testing Ollama MCP Tools ===\n")

    # Test 1: Ollama Health via MCP
    print("1. MCP Ollama Health Check")
    print("-" * 50)
    health_result = await check_ollama_health(
        QueryFilesInput(response_format=ResponseFormat.MARKDOWN)
    )
    print(health_result)
    print("\n")

    # Test 2: Generate Text via MCP
    print("2. MCP Text Generation")
    print("-" * 50)
    generate_result = await ollama_generate(
        OllamaGenerateInput(
            prompt="Explain what FastAPI is in one sentence.",
            temperature=0.7,
            response_format=ResponseFormat.MARKDOWN
        )
    )
    print(generate_result)
    print("\n")

    # Test 3: Chat via MCP
    print("3. MCP Chat")
    print("-" * 50)
    chat_result = await ollama_chat(
        OllamaChatInput(
            messages=[
                OllamaChatMessage(role="system", content="You are a helpful Python programming assistant."),
                OllamaChatMessage(role="user", content="What is a list comprehension? Answer briefly.")
            ],
            temperature=0.7,
            response_format=ResponseFormat.MARKDOWN
        )
    )
    print(chat_result)
    print("\n")

    # Test 4: JSON Response Format
    print("4. JSON Response Format")
    print("-" * 50)
    json_result = await ollama_generate(
        OllamaGenerateInput(
            prompt="What is 2+2?",
            temperature=0.3,
            response_format=ResponseFormat.JSON
        )
    )
    data = json.loads(json_result)
    print(f"Response: {data.get('response')}")
    print(f"Model: {data.get('model')}")
    print(f"Tokens: {data.get('eval_count')}")
    print("\n")

    print("=== All MCP Ollama Tests Complete! ===")


if __name__ == "__main__":
    asyncio.run(main())
