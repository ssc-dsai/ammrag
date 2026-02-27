"""Test Ollama integration"""
import asyncio
import httpx
import json

API_BASE_URL = "http://localhost:8000"


async def test_ollama():
    print("=== Testing Ollama Integration ===\n")

    # Test 1: Health Check
    print("1. Ollama Health Check")
    print("-" * 50)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/ollama/health")
        data = response.json()
        print(json.dumps(data, indent=2))
        print()

    # Test 2: List Models
    print("2. List Available Models")
    print("-" * 50)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/ollama/models")
        data = response.json()
        for model in data.get("models", []):
            print(f"  - {model['name']} ({model['size'] / 1e9:.2f} GB)")
        print()

    # Test 3: Generate Text (using correct model name)
    print("3. Generate Text")
    print("-" * 50)
    print("Prompt: What is the capital of France? Answer in one sentence.")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/ollama/generate",
            json={
                "prompt": "What is the capital of France? Answer in one sentence.",
                "temperature": 0.7,
                "max_tokens": 50
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', 'No response')}")
            print(f"Model: {data.get('model')}")
            print(f"Tokens: {data.get('eval_count')}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Detail: {response.json()}")
        print()

    # Test 4: Chat
    print("4. Chat Completion")
    print("-" * 50)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/ollama/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hello! What's your name?"}
                ],
                "temperature": 0.7
            }
        )
        if response.status_code == 200:
            data = response.json()
            message = data.get('message', {})
            print(f"Assistant: {message.get('content', 'No response')}")
            print(f"Model: {data.get('model')}")
        else:
            print(f"Error: {response.status_code}")
            print(f"Detail: {response.json()}")
        print()

    print("=== Ollama Integration Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(test_ollama())
