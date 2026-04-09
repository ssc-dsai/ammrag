from dotenv import load_dotenv

load_dotenv()

import httpx
import gradio as gr
from src.core.config import settings
from src.utils.response_formatter import format_crew_response

_API_BASE = f"http://localhost:{settings.port}"
_TIMEOUT = settings.ollama_timeout + 30.0


def _chat(message: str, history: list) -> str:
    try:
        with httpx.Client() as client:
            r = client.post(
                f"{_API_BASE}/crews/query",
                json={"question": message},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            return format_crew_response(r.json())
    except httpx.ConnectError:
        return "**Error:** Cannot connect to AMMRAG service. Is it running?"
    except httpx.HTTPStatusError as exc:
        return f"**Error:** API returned {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return f"**Error:** {exc}"


demo = gr.ChatInterface(
    fn=_chat,
    title="ammrag chat",
    chatbot=gr.Chatbot(height="75vh"),
)


def main():
    demo.launch()


if __name__ == "__main__":
    main()
