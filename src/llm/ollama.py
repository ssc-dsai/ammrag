import litellm
from openai import OpenAI
from crewai import LLM

from src.core.config import settings


# def _register_model() -> None:
#     """Register the custom Ollama model with LiteLLM so cost/context lookups don't warn."""
#     model_key = f"openai/{settings.ollama_model}"
#     litellm.register_model({
#         model_key: {
#             "max_tokens": settings.ollama_max_tokens,
#             "max_input_tokens": settings.ollama_max_tokens,
#             "input_cost_per_token": 0.0,
#             "output_cost_per_token": 0.0,
#             "litellm_provider": "openai",
#             "mode": "chat",
#         }
#     })


# _register_model()


def get_model() -> str:
    return f"openai/{settings.ollama_model}"


def get_api_base() -> str:
    return settings.ollama_host.rstrip("/")


def get_temperature() -> float:
    return settings.ollama_temperature


def get_openai_client() -> OpenAI:
    return OpenAI(base_url=f"{get_api_base()}/v1", api_key="ollama")

def get_llm(model: str | None = None) -> LLM:
    model_name = f"openai/{model}" if model else get_model()
    return LLM(
        model=model_name,
        provider="openai",
        base_url=f"{get_api_base()}/v1",
        api_key="ollama",
        temperature=get_temperature(),
        extra_body={"think": False, "num_ctx": settings.ollama_num_ctx},
    )

llm = get_llm()
