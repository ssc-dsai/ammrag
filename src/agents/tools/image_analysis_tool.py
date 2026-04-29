import base64
import logging
from pathlib import Path
from typing import Type
from urllib.parse import urlparse
from urllib.request import urlopen

import litellm
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.llm.ollama import get_api_base
from src.core.config import settings

logger = logging.getLogger(__name__)

_VISION_MODEL = "gemma4:e4b"
_SUPPORTED_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


class _ImageAnalysisInput(BaseModel):
    uri: str = Field(description="URI of the image to analyse (file:// or http/https)")
    query: str = Field(description="The question the image should be evaluated against")


class ImageAnalysisTool(BaseTool):
    """Analyse an image from a URI using a vision LLM and return relevant findings."""

    name: str = "image_analysis"
    description: str = (
        "Analyse an image from a URI to determine if it contains information relevant "
        "to the query. Returns a description of any relevant information found, or an "
        "empty string if the image contains nothing useful."
    )
    args_schema: Type[BaseModel] = _ImageAnalysisInput

    def _run(self, uri: str, query: str) -> str:
        image_bytes = self._fetch(uri)
        if not image_bytes:
            logger.error("ImageAnalysisTool: could not fetch image at uri=%s", uri)
            return ""

        ext = Path(urlparse(uri).path).suffix.lstrip(".").lower()
        mime = f"image/{ext if ext in _SUPPORTED_EXTS else 'png'}"
        b64 = base64.b64encode(image_bytes).decode()

        response = litellm.completion(
            model=f"litellm_proxy/{_VISION_MODEL}",
            api_base=f"{get_api_base()}/v1",
            api_key="ollama",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Query: {query}\n\n"
                            "Describe any information in this image that is relevant to the query. "
                            "Be concise and specific. If nothing in the image is relevant, say so briefly."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }],
            max_tokens=512,
            num_ctx=settings.ollama_num_ctx,
        )
        content = (response.choices[0].message.content or "").strip()
        logger.info("ImageAnalysisTool: uri=%s result_length=%d", uri, len(content))
        return content

    def _fetch(self, uri: str) -> bytes | None:
        try:
            parsed = urlparse(uri)
            if parsed.scheme in ("", "file"):
                return Path(parsed.path).read_bytes()
            with urlopen(uri, timeout=15) as resp:  # noqa: S310
                return resp.read()
        except Exception as exc:
            logger.error("ImageAnalysisTool: failed to fetch image uri=%s: %s", uri, exc)
            return None
