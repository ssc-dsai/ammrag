from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class _PlaceholderInput(BaseModel):
    input: str = Field(description="Input data")


class PlaceholderTool(BaseTool):
    """Placeholder tool reserved for future capability — do not call."""

    name: str = "placeholder"
    description: str = "Reserved for future use. Do not call this tool."
    args_schema: Type[BaseModel] = _PlaceholderInput

    def _run(self, input: str) -> str:
        return ""
