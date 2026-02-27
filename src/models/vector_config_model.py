
from dataclasses import dataclass
from typing import Optional

@dataclass
class VectorConfig:
    """Vector configuration for a catalog."""
    type: str  # "dense" or "sparse"
    model: Optional[str] = None
    embedder: Optional[str] = None
    tokenizer: Optional[str] = None
    distance: Optional[str] = None
