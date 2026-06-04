from __future__ import annotations

import json
import logging
import re

from src.agents.models.intent import QueryIntent, TemporalConstraint, VALID_TAGS
from src.llm.ollama import get_openai_client

logger = logging.getLogger(__name__)

_DIRECT_LOOKUP_RE = re.compile(
    r"^\s*(what\s+is|what\s+are\s+the|define|definition\s+of|what\s+does\s+\S+\s+stand\s+for|who\s+is)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> tuple[QueryIntent, TemporalConstraint | None]:
    """Classify query intent and extract any temporal constraint."""
    prompt = (
        "Classify the intent of the following question. Use ONLY the tags defined below.\n\n"
        "Tag definitions:\n"
        "  navigational — asks what something IS, its definition, meaning, or identity. "
        "Single fact lookup. Examples: 'What is WBA?', 'Define asbestos', 'Who is the site manager?'\n"
        "  spatial — asks about a physical location or position. "
        "Examples: 'Where is the kitchen?', 'What floor is reception on?'\n"
        "  visual — asks about appearance or imagery. "
        "Examples: 'Show me the floorplan', 'What does the entrance look like?'\n"
        "  structured — asks for counts, totals, comparisons using numbers. "
        "Examples: 'How many employees?', 'Which has the most exits?', 'Total invoice value'\n"
        "  temporal — has a date-based filter. "
        "Examples: 'since 2022', 'before March', 'Q3 2024'\n"
        "  compound — requires looking up multiple entities or doing multiple retrievals. "
        "Examples: 'Compare X across all offices', 'Find Y in each department'\n\n"
        "Return a JSON object with:\n"
        '  "intent": list of applicable tags (use the minimum number needed)\n'
        '  "temporal": null, or {"after": "YYYY-MM-DD", "before": "YYYY-MM-DD", "anchor": "YYYY-MM-DD"} '
        "(include only the keys that apply)\n"
        "Return raw JSON only — no markdown fences.\n\n"
        f'Question: "{query}"'
    )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gemma4:e4b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[-2].lstrip("json").strip()
        # Extract the first JSON object if extra text is present
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        data = json.loads(raw)

        intent: QueryIntent = [
            tag for tag in (data.get("intent") or [])
            if tag in VALID_TAGS
        ] or ["navigational"]

        temporal: TemporalConstraint | None = None
        tc = data.get("temporal")
        if tc and isinstance(tc, dict):
            temporal = TemporalConstraint(
                after=tc.get("after"),
                before=tc.get("before"),
                anchor=tc.get("anchor"),
            )

        logger.info("Classified '%s' → intent=%s temporal=%s", query[:60], intent, temporal)
        return intent, temporal

    except Exception as exc:
        logger.warning("Classifier failed for query '%s': %s — defaulting to navigational", query[:60], exc)
        return ["navigational"], None


def is_simple_query(intent: QueryIntent, query: str) -> bool:
    """True only for single-intent navigational direct-lookup queries."""
    if intent != ["navigational"]:
        return False
    return bool(_DIRECT_LOOKUP_RE.match(query.strip()))
