from __future__ import annotations

import json
import logging
import re

from src.agents.models.analysis import DataPoint
from src.agents.models.planning import InfoNeed, QueryPlan
from src.agents.models.schema import CollectionSchema, FacetValue
from src.agents.schema_service import save_schema
from src.llm.ollama import get_openai_client

logger = logging.getLogger(__name__)


def _extract_entities(datapoints: list[DataPoint], original_query: str) -> list[str]:
    """Ask the LLM to pull entity names from discovery DataPoint quotes."""
    quotes = "\n".join(
        f"- {dp.quote}" for dp in datapoints if dp.pertinent and dp.quote
    )
    if not quotes:
        return []

    prompt = (
        f"The following text was retrieved to discover entities relevant to: \"{original_query}\"\n\n"
        f"Quotes:\n{quotes}\n\n"
        "Extract the distinct entity names (e.g. office names, department names, locations). "
        "Return a JSON array of strings only. Return [] if nothing found."
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
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        entities = json.loads(raw)
        return [str(e) for e in entities if e]
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
        return []


def replan_from_discovery(
    original_query: str,
    discovery_datapoints: list[DataPoint],
    schema: CollectionSchema,
    collection_name: str,
) -> list[InfoNeed]:
    """
    Extract entities from discovery results, generate per-entity InfoNeeds,
    and write the discovered entities back to the schema cache.
    """
    logger.info("replan_from_discovery: extracting entities for '%s'", original_query[:60])
    entities = _extract_entities(discovery_datapoints, original_query)
    logger.info("Discovered entities: %s", entities)

    if not entities:
        logger.warning("No entities extracted — returning empty plan")
        return []

    # Write observed entities back to the schema cache
    if entities:
        category = _guess_category(original_query)
        existing = {fv.value for fv in schema.inferred_facets.get(category, [])}
        new_values = [FacetValue(e, "observed") for e in entities if e not in existing]
        if new_values:
            schema.inferred_facets.setdefault(category, []).extend(new_values)
            save_schema(collection_name, schema)
            logger.info("Wrote %d observed entities to schema cache under '%s'", len(new_values), category)

    # Build per-entity InfoNeeds from the original query
    needs: list[InfoNeed] = []
    # Strip the discovery part from the query to get the actual information need
    for entity in entities:
        needs.append(InfoNeed(
            info=f"{original_query} — for {entity}",
            query=f"{original_query} {entity}",
        ))

    logger.info("Generated %d InfoNeeds from %d entities", len(needs), len(entities))
    return needs


def _guess_category(query: str) -> str:
    """Heuristically pick a facet category name from the query."""
    q = query.lower()
    if any(w in q for w in ["office", "location", "site", "building", "address"]):
        return "Addresses"
    if any(w in q for w in ["department", "team", "division"]):
        return "Departments"
    if any(w in q for w in ["project", "initiative"]):
        return "Projects"
    return "entities"
