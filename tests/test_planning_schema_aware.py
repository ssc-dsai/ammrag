"""Live test — requires Ollama."""
import json
import pytest

from src.agents.crews.planning import PlanningCrew
from src.agents.models.planning import QueryPlan
from src.agents.utils import parse_crew_output


def _kickoff(query: str, facets: dict, temporal: str = "none") -> QueryPlan:
    facets_json = json.dumps(facets) if facets else "none"
    crew_output = PlanningCrew().crew().kickoff(inputs={
        "query": query,
        "facets_json": facets_json,
        "temporal_constraint": temporal,
    })
    return parse_crew_output(
        crew_output,
        QueryPlan,
        fallback=QueryPlan(orig_query=query, needed_info=[]),
    )


def test_facet_aware_plan_generates_per_entity_needs():
    facets = {"Addresses": ["100 Saint Joseph Road", "101 Boul Roland-Therien"]}
    plan = _kickoff("Compare the kitchen facilities across all offices", facets)

    print("\n--- Facet-aware plan ---")
    for n in plan.needed_info:
        print(f"  info={n.info!r}  query={n.query!r}")

    # Should produce at least one InfoNeed referencing each address
    queries_combined = " ".join(n.query for n in plan.needed_info).lower()
    assert len(plan.needed_info) >= 2, "Expected at least one InfoNeed per office"
    assert any("saint joseph" in n.query.lower() or "roland" in n.query.lower()
               for n in plan.needed_info), "Expected address-specific InfoNeeds"


def test_no_facets_compound_generates_discovery_need():
    plan = _kickoff("Compare the kitchen facilities across all offices", facets={})

    print("\n--- No-facet plan ---")
    for n in plan.needed_info:
        print(f"  info={n.info!r}  query={n.query!r}")

    assert len(plan.needed_info) >= 1
    # First InfoNeed should be a discovery step
    first = plan.needed_info[0]
    discovery_keywords = ["list", "all", "discover", "find all", "offices", "locations"]
    assert any(kw in first.query.lower() or kw in first.info.lower()
               for kw in discovery_keywords), \
        f"First InfoNeed should be a discovery step, got: info={first.info!r} query={first.query!r}"


def test_temporal_constraint_embedded_in_queries():
    plan = _kickoff(
        "What refurbishments were done since 2022?",
        facets={},
        temporal="after: 2022-01-01",
    )

    print("\n--- Temporal plan ---")
    for n in plan.needed_info:
        print(f"  info={n.info!r}  query={n.query!r}")

    queries_combined = " ".join(n.query for n in plan.needed_info).lower()
    assert "2022" in queries_combined, \
        f"Expected '2022' in query strings, got: {queries_combined}"
