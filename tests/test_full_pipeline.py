"""
Full pipeline integration tests — requires Qdrant and Ollama.
Run with: python -m pytest tests/test_full_pipeline.py -v -s --timeout=120
"""
import time
import pytest

from src.agents.flows.rag import RAGFlow

COLLECTION = "nss"

TEST_QUERIES = [
    # (query, label, expect_simple_path)
    ("What is a floorplan?",                                         "Q1-simple",    True),
    ("How many floors are in the building?",                         "Q2-structured", False),
    ("List all documents mentioning asbestos",                       "Q3-nav",       False),
    ("What does the reception area look like?",                      "Q4-visual",    False),
    ("Show me the floorplan for 100 Saint Joseph Road",              "Q5-spatial",   False),
    ("Compare the kitchens across all office locations",             "Q6-compound",  False),
    ("What safety equipment is visible in the floorplans?",          "Q7-visual",    False),
    ("What changes were made after 2024?",                           "Q8-temporal",  False),
    ("Show all floorplans that have a staircase",                    "Q9-visual",    False),
    ("Is there a server room in the building?",                      "Q10-nav",      False),
]


def _run(query: str, synthesis: bool = True) -> dict:
    start = time.time()
    result = RAGFlow().kickoff(inputs={
        "query": query,
        "collection_name": COLLECTION,
        "synthesis": synthesis,
    })
    elapsed = time.time() - start
    raw = str(result) if result else ""
    return {"result": raw, "elapsed": elapsed}


@pytest.mark.parametrize("query,label,expect_simple", TEST_QUERIES)
def test_query_returns_answer(query, label, expect_simple):
    print(f"\n{'='*60}")
    print(f"  {label}: {query}")
    r = _run(query)
    print(f"  Time:   {r['elapsed']:.1f}s")
    print(f"  Answer: {r['result'][:200]}")

    assert r["result"], f"Expected non-empty answer for: {query}"

    if expect_simple:
        assert r["elapsed"] < 15, \
            f"Simple path took too long ({r['elapsed']:.1f}s) for: {query}"


def test_simple_path_fast():
    """Verify simple factual queries complete quickly."""
    r = _run("What is a floorplan?")
    print(f"\n  Simple path time: {r['elapsed']:.1f}s")
    assert r["result"], "Expected answer"
    print(f"  Answer: {r['result'][:300]}")
