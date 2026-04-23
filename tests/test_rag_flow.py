"""
Integration test for the RAGFlow pipeline.

Runs the full plan → dispatch → format pipeline against the live Qdrant and
Ollama services.  Requires the stack to be running; skip with:

    pytest -k "not rag_flow"

or mark the whole session with --ignore=tests/test_rag_flow.py.
"""

import pytest

from src.core.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_flow(question: str, collection: str | None = None) -> dict:
    from src.agents.flows.rag import RAGFlow
    result = RAGFlow().kickoff(
        inputs={"query": question, "collection_name": collection}
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    return result


def _print_result(question: str, result: dict) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"QUESTION: {question}")
    print(sep)
    print(result.get("answer", "(no answer)"))
    if result.get("sources"):
        print("\nSources:")
        for s in result["sources"]:
            print(f"  {s}")
    print(sep)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_rag_flow_returns_answer():
    """RAGFlow result must contain a non-empty answer string."""
    question = "What is WBA?"
    result = _run_flow(question, collection=settings.mcp_collection_name)
    _print_result(question, result)

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


@pytest.mark.integration
def test_rag_flow_result_shape():
    """RAGFlow result must have all expected keys."""
    result = _run_flow("What is WBA?", collection=settings.mcp_collection_name)

    for key in ("answer", "sources", "images", "files"):
        assert key in result, f"Missing key: {key}"

    assert isinstance(result["sources"], list)
    assert isinstance(result["images"], list)
    assert isinstance(result["files"], list)


@pytest.mark.integration
def test_rag_flow_sources_are_valid_uris():
    """Every source returned must look like a URI (starts with http)."""
    result = _run_flow(
        "Get the floorplans for all offices",
        collection=settings.mcp_collection_name,
    )
    _print_result("Get the floorplans for all offices", result)

    for source in result["sources"]:
        assert source.startswith("http"), (
            f"Source does not look like a URI: {source!r}"
        )


@pytest.mark.integration
def test_rag_flow_answer_contains_citation_links():
    """The markdown answer must contain at least one inline citation link [n](...)."""
    import re
    result = _run_flow("What is WBA?", collection=settings.mcp_collection_name)

    citation_pattern = re.compile(r'\[\d+\]\(https?://[^\)]+\)')
    assert citation_pattern.search(result["answer"]), (
        "Answer contains no inline citation links.\n\n" + result["answer"]
    )
