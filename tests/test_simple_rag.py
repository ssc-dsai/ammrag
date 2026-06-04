"""Live test — requires Qdrant and Ollama."""
import time
import pytest

from src.agents.classifier import classify_query, is_simple_query
from src.agents.simple_rag import simple_rag


# ── is_simple_query gate ─────────────────────────────────────────────────────

def test_simple_gate_navigational_lookup():
    intent, _ = classify_query("What is the definition of WBA?")
    assert is_simple_query(intent, "What is the definition of WBA?") is True


def test_simple_gate_compound_not_simple():
    intent, _ = classify_query("Compare kitchens across all offices")
    assert is_simple_query(intent, "Compare kitchens across all offices") is False


def test_simple_gate_structured_not_simple():
    intent, _ = classify_query("What is the total number of employees?")
    assert is_simple_query(intent, "What is the total number of employees?") is False


# ── simple_rag live search ────────────────────────────────────────────────────

def test_simple_rag_returns_result_for_known_content():
    start = time.time()
    result = simple_rag("floorplan", collection_name="nss")
    elapsed = time.time() - start

    print(f"\n  Elapsed: {elapsed:.2f}s")
    print(f"  Result: {str(result)[:200]}")

    # Should find something in nss — collection has floorplan images
    assert result is not None, "Expected a result for 'floorplan' in nss"
    assert "Source:" in result
    assert elapsed < 5.0, f"simple_rag took too long: {elapsed:.1f}s"


def test_simple_rag_low_confidence_returns_none():
    # Extremely unlikely query — should score below threshold
    result = simple_rag(
        "floorplan",
        collection_name="nss",
        threshold=0.99,  # above any real score in this hybrid-embedding collection
    )
    assert result is None, "Expected None for very high threshold / nonsense query"
