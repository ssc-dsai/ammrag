"""Live test — requires Ollama at 192.168.68.92:11434."""
import pytest
from src.agents.classifier import classify_query, is_simple_query

CASES = [
    ("What is the definition of WBA?",                    ["navigational"],             False),
    ("Where is the kitchen in the London office?",         ["spatial"],                  False),
    ("Which invoices from Q3 2024 exceeded £10,000?",     ["structured", "temporal"],   False),
    ("Compare kitchens across all offices",               ["compound"],                  False),
    ("What refurbishments were done since 2022?",         ["temporal"],                  False),
    ("Show me the floorplan for Edinburgh",               ["visual"],                    False),
]

SIMPLE_CASES = [
    ("What is the definition of WBA?",              True),
    ("Define asbestos",                             True),
    ("What does WBA stand for?",                    True),
    ("Compare kitchens across all offices",         False),
    ("What is the total number of employees?",      False),
    ("Where is the kitchen?",                       False),
]


@pytest.mark.parametrize("query,expected_tags,_", CASES)
def test_classify_contains_expected_tags(query, expected_tags, _):
    intent, temporal = classify_query(query)
    print(f"\n  '{query}' → {intent}, temporal={temporal}")
    for tag in expected_tags:
        assert tag in intent, f"Expected '{tag}' in intent {intent} for query: {query}"


def test_temporal_constraint_extracted():
    _, tc = classify_query("Which invoices from Q3 2024 exceeded £10,000?")
    assert tc is not None, "Expected a TemporalConstraint for a temporal query"
    print(f"\n  TemporalConstraint: {tc}")

    _, tc2 = classify_query("What refurbishments were done since 2022?")
    assert tc2 is not None
    print(f"  TemporalConstraint (since 2022): {tc2}")


@pytest.mark.parametrize("query,expected", SIMPLE_CASES)
def test_is_simple_query(query, expected):
    intent, _ = classify_query(query)
    result = is_simple_query(intent, query)
    print(f"\n  '{query}' → intent={intent}, simple={result}")
    assert result == expected, f"is_simple_query wrong for: {query}"
