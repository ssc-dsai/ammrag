"""Live test — requires Qdrant and Ollama. Verifies value/date extraction in DataPoint."""
import json
import pytest

from src.agents.crews.text_analysis import TextAnalysisCrew
from src.agents.models.analysis import DataPoint


def _run_crew(vectors: list[dict], query: str, intent_flags: str = "structured",
              temporal: str = "none") -> list[DataPoint]:
    plan_steps_json = json.dumps([{"index": 1, "info": query, "query": query}])
    text_vectors_json = json.dumps(vectors)

    out = TextAnalysisCrew().crew().kickoff(inputs={
        "query": query,
        "plan_steps_json": plan_steps_json,
        "text_vectors_json": text_vectors_json,
        "intent_flags": intent_flags,
        "temporal_constraint": temporal,
    })

    raw = out.raw or ""
    if "```" in raw:
        raw = raw.split("```")[-2].lstrip("json").strip()
    raw = raw.strip()
    if not raw.startswith("["):
        # crew may wrap in object; try to extract array
        import re
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        raw = m.group(0) if m else "[]"

    points = []
    for d in json.loads(raw):
        try:
            points.append(DataPoint(**d))
        except Exception:
            pass
    return points


def test_value_extracted_from_numeric_text():
    vectors = [{"id": "test-1", "text": "Last configuration change at 14:22:25 EST Wed Feb 28 2024. Current config: 67522 bytes"}]
    points = _run_crew(vectors, "How many bytes is the configuration?", intent_flags="structured")

    print("\n--- DataPoints with value ---")
    for p in points:
        print(f"  pertinent={p.pertinent} value={p.value} date={p.date} quote={p.quote[:60]}")

    assert len(points) > 0
    values = [p.value for p in points if p.value is not None]
    print(f"  Extracted values: {values}")
    assert len(values) > 0, "Expected at least one numeric value to be extracted"


def test_date_extracted_from_dated_text():
    vectors = [{"id": "test-2", "text": "Last configuration change at 13:09:34 EDT Mon Aug 5 2024 by name. NVRAM config last updated at 13:09:42 EDT Mon Aug 5 2024."}]
    points = _run_crew(vectors, "When was the configuration last changed?", intent_flags="temporal")

    print("\n--- DataPoints with date ---")
    for p in points:
        print(f"  pertinent={p.pertinent} value={p.value} date={p.date}")

    dates = [p.date for p in points if p.date is not None]
    print(f"  Extracted dates: {dates}")
    assert len(dates) > 0, "Expected at least one date to be extracted"


def test_temporal_filter_excludes_out_of_range():
    vectors = [{"id": "test-3", "text": "Investment report from March 2019. All teams were expanded."}]
    points = _run_crew(
        vectors,
        "What investments were made after 2022?",
        intent_flags="temporal",
        temporal="after: 2022-01-01",
    )

    print("\n--- Temporal filter test ---")
    for p in points:
        print(f"  pertinent={p.pertinent} date={p.date} reasoning={p.reasoning[:80]}")

    pertinent = [p for p in points if p.pertinent]
    assert len(pertinent) == 0, \
        f"Expected all 2019 results to be marked not pertinent after 2022 filter, got: {pertinent}"
