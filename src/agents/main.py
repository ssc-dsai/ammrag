#!/usr/bin/env python
import sys
import json
import warnings
from src.core.config import settings


from src.agents.flows.rag import RAGFlow


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from src.agents.crews import PlanningCrew, FormatCrew
from src.agents.models.parsing import ParsedQuery
from src.services.qdrant_service import qdrant_service
from typing import List

def run():
    """
    Prompt the user for a query and run the RAG crew interactively.
    """
    try:
        query = input("Enter your query: ").strip()
        collection_name = input("Enter your collection name: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if not query or collection_name:
        print("Error: query or collection name cannot be empty.")
        sys.exit(1)

    answer = RAGFlow().kickoff(inputs={"query": query, "collection_name": collection_name})

    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(answer)


# def query(query: str, collection_name: str = "ammrag") -> str:
#     try:

#         result: List[dict] = []
#         crew_output = ParsingCrew().crew().kickoff(inputs={"query": query})
#         parsedQuery: ParsedQuery = crew_output.pydantic # type: ignore

#         for subquery in parsedQuery.subquery_list:
#             print(f"\nSub-query: {subquery}")
#             search_results = qdrant_service.search(query=subquery, collection_name=collection_name)
#             result.extend(vr.model_dump() for vr in search_results)

#         answer = FormatCrew().crew().kickoff(inputs={"query": query, "result": json.dumps(result)})
#         return answer
#     except Exception as exc:
#         raise Exception(f"An error occurred while running the crew: {exc}") from exc

def trigger():
    """
    Run the crew with a JSON payload supplied as the first CLI argument.
    Expected format: {"query": "..."}
    """
    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Pass a JSON string as the first argument, "
            'e.g.: run_with_trigger \'{"query": "What is X?"}\''
        )

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        raise Exception(f"Invalid JSON payload: {exc}") from exc

    query = payload.get("query", "").strip()
    if not query:
        raise Exception("The trigger payload must contain a non-empty 'query' field.")

    try:
        result = RAGFlow().kickoff(inputs={"query": query})
        return result
    except Exception as exc:
        raise Exception(f"An error occurred while running the crew: {exc}") from exc


def train():
    """Train the crew for a given number of iterations."""
    if len(sys.argv) < 3:
        raise Exception("Usage: train <n_iterations> <output_filename>")
    try:
        RAGFlow().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs={"query": "sample training query"},
        )
    except Exception as exc:
        raise Exception(f"An error occurred while training the crew: {exc}") from exc


def replay():
    """Replay the crew execution from a specific task ID."""
    if len(sys.argv) < 2:
        raise Exception("Usage: replay <task_id>")
    try:
        RAGFlow().replay(task_id=sys.argv[1])
    except Exception as exc:
        raise Exception(f"An error occurred while replaying the crew: {exc}") from exc


def test():
    try:
        answer = query("What is WBA and where is it implemented in the GOC?", "nss")
        print("\n" + "=" * 72)
        print("FINAL ANSWER")
        print("=" * 72)
        print(answer)
    except Exception as exc:
        raise Exception(f"An error occurred while running the crew: {exc}") from exc

if __name__ == "__main__":
    import asyncio
    from src.services.crew_service import crew_service

    query1 = "What is WBA and where is it implemented in the GOC?"
    result = asyncio.run(crew_service.query(question=query1, project_name="nss"))
    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(result)
