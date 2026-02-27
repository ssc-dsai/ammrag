#!/usr/bin/env python
import sys
import json
import warnings

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from src.agents.crew import AmmRagCrew


def run():
    """
    Prompt the user for a query and run the RAG crew interactively.
    """
    # try:
    #     query = input("Enter your query: ").strip()
    # except (EOFError, KeyboardInterrupt):
    #     print()
    #     sys.exit(0)

    # if not query:
    #     print("Error: query cannot be empty.")
    #     sys.exit(1)

    try:
        result = AmmRagCrew().crew().kickoff(inputs={"query": "What is WBA and where is it implemented in the GOC?"})
        print("\n" + "=" * 72)
        print("FINAL ANSWER")
        print("=" * 72)
        print(result)
    except Exception as exc:
        raise Exception(f"An error occurred while running the crew: {exc}") from exc


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
        result = AmmRagCrew().crew().kickoff(inputs={"query": query})
        return result
    except Exception as exc:
        raise Exception(f"An error occurred while running the crew: {exc}") from exc


def train():
    """Train the crew for a given number of iterations."""
    if len(sys.argv) < 3:
        raise Exception("Usage: train <n_iterations> <output_filename>")
    try:
        AmmRagCrew().crew().train(
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
        AmmRagCrew().crew().replay(task_id=sys.argv[1])
    except Exception as exc:
        raise Exception(f"An error occurred while replaying the crew: {exc}") from exc


# def test():
#     """Test the crew execution and return the results."""
#     if len(sys.argv) < 3:
#         raise Exception("Usage: test <n_iterations> <eval_llm>")
#     try:
#         AmmRagCrew().crew().test(
#             n_iterations=int(sys.argv[1]),
#             eval_llm=sys.argv[2],
#             inputs={"query": "What is WBA?"},
#         )
#     except Exception as exc:
#         raise Exception(f"An error occurred while testing the crew: {exc}") from exc

def test():
    try:
        result = AmmRagCrew().crew().kickoff(inputs={"query": "What is WBA?"})
        print("\n" + "=" * 72)
        print("FINAL ANSWER")
        print("=" * 72)
        print(result)
    except Exception as exc:
        raise Exception(f"An error occurred while running the crew: {exc}") from exc
    
if __name__ == "__main__":
    run()
