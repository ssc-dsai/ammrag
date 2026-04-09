"""
Pipeline for RAG flow
- qdrant retrieval
- if contains a db pointer, retrieve table from db

- context widening
"""
from crewai.flow.flow import Flow, listen, start

import json
import logging
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from src.agents.crews import ParsingCrew, FormatCrew, PostgresCrew
from src.agents.models.answer import QueryAnswer
from src.agents.models.parsing import ParsedQuery
from src.services.qdrant_service import qdrant_service
from typing import List
from src.models.qdrant_models import QdrantVector

logger = logging.getLogger(__name__)


class RAGFlow(Flow):

    @start()
    def parse_query(self):
        query = self.state["query"]
        logger.info("--- Step 1: parse_query ---")
        logger.info("Input query: %s", query)

        crew_output = ParsingCrew().crew().kickoff(inputs={"query": query})
        parsedQuery: ParsedQuery = crew_output.pydantic  # type: ignore

        logger.info("Sub-queries generated (%d):", len(parsedQuery.subquery_list))
        for i, sq in enumerate(parsedQuery.subquery_list, 1):
            logger.info("  [%d] %s", i, sq)

        self.state["parsed_query"] = parsedQuery

    @listen(parse_query)
    def retrieve_vectors(self, parsedQuery: ParsedQuery):
        logger.info("--- Step 2: retrieve_vectors ---")
        try:
            parsedQuery = self.state["parsed_query"]
            collection_name = self.state.get("collection_name")
            logger.info("Collection: %s", collection_name)
            vectors: List[QdrantVector] = []
            for subquery in parsedQuery.subquery_list:
                logger.info("Searching: %s", subquery)
                search_results = qdrant_service.search(subquery, collection_name=collection_name)
                logger.info("  -> %d results", len(search_results))
                vectors.extend(search_results)
            logger.info("Total vectors retrieved: %d", len(vectors))
            return vectors
        except Exception as exc:
            logger.exception("retrieve_vectors failed: %s", exc)
            raise Exception(f"An error occurred while running the crew: {exc}") from exc

    @listen(retrieve_vectors)
    def retrieve_tables(self, retrieval_results: List[QdrantVector]):
        logger.info("--- Step 3: retrieve_tables ---")
        query = self.state["query"]
        self.state["retrieval_results"] = retrieval_results

        structured = [
            v for v in retrieval_results
            if v.get_payload_field("structured") is True
        ]
        logger.info("%d structured (db-pointer) vector(s) found", len(structured))

        table_results: list[str] = []
        if structured:
            table_names = list({
                v.get_payload_field("uri") for v in structured
                if v.get_payload_field("uri")
            })
            logger.info("Querying PostgresCrew for tables: %s", table_names)
            crew_output = PostgresCrew().crew().kickoff(
                inputs={"user_query": query, "table_names": ", ".join(table_names)}
            )
            table_results.append(crew_output.raw)
            logger.info("PostgresCrew returned %d result(s)", len(table_results))

        self.state["table_results"] = table_results

    @listen(retrieve_tables)
    def format_answer(self):
        logger.info("--- Step 4: format_answer ---")
        query = self.state["query"]
        retrieval_results: List[QdrantVector] = self.state["retrieval_results"]
        table_results: list[str] = self.state.get("table_results", [])

        for i, vector in enumerate(retrieval_results, 1):
            text = (vector.get_payload_field("text") or "")[:120]
            source = vector.get_payload_field("uri") or "unknown"
            logger.info("  [%d] source=%s | text=%s...", i, source, text)

        serialized = [v.model_dump() for v in retrieval_results]
        inputs = {
            "query": query,
            "vectors": json.dumps(serialized),
            "table_data": json.dumps(table_results),
        }
        logger.info("Kicking off FormatCrew with %d chunks, %d table result(s)", len(serialized), len(table_results))
        crew_output = FormatCrew().crew().kickoff(inputs=inputs)

        answer: QueryAnswer | None = crew_output.pydantic  # type: ignore
        if answer is None:
            logger.warning("FormatCrew did not return a structured QueryAnswer — falling back to raw text")
            answer = QueryAnswer(aspects=[])

        sources = list({a.source for a in answer.aspects if a.source and a.source != "not found"})
        logger.info("FormatCrew complete. %d aspect(s), %d source(s)", len(answer.aspects), len(sources))
        return {
            "aspects": [a.model_dump() for a in answer.aspects],
            "sources": sources,
            "images": [],
            "files": [],
        }

