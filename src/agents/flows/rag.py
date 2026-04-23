"""
RAGFlow pipeline:
  1. plan_query     — PlanningCrew decomposes the question into InfoNeed items (QueryPlan)
  create_result_filters — looks at any file filters and returns findings (not implemented yet)
  retrieve_vectors — For each InfoNeed, search Qdrant and return relevant vectors
  retrieve_structured  — looks at any structured data and returns findings (not implemented yet)
  analyse_images   —  looks at any images and returns findings (not implemented yet)
  analyse_findings
  retry_without_filters — If no findings, retry the plan without filters (not implemented yet)
  repeat_retrieval — If no findings, repeat retrieval with modified queries (not implemented yet)
  3. format_answer  — FormatCrew synthesises the findings into a markdown answer
"""
import json
import logging
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from urllib.parse import urlparse

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from crewai.flow.flow import Flow, listen, start
from typing import List

from src.agents.crews import (
    PlanningCrew, FormatCrew,
    ImageAnalysisCrew, TextAnalysisCrew, StructuredAnalysisCrew,
)
from src.agents.models.analysis import AnalysisResult, DataPoint
from src.agents.models.planning import QueryPlan, InfoNeed
from src.agents.utils import parse_crew_output
from src.models.qdrant_models import FileVector, QdrantVector
from src.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_IMAGE_TAG_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)')


def _is_image_uri(uri: str) -> bool:
    return PurePosixPath(urlparse(uri).path).suffix.lower() in _IMAGE_EXTS


def _clean_markdown(md: str) -> str:
    """Remove duplicate images/links and strip non-image ![]() render syntax."""
    # 1. Convert any ![]() whose URI is not an image extension to plain []()
    md = _IMAGE_TAG_RE.sub(
        lambda m: m.group(0) if _is_image_uri(m.group(2)) else f"[{m.group(1)}]({m.group(2)})",
        md,
    )

    # 2. Deduplicate image renders — keep only the first ![]() per URI
    seen_images: set[str] = set()
    def _dedup_image(m: re.Match) -> str:
        uri = m.group(2)
        if uri in seen_images:
            return ""
        seen_images.add(uri)
        return m.group(0)
    md = _IMAGE_TAG_RE.sub(_dedup_image, md)

    # 3. Deduplicate plain links — keep only the first [text](uri) per URI
    seen_links: set[str] = set()
    def _dedup_link(m: re.Match) -> str:
        uri = m.group(2)
        if uri in seen_links:
            return ""
        seen_links.add(uri)
        return m.group(0)
    md = _LINK_RE.sub(_dedup_link, md)

    # 4. Collapse runs of blank lines left by removals
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()


class RAGFlow(Flow):

    def _emit(self, stage: str, n: int, total: int = 3) -> None:
        logger.info(stage)
        cb = self.state.get("on_progress")
        if callable(cb):
            cb(stage, n, total)

    # ------------------------------------------------------------------ stage 1

    @start()
    def plan_query(self):
        self._emit("Planning retrieval tasks", 1)
        query = self.state["query"]
        logger.info("--- Step 1: plan_query ---")
        logger.info("Input query: %s", query)

        crew_output = PlanningCrew().crew().kickoff(inputs={"query": query})
        plan = parse_crew_output(
            crew_output,
            QueryPlan,
            fallback=QueryPlan(orig_query=query, needed_info=[
                InfoNeed(info="Search knowledge base", query=query)
            ]),
        )

        logger.info("Plan produced (%d task(s)):", len(plan.needed_info))
        for i, t in enumerate(plan.needed_info, 1):
            logger.info("  [%d] info=%s  query=%s", i, t.info, t.query)

        self.state["plan"] = plan

    # ------------------------------------------------------------------ stage 2
    @listen(plan_query)
    def create_result_filters(self):
        logger.info("--- Step 2: create_result_filters ---")
        plan: QueryPlan = self.state["plan"]
        collection_name = self.state.get("collection_name")
        uris: set[str] = set()
        image_vectors: list[FileVector] = []
        structured_vectors: list[FileVector] = []
        seen_uris: set[str] = set()
        for plan_step in plan.needed_info:
            logger.info("Filter search: %s", plan_step.query)
            results = qdrant_service.search(
                plan_step.query,
                collection_name=collection_name,
                limit=10,
                point_type=["file", "directory"],
            )
            for r in results:
                uri = r.get_uri()
                if not uri:
                    continue
                uris.add(uri)
                if isinstance(r, FileVector) and uri not in seen_uris:
                    seen_uris.add(uri)
                    if r.payload.image:
                        image_vectors.append(r)
                    elif r.payload.structured:
                        structured_vectors.append(r)
        uri_list = sorted(uris)
        logger.info(
            "URI filters: %d total, %d image, %d structured",
            len(uri_list), len(image_vectors), len(structured_vectors),
        )
        self.state["uri_filter"] = uri_list
        self.state["image_vectors"] = image_vectors
        self.state["structured_vectors"] = structured_vectors

    @listen(create_result_filters)
    def retrieve_vectors(self):
        logger.info("--- Step 3: retrieve_vectors ---")
        try:
            plan: QueryPlan = self.state["plan"]
            collection_name = self.state.get("collection_name")
            uri_filter: list[str] = self.state.get("uri_filter") or []
            logger.info("Collection: %s, URI filter size: %d", collection_name, len(uri_filter))
            vectors: List[QdrantVector] = []
            for plan_step in plan.needed_info:
                logger.info("Searching: %s", plan_step.query)
                search_results = qdrant_service.search(
                    plan_step.query,
                    collection_name=collection_name,
                    uri_filter=uri_filter or None,
                )
                logger.info("  -> %d results", len(search_results))
                vectors.extend(search_results)
            logger.info("Total vectors retrieved: %d", len(vectors))
            self.state["vectors"] = vectors
            return vectors
        except Exception as exc:
            logger.exception("retrieve_vectors failed: %s", exc)
            raise Exception(f"An error occurred while running the crew: {exc}") from exc

    @listen(retrieve_vectors)
    def analyse_results(self):
        logger.info("--- Step 4: analyse_results ---")
        query: str = self.state["query"]
        plan: QueryPlan = self.state["plan"]
        text_vecs: List[QdrantVector] = self.state.get("vectors") or []
        image_vecs: List[FileVector] = self.state.get("image_vectors") or []
        structured_vecs: List[FileVector] = self.state.get("structured_vectors") or []
        logger.info(
            "Queues — image: %d, structured: %d, text: %d",
            len(image_vecs), len(structured_vecs), len(text_vecs),
        )

        plan_steps_json = json.dumps([
            {"index": i, "info": step.info, "query": step.query}
            for i, step in enumerate(plan.needed_info, 1)
        ])

        # Build point_id → uri lookup so LLMs never touch URIs during analysis.
        pid_to_uri: dict[str, str] = {}
        for v in list(image_vecs) + list(structured_vecs) + list(text_vecs):
            pid_to_uri[v.point_id] = v.get_uri() or ""

        def _parse(raw: str, label: str) -> list[DataPoint]:
            try:
                raw = raw.strip()
                if "```" in raw:
                    raw = raw.split("```")[-2].lstrip("json").strip()
                return [DataPoint(**dp) for dp in json.loads(raw)]
            except Exception as exc:
                logger.warning("%s DataPoint parse failed: %s", label, exc)
                return []

        def _run_image() -> list[DataPoint]:
            if not image_vecs:
                return []
            # Include both id (stable identifier for DataPoint) and uri (needed for tool calls).
            image_vectors_json = json.dumps([
                {"id": v.point_id, "uri": v.get_uri() or "", "text": v.get_payload_field("text") or ""}
                for v in image_vecs
            ])
            out = ImageAnalysisCrew().crew().kickoff(inputs={
                "query": query,
                "plan_steps_json": plan_steps_json,
                "image_vectors_json": image_vectors_json,
            })
            return _parse(out.raw or "", "ImageAnalysisCrew")  # type: ignore[union-attr]

        def _run_structured() -> list[DataPoint]:
            if not structured_vecs:
                return []
            out = StructuredAnalysisCrew().crew().kickoff(inputs={
                "query": query,
                "plan_steps_json": plan_steps_json,
            })
            return _parse(out.raw or "", "StructuredAnalysisCrew")  # type: ignore[union-attr]

        def _run_text() -> list[DataPoint]:
            if not text_vecs:
                return []
            # Use id instead of uri so the LLM cannot modify the identifier.
            text_vectors_json = json.dumps([
                {
                    "id": v.point_id,
                    "text": v.get_payload_field("text") or "",
                }
                for v in text_vecs
            ])
            out = TextAnalysisCrew().crew().kickoff(inputs={
                "query": query,
                "plan_steps_json": plan_steps_json,
                "text_vectors_json": text_vectors_json,
            })
            return _parse(out.raw or "", "TextAnalysisCrew")  # type: ignore[union-attr]

        with ThreadPoolExecutor(max_workers=3) as executor:
            image_future = executor.submit(_run_image)
            structured_future = executor.submit(_run_structured)
            text_future = executor.submit(_run_text)
            image_points = image_future.result()
            structured_points = structured_future.result()
            text_points = text_future.result()

        all_data_points = image_points + structured_points + text_points
        logger.info(
            "Analysis complete — image: %d, structured: %d, text: %d, total: %d",
            len(image_points), len(structured_points), len(text_points), len(all_data_points),
        )
        result = AnalysisResult(data_points=all_data_points)
        self.state["analysis"] = result
        self.state["pid_to_uri"] = pid_to_uri
        return result

    @listen(analyse_results)
    def format_answer(self):
        if not self.state.get("synthesis"):
            return None
        logger.info("--- Step 5: format_answer ---")
        query: str = self.state["query"]
        analysis: AnalysisResult = self.state["analysis"]

        findings = "\n\n".join(
            f"uri: {dp.uri}\nstep: {dp.step}\nquote: {dp.quote}\nreasoning: {dp.reasoning}"
            for dp in analysis.data_points
            if dp.pertinent
        )

        crew_output = FormatCrew().crew().kickoff(inputs={"query": query, "findings": findings})  # type: ignore[union-attr]
        markdown = _clean_markdown(crew_output.raw or "")

        # Swap point_ids back to real URIs now that all LLM processing is complete.
        pid_to_uri: dict[str, str] = self.state.get("pid_to_uri") or {}
        for pid, uri in pid_to_uri.items():
            markdown = markdown.replace(pid, uri)

        logger.info("FormatCrew complete: %d chars", len(markdown))
        self.state["answer"] = markdown
        return markdown

    # @listen(plan_query_old)
    # def dispatch_tasks(self):
    #     self._emit("Dispatching to agents", 2)
    #     logger.info("--- Step 2: dispatch_tasks ---")

    #     plan: QueryPlan = self.state["plan"]
    #     collection_name = self.state.get("collection_name")

    #     plan_text = "\n".join(
    #         f"  {i}. info={t.info!r}  query={t.query!r}"
    #         for i, t in enumerate(plan.needed_info, 1)
    #     )
    #     logger.info("Dispatching plan:\n%s", plan_text)

    #     crew_output = DispatchCrew(collection_name=collection_name).crew().kickoff(
    #         inputs={
    #             "orig_query": plan.orig_query,
    #             "plan": plan_text,
    #         }
    #     )
    #     findings = crew_output.raw or ""
    #     logger.info("DispatchCrew findings length: %d chars", len(findings))
    #     self.state["findings"] = findings

    # ------------------------------------------------------------------ stage 3

    # @listen(dispatch_tasks)
    # def format_answer(self):
    #     self._emit("Synthesising answer", 3)
    #     logger.info("--- Step 3: format_answer ---")

    #     query = self.state["query"]
    #     findings: str = self.state.get("findings", "")

    #     # Extract URIs from findings and replace with opaque tokens so FormatCrew
    #     # cannot normalise percent-encoded sequences (e.g. %20%20).
    #     uri_map: dict[str, str] = {}
    #     def _tokenise(m: re.Match) -> str:
    #         uri = m.group(0)
    #         if uri not in uri_map.values():
    #             token = f"src_{len(uri_map)}"
    #             uri_map[token] = uri
    #         else:
    #             token = next(k for k, v in uri_map.items() if v == uri)
    #         return token

    #     tokenised_findings = _URI_RE.sub(_tokenise, findings)
    #     logger.info("URI tokens created: %d", len(uri_map))

    #     crew_output = FormatCrew().crew().kickoff(
    #         inputs={"query": query, "findings": tokenised_findings}
    #     )
    #     markdown = crew_output.raw or ""

    #     # Restore original URIs.
    #     for token, uri in uri_map.items():
    #         markdown = markdown.replace(token, uri)

    #     sources = [uri for token, uri in uri_map.items() if token in (crew_output.raw or "")]
    #     logger.info("FormatCrew complete. output=%d chars, sources=%d", len(markdown), len(sources))
    #     logger.info("=== Final answer ===\n%s", markdown)

    #     return {
    #         "answer": markdown,
    #         "sources": sources,
    #         "images": [],
    #         "files": [],
    #     }


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()

    query = " ".join(sys.argv[1:]) or "Get the floorplans for all offices, then find the kitchen location in each one."

    inputs = {"query": query, "collection_name": "nss", "synthesis": True}
    result = RAGFlow().kickoff(inputs=inputs)
    print("=== Final result ===")
    print(result)