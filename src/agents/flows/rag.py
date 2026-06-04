"""
RAGFlow pipeline:
  1. plan_query     — PlanningCrew decomposes the question into InfoNeed items (QueryPlan)
  2. retrieve       — filter-search Qdrant to identify relevant files, then content-search within them
  3. analyse_results — run TextAnalysisCrew / StructuredAnalysisCrew in parallel on retrieved vectors
  4. format_answer  — FormatCrew synthesises the findings into a markdown answer
"""
import json
import logging
import re
import sys
import warnings
from pathlib import PurePosixPath
from urllib.parse import urlparse

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from crewai.flow.flow import Flow, listen, start
from typing import List

from src.agents.aggregator import aggregate
from src.agents.classifier import classify_query, is_simple_query
from src.agents.crews import (
    PlanningCrew, FormatCrew,
    TextAnalysisCrew, StructuredAnalysisCrew,
)
from src.agents.models.analysis import AnalysisResult, DataPoint
from src.agents.models.intent import QueryIntent, TemporalConstraint
from src.agents.models.planning import QueryPlan, InfoNeed
from src.agents.schema_service import get_or_build_schema
from src.agents.simple_rag import simple_rag
from src.agents.utils import parse_crew_output
from src.models.qdrant_models import FileVector, QdrantVector
from src.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_IMAGE_TAG_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_REF_LINE_RE = re.compile(r'^(\d+)\. \[[^\]]*\]\([^)]+\)\s*$', re.MULTILINE)


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

    # 3. Deduplicate reference list entries — keep first occurrence of each number
    seen_ref_nums: set[str] = set()
    def _dedup_ref(m: re.Match) -> str:
        num = m.group(1)
        if num in seen_ref_nums:
            return ""
        seen_ref_nums.add(num)
        return m.group(0)
    md = _REF_LINE_RE.sub(_dedup_ref, md)

    # 4. Collapse runs of blank lines left by removals
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()


def _analyse_each_vector(
    vecs: List[QdrantVector], query: str, plan_steps_json: str,
    intent_flags: str = "navigational", temporal_constraint: str = "none",
) -> list[dict]:
    """Run TextAnalysisCrew on each vector individually; return a per-vector markdown report."""
    reports = []
    for v in vecs:
        logger.info("Analysing vector %s (%s)...", v.point_id, v.get_uri() or "no uri")
        vec_json = json.dumps([{"id": v.point_id, "text": v.get_payload_field("text") or ""}])
        out = TextAnalysisCrew().crew().kickoff(inputs={
            "query": query,
            "plan_steps_json": plan_steps_json,
            "text_vectors_json": vec_json,
            "intent_flags": intent_flags,
            "temporal_constraint": temporal_constraint,
        })
        report = out.raw or ""
        logger.info("Vector %s: report is %d chars", v.point_id, len(report))
        reports.append({"id": v.point_id, "report": report})
    logger.info("Per-vector analysis complete — %d reports generated", len(reports))
    return reports


class RAGFlow(Flow):

    def _emit(self, stage: str, n: int, total: int = 3) -> None:
        logger.info(stage)
        cb = self.state.get("on_progress")
        if callable(cb):
            cb(stage, n, total)

    # ------------------------------------------------------------------ stage 1

    @start()
    def plan_query(self):
        self._emit("Planning retrieval tasks", 1, 4)
        query = self.state["query"]
        collection_name = self.state.get("collection_name")
        logger.info("--- Step 1: plan_query ---")
        logger.info("Input query: %s", query)

        # Load (or build) collection schema
        schema = get_or_build_schema(collection_name) if collection_name else None
        self.state["schema"] = schema

        # Classify query intent and extract temporal constraint
        intent, temporal = classify_query(query)
        logger.info("Intent: %s  Temporal: %s", intent, temporal)
        self.state["intent"] = intent
        self.state["temporal"] = temporal

        # Simple path: single navigational lookup — skip the full pipeline
        if is_simple_query(intent, query) and collection_name:
            logger.info("Simple path triggered for query: %s", query)
            result = simple_rag(query, collection_name)
            if result:
                logger.info("Simple path returned answer (%d chars)", len(result))
                self.state["answer"] = result
                self.state["simple_path"] = True
                return
            logger.info("Simple path returned None — falling through to full pipeline")

        self.state["simple_path"] = False

        # Build facets string for planner
        facets_json = "none"
        if schema and schema.inferred_facets:
            import json as _json
            facets_json = _json.dumps({
                k: [fv.value for fv in vs]
                for k, vs in schema.inferred_facets.items()
            })
        temporal_str = "none"
        if temporal:
            parts = []
            if temporal.after:
                parts.append(f"after: {temporal.after}")
            if temporal.before:
                parts.append(f"before: {temporal.before}")
            temporal_str = "; ".join(parts) or "none"

        crew_output = PlanningCrew().crew().kickoff(inputs={
            "query": query,
            "facets_json": facets_json,
            "temporal_constraint": temporal_str,
        })
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
    def retrieve(self):
        if self.state.get("simple_path"):
            return  # answer already set
        self._emit("Retrieving relevant vectors", 2, 4)
        logger.info("--- Step 2: retrieve ---")
        plan: QueryPlan = self.state["plan"]
        collection_name = self.state.get("collection_name")

        # Filter search: identify relevant files and classify them by type.
        # Only include URIs scoring above the threshold — low-scoring results
        # often belong to unrelated documents that share the same directory.
        _URI_SCORE_THRESHOLD = 0.35
        file_uris: set[str] = set()   # specific file URIs → safe for chunk filtering
        dir_uris: set[str] = set()    # directory URIs → used only if no file URIs found
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
                if not uri or r.score < _URI_SCORE_THRESHOLD:
                    continue
                if isinstance(r, FileVector):
                    file_uris.add(uri)
                    if uri not in seen_uris:
                        seen_uris.add(uri)
                        if r.payload.image:
                            image_vectors.append(r)
                        elif r.payload.structured:
                            structured_vectors.append(r)
                else:
                    dir_uris.add(uri)

        # Use file URIs as the Pass 2 filter (precise). Fall back to directory
        # URIs only when no files were found above the threshold.
        uri_list = sorted(file_uris or dir_uris)
        logger.info(
            "URI filters: %d file, %d dir, %d image, %d structured",
            len(file_uris), len(dir_uris), len(image_vectors), len(structured_vectors),
        )

        # Content search: chunks only — file/directory vectors are already captured
        # above in image_vectors and structured_vectors, so we skip them here.
        try:
            vectors: List[QdrantVector] = []
            for plan_step in plan.needed_info:
                logger.info("Chunk search: %s", plan_step.query)
                search_results = qdrant_service.search(
                    plan_step.query,
                    collection_name=collection_name,
                    uri_filter=uri_list or None,
                    point_type=["chunk"],
                )
                logger.info("  -> %d chunk results", len(search_results))
                vectors.extend(search_results)
            logger.info("Total chunk vectors retrieved: %d", len(vectors))
        except Exception as exc:
            logger.exception("retrieve failed: %s", exc)
            raise Exception(f"An error occurred while running the crew: {exc}") from exc

        self.state["image_vectors"] = image_vectors
        self.state["structured_vectors"] = structured_vectors
        self.state["vectors"] = vectors
        return vectors

    @listen(retrieve)
    def analyse_results(self):
        if self.state.get("simple_path"):
            return  # answer already set

        self._emit("Analysing findings", 3, 4)
        logger.info("--- Step 3: analyse_results ---")
        query: str = self.state["query"]
        plan: QueryPlan = self.state["plan"]
        intent: QueryIntent = self.state.get("intent") or ["navigational"]
        temporal: TemporalConstraint | None = self.state.get("temporal")
        text_vecs: List[QdrantVector] = self.state.get("vectors") or []
        image_vecs: List[FileVector] = self.state.get("image_vectors") or []
        structured_vecs: List[FileVector] = self.state.get("structured_vectors") or []
        logger.info("Queues — image: %d, structured: %d, text: %d", len(image_vecs), len(structured_vecs), len(text_vecs))

        plan_steps_json = json.dumps([
            {"index": i, "info": step.info, "query": step.query}
            for i, step in enumerate(plan.needed_info, 1)
        ])

        intent_flags = " ".join(intent)
        temporal_str = "none"
        if temporal:
            parts = []
            if temporal.after:
                parts.append(f"after: {temporal.after}")
            if temporal.before:
                parts.append(f"before: {temporal.before}")
            temporal_str = "; ".join(parts) or "none"

        # Build point_id → uri lookup so LLMs never touch URIs during analysis.
        pid_to_uri: dict[str, str] = {}
        for v in list(image_vecs) + list(structured_vecs) + list(text_vecs):
            pid_to_uri[v.point_id] = v.get_uri() or ""
        logger.info("Built pid→uri map: %d entries", len(pid_to_uri))

        def _parse(raw: str, label: str) -> list[DataPoint]:
            try:
                raw = raw.strip()
                if "```" in raw:
                    raw = raw.split("```")[-2].lstrip("json").strip()
                items = json.loads(raw)
            except Exception as exc:
                logger.warning("%s JSON parse failed: %s", label, exc)
                return []
            points: list[DataPoint] = []
            for i, dp in enumerate(items):
                try:
                    points.append(DataPoint(**dp))
                except Exception as exc:
                    logger.warning("%s DataPoint[%d] invalid: %s — skipping", label, i, exc)
            return points

        all_data_points: list[DataPoint] = []

        # Build deduplicated text batch: image descriptions first, then text-only
        # chunks from Pass 2. Exclude any Pass 2 vector whose point_id is already
        # covered by image_vecs or structured_vecs (Pass 2 often re-returns them).
        classified_pids: set[str] = {v.point_id for v in list(image_vecs) + list(structured_vecs)}
        text_only_vecs = [v for v in text_vecs if v.point_id not in classified_pids]

        seen_pids: set[str] = set()
        all_text: list[dict] = []
        for v in list(image_vecs) + list(text_only_vecs):
            if v.point_id not in seen_pids:
                seen_pids.add(v.point_id)
                all_text.append({"id": v.point_id, "text": v.get_payload_field("text") or ""})

        if all_text:
            logger.info("Running TextAnalysisCrew on %d vectors (%d image, %d text-only)...",
                        len(all_text), len(image_vecs), len(text_only_vecs))
            out = TextAnalysisCrew().crew().kickoff(inputs={
                "query": query,
                "plan_steps_json": plan_steps_json,
                "text_vectors_json": json.dumps(all_text),
                "intent_flags": intent_flags,
                "temporal_constraint": temporal_str,
            })
            points = _parse(out.raw or "", "TextAnalysisCrew")  # type: ignore[union-attr]
            logger.info("TextAnalysisCrew returned %d data points", len(points))
            all_data_points.extend(points)
        else:
            logger.info("No text vectors — skipping TextAnalysisCrew")

        # Structured analysis.
        if structured_vecs:
            logger.info("Running StructuredAnalysisCrew on %d vectors...", len(structured_vecs))
            out = StructuredAnalysisCrew().crew().kickoff(inputs={
                "query": query,
                "plan_steps_json": plan_steps_json,
            })
            points = _parse(out.raw or "", "StructuredAnalysisCrew")  # type: ignore[union-attr]
            logger.info("StructuredAnalysisCrew returned %d data points", len(points))
            all_data_points.extend(points)
        else:
            logger.info("No structured vectors — skipping StructuredAnalysisCrew")

        logger.info("Analysis complete — %d total data points", len(all_data_points))

        # Numeric aggregation
        agg = aggregate(all_data_points, intent, query)
        if agg:
            logger.info("Aggregation: %s", agg.summary())
        self.state["aggregation"] = agg

        result = AnalysisResult(data_points=all_data_points)
        self.state["analysis"] = result
        self.state["pid_to_uri"] = pid_to_uri
        return result

    @listen(analyse_results)
    def format_answer(self):
        if self.state.get("simple_path"):
            return self.state.get("answer")  # already formatted
        if not self.state.get("synthesis"):
            return None
        self._emit("Formatting answer", 4, 4)
        logger.info("--- Step 4: format_answer ---")
        query: str = self.state["query"]
        analysis: AnalysisResult = self.state["analysis"]
        pid_to_uri: dict[str, str] = self.state.get("pid_to_uri") or {}

        # Swap point_ids to real URIs before building findings — FormatCrew must
        # see actual URLs so it can cite them correctly. Doing this after FormatCrew
        # causes hallucinated placeholders because UUIDs look meaningless to the LLM.
        def _resolve(pid: str) -> str:
            return pid_to_uri.get(pid, pid)

        findings = "\n\n".join(
            f"uri: {_resolve(dp.uri)}\nstep: {dp.step}\nquote: {dp.quote}\nreasoning: {dp.reasoning}"
            for dp in analysis.data_points
            if dp.pertinent
        )

        # Prepend aggregated numeric result if available
        agg = self.state.get("aggregation")
        if agg:
            agg_line = f"Computed result: {agg.summary()}\n\n"
            findings = agg_line + findings

        crew_output = FormatCrew().crew().kickoff(inputs={"query": query, "findings": findings})  # type: ignore[union-attr]
        markdown = _clean_markdown(crew_output.raw or "")

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