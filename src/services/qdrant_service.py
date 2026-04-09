"""
Qdrant vector search service for FastAPI

Handles semantic similarity search and document retrieval
from the Qdrant vector database.
"""
import json
import logging
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
import uuid as uuid_lib
from src.core.config import settings
from src.models.qdrant_models import (
    QdrantVector, ChunkVector, FileVector, DirectoryVector,
    ChunkPayload, FilePayload, DirectoryPayload,
)
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.schema import TextNode
from transformers import AutoTokenizer, AutoModelForMaskedLM
from fastembed import SparseTextEmbedding, SparseEmbedding
import torch
logger = logging.getLogger(__name__)

DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}


def _make_vector(rank: int, point_id: str, score: float, payload: dict, collection_name: str = "") -> QdrantVector:
    """Build a typed vector subclass from a raw payload dict."""
    vtype = payload.get("type", "")
    try:
        if vtype == "chunk":
            return ChunkVector(rank=rank, point_id=point_id, score=score, collection_name=collection_name, payload=ChunkPayload(**payload))
        if vtype == "file":
            return FileVector(rank=rank, point_id=point_id, score=score, collection_name=collection_name, payload=FilePayload(**payload))
        if vtype == "directory":
            return DirectoryVector(rank=rank, point_id=point_id, score=score, collection_name=collection_name, payload=DirectoryPayload(**payload))
    except Exception as exc:
        logger.warning("Failed to parse payload as %s (point_id=%s): %s — using base QdrantVector", vtype, point_id, exc)
    return QdrantVector(rank=rank, point_id=point_id, score=score, collection_name=collection_name, payload=payload)


class EmbedderInfo:
    """Class to hold embedder and tokenizer information"""
    @property
    def embedder_name(self) -> str:
        return settings.dense_embedder_model

    @property
    def embedder_token_limit(self) -> int:
        return 256

    @property
    def llm_name(self) -> str:
        return settings.ollama_model

    @property
    def llm_token_limit(self) -> int:
        return 4096


embedder_info = EmbedderInfo()


class QdrantService:
    """Service for Qdrant vector search operations"""

    def __init__(self):
        self._client = None
        self._dense_embedder = None
        self._tokenizer = None
        self._sparse_model = None
        self._sparse_tokenizer = None
        self._check_connection()

    @property
    def client(self):
        if self._client is None:
            kwargs = {"url": settings.qdrant_url}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            if settings.qdrant_timeout:
                kwargs["timeout"] = settings.qdrant_timeout
            self._client = QdrantClient(**kwargs)
        return self._client

    @property
    def embedder(self) -> SentenceTransformer:
        if self._dense_embedder is None:
            self._dense_embedder = SentenceTransformer(settings.dense_embedder_model)
        return self._dense_embedder

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.dense_embedder_model)
        return self._tokenizer

    @property
    def sparse_tokenizer(self) -> AutoTokenizer:
        if self._sparse_tokenizer is None:
            self._sparse_tokenizer = AutoTokenizer.from_pretrained(settings.sparse_embedder_model)
        return self._sparse_tokenizer

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=settings.sparse_embedder_model)
        return self._sparse_model

    def _encode_sparse(self, text: str) -> models.SparseVector:
        """Encode text using SPLADE and return a SparseVector."""
        tokens = self.sparse_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.sparse_model(**tokens)
        vec = torch.log(1 + torch.relu(output.logits)).max(dim=1).values.squeeze()
        nonzero = vec.nonzero(as_tuple=False).squeeze(dim=1)
        indices = nonzero.tolist()
        values = vec[nonzero].tolist()
        if isinstance(indices, int):
            indices, values = [indices], [values]
        return models.SparseVector(indices=indices, values=values)

    def _check_connection(self) -> None:
        """Verify Qdrant connectivity on startup."""
        try:
            collections = self.client.get_collections()
            logger.info("Qdrant connection OK (%s) — %d collection(s)",
                        settings.qdrant_url, len(collections.collections))
        except Exception as e:
            logger.error("Qdrant connection FAILED: %s", e)

    def ensure_collection(self, collection_name: str) -> None:
        """Create the collection if it doesn't already exist."""
        if not self.client.collection_exists(collection_name):
            self.create_collection(collection_name)

    def create_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection using embedder settings from config."""
        distance = DISTANCE_MAP.get(settings.qdrant_distance, models.Distance.COSINE)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedder.get_sentence_embedding_dimension() or 384,
                    distance=distance,
                )
            },
            # sparse_vectors_config={
            #     "sparse": models.SparseVectorParams()
            # },
        )
        self.client.create_payload_index(
            collection_name,
            "uri",
            models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                on_disk=True,
                tokenizer=models.TokenizerType.PREFIX,
            ),
        )
        logger.info("Created Qdrant collection '%s' (distance=%s, sparse=%s)",
                    collection_name, settings.qdrant_distance, settings.sparse_embedder_model)

    def semantic_splitter(self, document: Document) -> list[TextNode]:
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=HuggingFaceEmbedding(model_name=settings.dense_embedder_model),
        )
        return splitter.get_nodes_from_documents([document])


    # Internal 

    def _upsert_point(
        self,
        collection_name: str,
        payload: dict,
        point_id: str | None = None,
    ) -> str:
        """Embed *text* and upsert a single point with the given payload dict."""
        text=payload["text"]
        if point_id is None:
            point_id = str(uuid_lib.uuid4())
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"dense": self.embedder.encode(text).tolist()},
                    #vector={"dense": self.embedder.encode(text).tolist(), "sparse": self._encode_sparse(text)},
                    payload=payload,
                )
            ],
        )
        return point_id

    def _retrieve_point(
        self,
        collection_name: str,
        point_id: str,
        uri: Optional[str] = None,
        point_type: Optional[str] = None,
        error: str = "Point not found",
    ) -> QdrantVector:
        """Retrieve a vector by point ID, with optional URI fallback.

        Tries *point_id* first; if that doesn't yield a result with the
        expected *point_type* (or any type when *point_type* is None),
        falls back to a URI scroll filtered by *point_type*.
        Raises HTTPException 404 if nothing is found.
        """
        try:
            pts = self.client.retrieve(
                collection_name=collection_name, ids=[point_id], with_payload=True
            )
            if pts:
                return _make_vector(0, str(pts[0].id), 0.0, pts[0].payload or {}, collection_name=collection_name)
 
        except Exception as exc:
            logger.warning("_retrieve_point failed for point_id=%s: %s", point_id, exc)

        if uri is not None:
            v = self.get_point_by_uri(collection_name=collection_name, uri=uri, point_type=point_type or "file")
            if v is not None:
                return v

        raise HTTPException(status_code=404, detail=error)

    def _delete_points_by_filter(self, filter: models.Filter, collection_name: str) -> None:
        """Delete all points whose payload URI matches."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=filter
            ),
        )



    # Queries

    def get_point_by_uri(self, uri: str,  collection_name: str, point_type: str = "file") -> Optional[QdrantVector]:
        """Return the typed vector for the first matching point, or None."""
        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="uri", match=models.MatchValue(value=uri)),
                        models.FieldCondition(key="type", match=models.MatchValue(value=point_type)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if points:
                return _make_vector(0, str(points[0].id), 0.0, points[0].payload or {}, collection_name=collection_name or "")
            return None
        except Exception as exc:
            logger.warning("get_point_by_uri failed (uri=%s, type=%s): %s", uri, point_type, exc)
            return None

    def get_all_uris(self, collection_name: str, point_type: Optional[str] = None) -> List[str]:
        """Return a deduplicated list of all URIs in the collection.

        Args:
            collection_name: Qdrant collection to query.
            point_type: Optional payload type filter (e.g. "file", "directory", "chunk").

        Returns:
            Sorted list of unique URI strings.
        """
        uris: set[str] = set()
        offset = None

        must = []
        if point_type:
            must.append(models.FieldCondition(key="type", match=models.MatchValue(value=point_type)))
        scroll_filter = models.Filter(must=must) if must else None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=["uri"],
            )
            for p in points:
                uri = (p.payload or {}).get("uri")
                if uri:
                    uris.add(uri)
            if next_offset is None:
                break
            offset = next_offset

        return sorted(uris)

    def get_point_descriptions(
        self, uris: List[str],  point_type: str, collection_name: str
    ) -> Dict[str, str]:
        """Return {uri: text} for points of the given type with URIs in the list."""
        if not uris:
            return {}
        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value=point_type)),
                        models.FieldCondition(key="uri", match=models.MatchAny(any=list(uris))),
                    ]
                ),
                limit=len(uris),
                with_payload=True,
            )
            vectors = [_make_vector(0, str(p.id), 0.0, p.payload or {}, collection_name=collection_name or "") for p in points]
            return {
                v.get_payload_field("uri"): v.get_payload_field("text") or ""
                for v in vectors
                if v.get_payload_field("uri")
            }
        except Exception as exc:
            logger.warning("get_point_descriptions failed (collection=%s, type=%s): %s", collection_name, point_type, exc)
            return {}


    # Upsert



    def upsert_file_point(
        self,
        collection_name: str,
        description: str,
        payload: FilePayload
    ) -> FileVector:
        """Set payload text to the summary description, embed, and upsert. Returns a FileVector."""
        payload.text = description
        id = self._upsert_point(collection_name=collection_name, payload=payload.to_dict())
        return FileVector(rank=0, point_id=id, score=0.0, collection_name=collection_name, payload=FilePayload(**payload.to_dict()))

    def upsert_chunk_points(self, collection_name: str, payload: FilePayload) -> None:
        """Split text into chunks, embed each, and upsert all to Qdrant."""

        uri = payload.uri
        metadata = {"type": "chunk", "uri": uri}

        document = Document(text=payload.text, metadata=metadata)

        chunks = self.semantic_splitter(document)
        chunk_total = len(chunks)

        points = [
            models.PointStruct(
                id=str(uuid_lib.uuid4()),
                vector={"dense": self.embedder.encode(chunk.text).tolist()},
               # vector={"dense": self.embedder.encode(chunk.text).tolist(), "sparse": self._encode_sparse(chunk.text)},
                payload={"text": chunk.text, "chunk_id": i, "chunk_total": chunk_total, **metadata},
            )
            for i, chunk in enumerate(chunks)
        ]
        if points:
            self.client.upsert(collection_name=collection_name, points=points)

    def upsert_directory_point(
        self,
        collection_name: str,
        uri: str,
        description: str,
        point_id: Optional[str] = None,
    ) -> DirectoryVector:
        """Embed a directory description and upsert a directory-level point. Returns a DirectoryVector."""
        payload = {"type": "directory", "text": description, "uri": uri}
        self._upsert_point(collection_name=collection_name, payload=payload)
        return DirectoryVector(rank=0, point_id=point_id or "", score=0.0, collection_name=collection_name, type="directory", payload=DirectoryPayload(**payload))

    # Deletion

    def delete_point_by_uri(self,  uri: str, collection_name: str) -> None:
        """Delete all points whose payload URI matches."""
        filter = models.Filter(
            must=[models.FieldCondition(key="uri", match=models.MatchValue(value=uri))]
        )
        self._delete_points_by_filter(filter, collection_name=collection_name)

    def delete_directory_by_uri(
        self,
        uri: str,
        collection_name: str
    ):

        from urllib.parse import urlparse

        parsed = urlparse(uri)
        parent_path = "/".join(parsed.path.rstrip("/").split("/")[:-1])
        dir_uri = f"{parsed.scheme}://{parsed.netloc}{parent_path}"

        filter = models.Filter(
            must=[
                models.FieldCondition(key="type", match=models.MatchValue(value="directory")),
                models.FieldCondition(key="uri", match=models.MatchText(text=uri))
                ]
        )

        self._delete_points_by_filter(filter, collection_name=collection_name)

    

    # Retrieval


    def get_point(self, collection_name: str, identifier: str) -> QdrantVector:
        """Retrieve any point by point ID or URI. Raises 404 if not found."""
        return self._retrieve_point(
            collection_name, identifier, uri=identifier
        )

    def get_file_vector(
        self,
        collection_name: str,
        point_id: Optional[str] = None,
        uri: Optional[str] = None,
        chunk: Optional[ChunkVector] = None,
    ) -> FileVector:
        """Retrieve a FileVector by point ID, URI, or ChunkVector (at least one required)."""
        if chunk is not None:
            lookup_id, lookup_uri = "", chunk.payload.uri
        elif point_id is not None:
            lookup_id, lookup_uri = point_id, uri
        elif uri is not None:
            lookup_id, lookup_uri = "", uri
        else:
            raise ValueError("One of point_id, uri, or chunk must be provided")
        v = self._retrieve_point(
            collection_name, lookup_id, uri=lookup_uri, point_type="file", error="FileVector not found"
        )
        if isinstance(v, FileVector):
            return v
        raise HTTPException(status_code=404, detail="FileVector not found")

    def get_directory_vector(self, collection_name: str, identifier: "str | FileVector") -> DirectoryVector:
        """Retrieve a DirectoryVector by point ID, URI, or from a FileVector's parent directory."""
        from urllib.parse import urlparse
        if isinstance(identifier, FileVector):
            parsed = urlparse(identifier.payload.uri)
            parent_path = "/".join(parsed.path.rstrip("/").split("/")[:-1])
            lookup_uri = f"{parsed.scheme}://{parsed.netloc}{parent_path}"
            point_id, uri = "", lookup_uri
        else:
            point_id, uri = identifier, identifier
        v = self._retrieve_point(
            collection_name, point_id, uri=uri, point_type="directory", error="DirectoryVector not found"
        )
        if isinstance(v, DirectoryVector):
            return v
        raise HTTPException(status_code=404, detail="DirectoryVector not found")

    def get_document(self, point_id: str) -> Dict[str, Any]:
        """Retrieve a single point by ID and return a content-focused dict."""
        v = self.get_point(settings.qdrant_collection_name, point_id)
        payload = v.payload if isinstance(v.payload, dict) else v.payload.model_dump()
        node = payload.get("node", {})
        return {
            "id": v.point_id,
            "content": node.get("text") if isinstance(node, dict) else payload.get("document"),
            "metadata": payload,
            "source": payload.get("uri"),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _get_vector_name(self, collection: str) -> Optional[str]:
        """Return the named vector to query against (auto-detected or from settings)."""
        if settings.qdrant_vector_name:
            return settings.qdrant_vector_name
        try:
            info = self.client.get_collection(collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict) and vectors:
                return next(iter(vectors))
        except Exception as exc:
            logger.warning("_get_vector_name failed for collection=%s: %s", collection, exc)
        return None

    def query(
        self,
        query_text: str,
        collection_name: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Embed query_text and run nearest-neighbour search, returning raw result dicts."""
        logger.info("query: encoding query for collection=%s", collection_name)
        query_vector = self.embedder.encode(query_text).tolist()

        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
            )

        logger.info("query: searching collection=%s limit=%d threshold=%s",
                    collection_name, limit, score_threshold)
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=self._get_vector_name(collection_name),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )
        logger.info("query: got %d points from %s", len(results.points), collection_name)
        return [{"id": p.id, "score": p.score, **p.payload} for p in results.points]

    def search(self, query: str, collection_name: str, limit: int = 5, point_type: str | None = None) -> List[QdrantVector]:
        """Search a collection and return ranked typed vector results."""
        collection_name = collection_name or settings.qdrant_collection_name
        filter_conditions = {"type": point_type} if point_type else None
        raw = self.query(query_text=query, collection_name=collection_name, limit=limit, filter_conditions=filter_conditions)
        return [
            _make_vector(i + 1, str(r["id"]), round(r["score"], 4),
                         {k: v for k, v in r.items() if k not in ("id", "score")}, collection_name=collection_name)
            for i, r in enumerate(raw)
        ]

    async def vector_search(
        self,
        query: str,
        collection_name: str = settings.qdrant_collection_name,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Semantic search returning a VectorSearchResponse-compatible dict."""
        logger.info("vector_search: collection=%s query=%r limit=%d", collection_name, query[:80], limit)
        vectors = self.search(query, limit=limit, collection_name=collection_name)
        if score_threshold > 0:
            vectors = [v for v in vectors if v.score >= score_threshold]
        logger.info("vector_search: got %d results from %s", len(vectors), collection_name)
        results = [
            {
                "id": v.point_id,
                "score": v.score,
                "content": v.get_payload_field("text"),
                "metadata": v.payload if isinstance(v.payload, dict) else v.payload.model_dump(),
            }
            for v in vectors
        ]
        return {"results": results, "total": len(results), "query": query}

    def fetch_chunks_as_text(self, uri: str, chunk_start: int, chunk_end: int) -> str:
        """Scroll chunks of uri in [chunk_start, chunk_end], join text, return as JSON."""
        all_chunks, _ = self.client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="uri", match=models.MatchValue(value=uri)),
                    models.FieldCondition(key="chunk_id", range=models.Range(gte=chunk_start, lte=chunk_end)),
                ]
            ),
            limit=chunk_end - chunk_start + 1,
            with_payload=True,
        )
        vectors = [_make_vector(0, str(p.id), 0.0, p.payload or {}, collection_name=settings.qdrant_collection_name) for p in all_chunks]
        sorted_vectors = sorted(vectors, key=lambda v: v.get_payload_field("chunk_id") or 0)
        combined_text = " ".join(v.get_payload_field("text") or "" for v in sorted_vectors)
        return json.dumps({"text": combined_text}, indent=2)

    # def fetch_neighbors(self, vectors: List[QdrantVector], window: int = 2, collection_name: str = settings.qdrant_collection_name) -> List[QdrantVector]:
    #     """Expand context by fetching neighboring chunks for each vector."""
    #     seen: set[str] = set()
    #     expanded: List[QdrantVector] = []
    #     rank = 1

    #     for vector in vectors:
    #         chunk_id = vector.get_payload_field("chunk_id")
    #         chunk_total = vector.get_payload_field("chunk_total")
    #         if chunk_id is None or chunk_total is None or chunk_total == 1:
    #             continue
    #         chunk_start = max(chunk_id - window, 0)
    #         chunk_end = min(chunk_id + window, chunk_total - 1)

    #         points, _ = self.client.scroll(
    #             collection_name=collection_name,
    #             scroll_filter=models.Filter(
    #                 must=[
    #                     models.FieldCondition(key="uri", match=models.MatchValue(value=vector.get_payload_field("uri"))),
    #                     models.FieldCondition(key="chunk_id", range=models.Range(gte=chunk_start, lte=chunk_end)),
    #                 ]
    #             ),
    #             limit=chunk_end - chunk_start + 1,
    #             with_payload=True,
    #         )
    #         for point in sorted(points, key=lambda p: (p.payload or {}).get("chunk_id", 0)):
    #             pid = str(point.id)
    #             if pid in seen:
    #                 continue
    #             seen.add(pid)
    #             expanded.append(_make_vector(rank, pid, 0.0, point.payload or {}, collection_name=collection_name))
    #             rank += 1

    #     return expanded


# Global service instance
qdrant_service = QdrantService()
