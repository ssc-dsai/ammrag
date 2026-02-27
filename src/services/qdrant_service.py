"""
Qdrant vector search service for FastAPI

Handles semantic similarity search and document retrieval
from the Qdrant vector database.
"""
from json import encoder
import logging
from typing import List, Optional, Dict, Any
from annotated_types import doc
from fastapi import HTTPException
import uuid as uuid_lib
from src.core.config import settings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from transformers import AutoTokenizer
import textwrap
from src.models.vector_config_model import VectorConfig
logger = logging.getLogger(__name__)

DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}

class EmbedderInfo:
    """Class to hold embedder and tokenizer information"""
    @property
    def embedder_name(self) -> str:
        return settings.embedder_model
    
    @property
    def embedder_token_limit(self) -> int:
        return 256

    @property
    def tokenizer_name(self) -> str:
        return settings.tokenizer_model
    
    @property
    def llm_name(self) -> str:
        return settings.ollama_model
    
    @ property
    def llm_token_limit(self) -> int:
        return 4096 
    
embedder_info = EmbedderInfo()

class QdrantService:
    """Service for Qdrant vector search operations"""


    def __init__(self):
        self._client = None
        self._embedder = None
        self._tokenizer = None
        self._check_connection()


    @property
    def client(self):
        if self._client is None:
            kwargs = {"url": settings.qdrant_url}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            self._client = QdrantClient(**kwargs)
        return self._client

    @property
    def embedder(self)-> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(settings.embedder_model)
        return self._embedder

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_model)
        return self._tokenizer
 
    def __semantic_splitter(self, text):
        document = Document(text=text)

        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        nodes = semantic_splitter.get_nodes_from_documents([document])  # Pass list of Document objects
        return [n.text for n in nodes]
    
    def _check_connection(self) -> None:
        """Verify Qdrant connectivity on startup."""
        try:
            collections = self.client.get_collections()
            logger.info("Qdrant connection OK (%s) — %d collection(s)",
                        settings.qdrant_url, len(collections.collections))
        except Exception as e:
            logger.error("Qdrant connection FAILED: %s", e)

    def create_collection(self, collection_name: str, vector_configs: list[VectorConfig]) -> None:
        """Create a Qdrant collection using the catalog's vector configuration."""
        vectors_config = {}
        sparse_vectors_config = {}

        for vector_config in vector_configs:
            if vector_config.type == "dense":
                distance = DISTANCE_MAP.get(vector_config.distance or "cosine", models.Distance.COSINE)
                vectors_config["dense"] = models.VectorParams(
                    size=self.embedder.get_sentence_embedding_dimension() or 384,
                    distance=distance,
                )
            elif vector_config.type == "sparse":
                sparse_vectors_config["sparse"] = models.SparseVectorParams()

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config or None,
        )
        logger.info("Created Qdrant collection '%s' (dense=%s, sparse=%s)",
                    collection_name,
                    "dense" in vectors_config,
                    "sparse" in sparse_vectors_config)

    def semantic_splitter(self, document: Document):
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=HuggingFaceEmbedding(model_name=settings.tokenizer_model)
        )
        return semantic_splitter.get_nodes_from_documents([document])  # Pass list of Document objects


    def delete_by_file_id(self, collection_name: str, file_id: int) -> None:
        """Delete all points in a collection whose payload contains the given file_id."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
        )

    def chunk_and_upsert(
        self,
        text: str,
        metadata: Dict[str, Any],
        collection_name: str,
    ):
        """
        Split text into chunks, embed each chunk, and upsert to Qdrant.

        Args:
            text: The full extracted text content
            metadata: Metadata to attach to every chunk (uri, uuid, etc.)
            collection_name: Qdrant collection (defaults to settings)

        Returns:
            List of Qdrant point IDs for the inserted chunks
        """
        collection = collection_name
        document = Document(text=text, metadata=metadata)
        points = []
        
        chunk_id = 0
        chunks = self.semantic_splitter(document)
        chunk_length = len(chunks)
        
        # Add document indexing to nodes
        for chunks in chunks:
            points.append(models.PointStruct(
                id=str(uuid_lib.uuid4()),
                vector={"dense": self.embedder.encode(chunks.text).tolist()},
                payload={"chunk": chunks.text, "chunk_id": chunk_id, "chunk_total": chunk_length, **metadata}
            ))
            chunk_id += 1

        if points:
            self.client.upsert(
                collection_name=collection,
                points=points,
            )

    async def vector_search(
        self,
        query: str,
        collection_name: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Semantic vector search returning a VectorSearchResponse-compatible dict.

        Args:
            query: The search query string
            collection_name: Qdrant collection to search (uses default if None)
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            Dict with keys: results, total, query
        """
        coll = collection_name or settings.qdrant_collection_name
        logger.info("vector_search: collection=%s query=%r limit=%d", coll, query[:80], limit)
        raw = self.query(
            query_text=query,
            collection_name=coll,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )
        logger.info("vector_search: got %d raw results from %s", len(raw), coll)
        results = []
        for r in raw:
            node = r.get("node", {})
            results.append({
                "id": str(r["id"]),
                "score": r["score"],
                "content": node.get("text") if isinstance(node, dict) else None,
                "metadata": {k: v for k, v in r.items() if k not in ("id", "score")},
            })
        return {"results": results, "total": len(results), "query": query}

    async def get_document(self, point_id: str) -> Dict[str, Any]:
        """
        Retrieve a single document point by its Qdrant ID.

        Args:
            point_id: The Qdrant point UUID string

        Returns:
            Dict with keys: id, content, metadata, source
        """
        try:
            points = self.client.retrieve(
                collection_name=settings.qdrant_collection_name,
                ids=[point_id],
                with_payload=True,
            )
        except Exception:
            points = []

        if not points:
            raise HTTPException(status_code=404, detail="Document not found")

        point = points[0]
        payload = point.payload or {}
        node = payload.get("node", {})
        return {
            "id": str(point.id),
            "content": node.get("text") if isinstance(node, dict) else payload.get("document"),
            "metadata": payload,
            "source": payload.get("uri"),
        }

    def query(
        self,
        query_text: str,
        collection_name: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nodes from Qdrant based on a text query.

        Embeds the query with the same model used for indexing, then
        performs a nearest-neighbour search on the dense vector.

        Args:
            query_text: The search query string
            collection_name: Qdrant collection to search
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (None = no threshold)
            filter_conditions: Optional payload filter as
                ``{"key": "value"}`` pairs (matched with MatchValue)

        Returns:
            List of dicts with keys: id, score, node, document, and
            all payload metadata fields.
        """
        logger.info("query: encoding query for collection=%s", collection_name)
        query_vector = self.embedder.encode(query_text).tolist()

        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                    for k, v in filter_conditions.items()
                ]
            )

        logger.info("query: searching collection=%s limit=%d threshold=%s",
                     collection_name, limit, score_threshold)
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        logger.info("query: got %d points from %s", len(results.points), collection_name)
        return [
            {
                "id": point.id,
                "score": point.score,
                **point.payload,
            }
            for point in results.points
        ]

# Global service instance
qdrant_service = QdrantService()
