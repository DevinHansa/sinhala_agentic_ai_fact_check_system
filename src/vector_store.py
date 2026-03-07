"""Qdrant vector store for semantic search."""
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer


class QdrantVectorStore:
    """Manage Qdrant vector database for fact-checking domains."""
    
    def __init__(self, storage_path: str = "./qdrant_data"):
        """Initialize Qdrant client and encoder."""
        import os
        self.storage_path = storage_path
        
        # Check for remote connection first
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        try:
            if url:
                self.qdrant = QdrantClient(url=url, api_key=api_key)
            else:
                self.qdrant = QdrantClient(path=storage_path)
        except RuntimeError as e:
            msg = str(e)
            if "already accessed by another instance" in msg:
                raise RuntimeError(
                    f"Qdrant local storage at '{storage_path}' is locked by another process. "
                    "Stop other Streamlit/Python processes using this folder, or set a different "
                    "QDRANT_PATH in your .env, or run a Qdrant server for concurrent access."
                ) from e
            raise
        self.encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.vector_size = 768
        
        # Initialize collections for each domain
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure all domain collections exist."""
        domains = ["politics", "economics", "health"]
        
        for domain in domains:
            collection_name = f"sinhala_{domain}"
            try:
                self.qdrant.get_collection(collection_name)
            except (RuntimeError, KeyError, ValueError):
                # Collection doesn't exist, create it
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                )
    
    def add_documents(self, documents: List[dict], domain: str):
        """Add documents to vector store."""
        collection_name = f"sinhala_{domain}"
        points = []
        
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text:
                continue
            
            vector = self.encoder.encode(text)
            points.append(
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={
                        "text": text,
                        "domain": domain,
                        "source": doc.get("source", ""),
                        "date": doc.get("date", "")
                    }
                )
            )
        
        if points:
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            return len(points)
        return 0
    
    def search(self, query: str, domain: str, limit: int = 5) -> List[dict]:
        """Search documents by semantic similarity."""
        collection_name = f"sinhala_{domain}"
        query_vector = self.encoder.encode(query)
        
        try:
            if hasattr(self.qdrant, "search"):
                results = self.qdrant.search(
                    collection_name=collection_name,
                    query_vector=query_vector.tolist(),
                    limit=limit,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="domain",
                                match=MatchValue(value=domain)
                            )
                        ]
                    )
                )
            else:
                # Compatibility path for qdrant-client builds that expose `query_points` instead of `search`.
                results = self.qdrant.query_points(
                    collection_name=collection_name,
                    query=query_vector.tolist(),
                    limit=limit,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="domain",
                                match=MatchValue(value=domain)
                            )
                        ]
                    ),
                    with_payload=True,
                    with_vectors=False,
                ).points
            
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": getattr(hit, "score", None),
                    "source": hit.payload.get("source", ""),
                    "date": hit.payload.get("date", "")
                }
                for hit in results
            ]
        except (RuntimeError, ValueError):
            return []
    
    def get_collection_stats(self, domain: str) -> dict:
        """Get statistics for a collection."""
        collection_name = f"sinhala_{domain}"
        try:
            info = self.qdrant.get_collection(collection_name)
            return {
                "domain": domain,
                "document_count": info.points_count
            }
        except Exception:
            return {"domain": domain, "document_count": 0}

    def close(self) -> None:
        """Close underlying Qdrant client resources (releases local storage lock)."""
        qdrant = getattr(self, "qdrant", None)
        if qdrant is None:
            return
        close_method = getattr(qdrant, "close", None)
        if callable(close_method):
            close_method()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Best-effort cleanup only.
            pass
