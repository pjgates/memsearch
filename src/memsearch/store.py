"""Milvus vector storage layer using MilvusClient API."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MilvusStore:
    """Thin wrapper around ``pymilvus.MilvusClient`` for chunk storage."""

    COLLECTION = "memsearch_chunks"

    def __init__(
        self,
        uri: str = "~/.memsearch/milvus.db",
        *,
        dimension: int = 1536,
    ) -> None:
        from pymilvus import MilvusClient

        resolved = str(Path(uri).expanduser()) if not uri.startswith(("http", "tcp")) else uri
        Path(resolved).parent.mkdir(parents=True, exist_ok=True) if not uri.startswith(("http", "tcp")) else None
        self._client = MilvusClient(uri=resolved)
        self._dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.COLLECTION):
            return
        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=5, is_primary=True)  # INT64
        schema.add_field(
            field_name="embedding",
            datatype=101,  # FLOAT_VECTOR
            dim=self._dimension,
        )
        schema.add_field(field_name="content", datatype=21, max_length=65535)  # VARCHAR
        schema.add_field(field_name="source", datatype=21, max_length=1024)  # VARCHAR
        schema.add_field(field_name="heading", datatype=21, max_length=1024)  # VARCHAR
        schema.add_field(field_name="chunk_hash", datatype=21, max_length=64)  # VARCHAR
        schema.add_field(field_name="heading_level", datatype=5)  # INT64
        schema.add_field(field_name="start_line", datatype=5)  # INT64
        schema.add_field(field_name="end_line", datatype=5)  # INT64
        schema.add_field(field_name="doc_type", datatype=21, max_length=64)  # VARCHAR

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self.COLLECTION,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or update chunks.

        Each dict must contain at minimum: ``embedding``, ``content``,
        ``source``, ``chunk_hash``.  Additional fields are stored as-is.
        """
        if not chunks:
            return 0
        # Remove existing chunks with same hashes to achieve upsert
        hashes = [c["chunk_hash"] for c in chunks]
        self.delete_by_hashes(hashes)
        result = self._client.insert(
            collection_name=self.COLLECTION,
            data=chunks,
        )
        return result.get("insert_count", len(chunks)) if isinstance(result, dict) else len(chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """Semantic search returning top-k results."""
        kwargs: dict[str, Any] = {
            "collection_name": self.COLLECTION,
            "data": [query_embedding],
            "limit": top_k,
            "output_fields": [
                "content", "source", "heading", "chunk_hash",
                "heading_level", "start_line", "end_line", "doc_type",
            ],
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        results = self._client.search(**kwargs)
        if not results or not results[0]:
            return []
        return [
            {**hit["entity"], "score": hit["distance"]}
            for hit in results[0]
        ]

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        self._client.delete(
            collection_name=self.COLLECTION,
            filter=f'source == "{source}"',
        )

    def delete_by_hashes(self, hashes: list[str]) -> None:
        """Delete chunks by their content hashes."""
        if not hashes:
            return
        hash_list = ", ".join(f'"{h}"' for h in hashes)
        self._client.delete(
            collection_name=self.COLLECTION,
            filter=f"chunk_hash in [{hash_list}]",
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        stats = self._client.get_collection_stats(self.COLLECTION)
        return stats.get("row_count", 0)

    def drop(self) -> None:
        """Drop the entire collection."""
        if self._client.has_collection(self.COLLECTION):
            self._client.drop_collection(self.COLLECTION)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MilvusStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
