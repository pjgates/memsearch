"""MemSearch — main orchestrator class."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .cache import EmbeddingCache
from .chunker import Chunk, chunk_markdown
from .embeddings import EmbeddingProvider, get_provider
from .flush import flush_chunks
from .scanner import ScannedFile, scan_paths
from .session import Session, parse_session_file
from .store import MilvusStore

logger = logging.getLogger(__name__)


class MemSearch:
    """High-level API for semantic memory search.

    Parameters
    ----------
    paths:
        Directories / files to index.
    embedding_provider:
        Name of the embedding backend (``"openai"``, ``"google"``, etc.).
    embedding_model:
        Override the default model for the chosen provider.
    milvus_uri:
        Milvus connection URI (defaults to local Milvus Lite file).
    cache_path:
        Path to the SQLite embedding cache.
    """

    def __init__(
        self,
        paths: list[str | Path] | None = None,
        *,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        milvus_uri: str = "~/.memsearch/milvus.db",
        cache_path: str = "~/.memsearch/cache.db",
    ) -> None:
        self._paths = [str(p) for p in (paths or [])]
        self._embedder: EmbeddingProvider = get_provider(
            embedding_provider, model=embedding_model
        )
        self._store = MilvusStore(uri=milvus_uri, dimension=self._embedder.dimension)
        self._cache = EmbeddingCache(db_path=cache_path)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index(self, *, force: bool = False) -> int:
        """Scan paths and index all markdown files.

        Returns the number of chunks indexed.
        """
        files = scan_paths(self._paths)
        total = 0
        for f in files:
            n = await self._index_file(f, force=force)
            total += n
        logger.info("Indexed %d chunks from %d files", total, len(files))
        return total

    async def index_file(self, path: str | Path) -> int:
        """Index a single file.  Returns number of chunks."""
        p = Path(path).expanduser().resolve()
        sf = ScannedFile(path=p, mtime=p.stat().st_mtime, size=p.stat().st_size)
        return await self._index_file(sf)

    async def index_session(self, path: str | Path) -> int:
        """Parse and index a JSONL session log."""
        sessions = parse_session_file(path)
        total = 0
        for session in sessions:
            md = session.to_markdown()
            chunks = chunk_markdown(md, source=str(path))
            n = await self._embed_and_store(chunks, doc_type="session")
            total += n
        logger.info("Indexed %d chunks from session %s", total, path)
        return total

    async def _index_file(self, f: ScannedFile, *, force: bool = False) -> int:
        text = f.path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=str(f.path))
        if not chunks:
            return 0

        if not force:
            # Skip chunks already in store (by hash)
            existing_hashes = {c.chunk_hash for c in chunks}
            # We just re-index all for simplicity; the store upserts by hash
            pass

        return await self._embed_and_store(chunks, doc_type="markdown")

    async def _embed_and_store(
        self, chunks: list[Chunk], *, doc_type: str = "markdown"
    ) -> int:
        if not chunks:
            return 0

        model = self._embedder.model_name
        hashes = [c.chunk_hash for c in chunks]
        contents = [c.content for c in chunks]

        # Check cache
        cached = self._cache.get_batch(hashes, model)
        to_embed_indices = [i for i, h in enumerate(hashes) if cached[h] is None]

        if to_embed_indices:
            texts_to_embed = [contents[i] for i in to_embed_indices]
            new_embeddings = await self._embedder.embed(texts_to_embed)
            # Populate cache
            cache_items = [
                (hashes[to_embed_indices[j]], model, emb)
                for j, emb in enumerate(new_embeddings)
            ]
            self._cache.put_batch(cache_items)
            # Merge into result
            for j, idx in enumerate(to_embed_indices):
                cached[hashes[idx]] = new_embeddings[j]

        # Build store records
        records: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            emb = cached[chunk.chunk_hash]
            if emb is None:
                continue  # should not happen
            records.append(
                {
                    "embedding": emb,
                    "content": chunk.content,
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "chunk_hash": chunk.chunk_hash,
                    "heading_level": chunk.heading_level,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "doc_type": doc_type,
                }
            )

        return self._store.upsert(records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across indexed chunks.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Maximum results to return.
        doc_type:
            Filter by document type (``"markdown"``, ``"session"``, ``"flush"``).

        Returns
        -------
        list[dict]
            Each dict contains ``content``, ``source``, ``heading``,
            ``score``, and other metadata.
        """
        embeddings = await self._embedder.embed([query])
        filter_expr = f'doc_type == "{doc_type}"' if doc_type else ""
        return self._store.search(
            embeddings[0], top_k=top_k, filter_expr=filter_expr
        )

    # ------------------------------------------------------------------
    # Flush (compress memories)
    # ------------------------------------------------------------------

    async def flush(
        self,
        *,
        source: str | None = None,
        llm_provider: str = "openai",
        llm_model: str | None = None,
    ) -> str:
        """Compress indexed chunks into a summary and re-index it.

        Parameters
        ----------
        source:
            If given, only flush chunks from this source file.
        llm_provider:
            LLM backend for summarization.
        llm_model:
            Override the default model.

        Returns
        -------
        str
            The generated summary markdown.
        """
        filter_expr = f'source == "{source}"' if source else ""
        # Retrieve all chunks (use a dummy search with high top_k)
        # For a proper implementation we'd iterate, but this is practical
        dummy_emb = [0.0] * self._embedder.dimension
        all_chunks = self._store.search(
            dummy_emb, top_k=10000, filter_expr=filter_expr
        )
        if not all_chunks:
            return ""

        summary = await flush_chunks(
            all_chunks, llm_provider=llm_provider, model=llm_model
        )

        # Index the summary as a new "flush" document
        flush_chunks_list = chunk_markdown(summary, source="flush://memory")
        await self._embed_and_store(flush_chunks_list, doc_type="flush")

        logger.info("Flushed %d chunks into summary", len(all_chunks))
        return summary

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def store(self) -> MilvusStore:
        return self._store

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    def close(self) -> None:
        """Release resources."""
        self._store.close()
        self._cache.close()

    def __enter__(self) -> MemSearch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
