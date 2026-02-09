"""MemSearch — main orchestrator class."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .watcher import FileWatcher

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
        Milvus connection URI.  A local ``*.db`` path uses Milvus Lite,
        ``http://host:port`` connects to a Milvus server, and a
        ``https://*.zillizcloud.com`` URL connects to Zilliz Cloud.
    milvus_token:
        Authentication token for Milvus server or Zilliz Cloud.
        Not needed for Milvus Lite (local).
    """

    def __init__(
        self,
        paths: list[str | Path] | None = None,
        *,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        milvus_uri: str = "~/.memsearch/milvus.db",
        milvus_token: str | None = None,
    ) -> None:
        self._paths = [str(p) for p in (paths or [])]
        self._embedder: EmbeddingProvider = get_provider(
            embedding_provider, model=embedding_model
        )
        self._store = MilvusStore(
            uri=milvus_uri, token=milvus_token, dimension=self._embedder.dimension
        )

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
            # Skip chunks whose hash already exists in Milvus
            all_hashes = [c.chunk_hash for c in chunks]
            existing = self._store.existing_hashes(all_hashes)
            chunks = [c for c in chunks if c.chunk_hash not in existing]
            if not chunks:
                return 0

        return await self._embed_and_store(chunks, doc_type="markdown")

    async def _embed_and_store(
        self, chunks: list[Chunk], *, doc_type: str = "markdown"
    ) -> int:
        if not chunks:
            return 0

        contents = [c.content for c in chunks]
        embeddings = await self._embedder.embed(contents)

        records: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_hash": chunk.chunk_hash,
                    "embedding": embeddings[i],
                    "content": chunk.content,
                    "source": chunk.source,
                    "heading": chunk.heading,
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
    # Watch
    # ------------------------------------------------------------------

    def watch(
        self,
        *,
        on_event: Callable[[str, str, Path], None] | None = None,
    ) -> FileWatcher:
        """Watch configured paths for markdown changes and auto-index.

        Starts a background thread that monitors the filesystem.  When a
        markdown file is created or modified it is re-indexed automatically;
        when deleted its chunks are removed from the store.

        Parameters
        ----------
        on_event:
            Optional callback invoked *after* each event is processed.
            Signature: ``(event_type, action_summary, file_path)``.
            ``event_type`` is ``"created"``, ``"modified"``, or ``"deleted"``.

        Returns
        -------
        FileWatcher
            The running watcher.  Call ``watcher.stop()`` when done, or
            use it as a context manager.

        Example
        -------
        ::

            ms = MemSearch(paths=["./docs/"])
            watcher = ms.watch()
            # ... watcher auto-indexes in background ...
            watcher.stop()
        """
        from .watcher import FileWatcher

        def _on_change(event_type: str, file_path: Path) -> None:
            if event_type == "deleted":
                self._store.delete_by_source(str(file_path))
                summary = f"Removed chunks for {file_path}"
            else:
                n = asyncio.run(self.index_file(file_path))
                summary = f"Indexed {n} chunks from {file_path}"
            logger.info(summary)
            if on_event is not None:
                on_event(event_type, summary, file_path)

        watcher = FileWatcher(self._paths, _on_change)
        watcher.start()
        return watcher

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def store(self) -> MilvusStore:
        return self._store

    def close(self) -> None:
        """Release resources."""
        self._store.close()

    def __enter__(self) -> MemSearch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
