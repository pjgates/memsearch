"""SQLite-based embedding cache.

Caches embeddings keyed by (content_hash, model_name) to avoid
re-computing embeddings for unchanged content.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class EmbeddingCache:
    """Persistent embedding cache backed by SQLite."""

    def __init__(self, db_path: str | Path = "~/.memsearch/cache.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                embedding    TEXT NOT NULL,
                created_at   REAL NOT NULL DEFAULT (unixepoch('now')),
                PRIMARY KEY (content_hash, model)
            )
            """
        )
        self._conn.commit()

    def get(self, content_hash: str, model: str) -> list[float] | None:
        """Return cached embedding or ``None``."""
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE content_hash = ? AND model = ?",
            (content_hash, model),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_batch(
        self, hashes: list[str], model: str
    ) -> dict[str, list[float] | None]:
        """Look up multiple hashes at once; returns {hash: embedding_or_None}."""
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        rows = self._conn.execute(
            f"SELECT content_hash, embedding FROM embeddings "
            f"WHERE model = ? AND content_hash IN ({placeholders})",
            [model, *hashes],
        ).fetchall()
        found = {row[0]: json.loads(row[1]) for row in rows}
        return {h: found.get(h) for h in hashes}

    def put(self, content_hash: str, model: str, embedding: list[float]) -> None:
        """Store an embedding."""
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (content_hash, model, embedding) "
            "VALUES (?, ?, ?)",
            (content_hash, model, json.dumps(embedding)),
        )
        self._conn.commit()

    def put_batch(
        self, items: list[tuple[str, str, list[float]]]
    ) -> None:
        """Store multiple (content_hash, model, embedding) tuples."""
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (content_hash, model, embedding) "
            "VALUES (?, ?, ?)",
            [(h, m, json.dumps(e)) for h, m, e in items],
        )
        self._conn.commit()

    def clear(self, *, model: str | None = None) -> int:
        """Delete cached entries. If *model* is given, only delete for that model."""
        if model:
            cur = self._conn.execute(
                "DELETE FROM embeddings WHERE model = ?", (model,)
            )
        else:
            cur = self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
