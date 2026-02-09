"""Local embedding via sentence-transformers (runs on CPU/GPU).

Requires: ``pip install 'memsearch[local]'``
No API key needed.
"""

from __future__ import annotations

import asyncio
from functools import partial


class LocalEmbedding:
    """sentence-transformers embedding provider."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(model)
        self._model = model
        self._dimension = self._st_model.get_sentence_embedding_dimension() or 384

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            partial(self._st_model.encode, texts, normalize_embeddings=True),
        )
        return embeddings.tolist()
