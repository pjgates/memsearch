"""Voyage AI embedding provider.

Requires: ``pip install 'memsearch[voyage]'``
Environment variables:
    VOYAGE_API_KEY — required
"""

from __future__ import annotations


class VoyageEmbedding:
    """Voyage AI embedding provider."""

    def __init__(self, model: str = "voyage-3-lite") -> None:
        import voyageai

        self._client = voyageai.AsyncClient()  # reads VOYAGE_API_KEY
        self._model = model
        self._dimension = _default_dimension(model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model)
        return result.embeddings


def _default_dimension(model: str) -> int:
    dimensions: dict[str, int] = {
        "voyage-3-lite": 512,
        "voyage-3": 1024,
        "voyage-code-3": 1024,
    }
    return dimensions.get(model, 1024)
