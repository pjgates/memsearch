"""Tests for the embedding cache."""

from pathlib import Path

from memsearch.cache import EmbeddingCache


def test_put_and_get(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        cache.put("hash1", "model-a", [1.0, 2.0, 3.0])
        result = cache.get("hash1", "model-a")
        assert result == [1.0, 2.0, 3.0]


def test_get_missing(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        assert cache.get("nonexistent", "model-a") is None


def test_different_models(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        cache.put("hash1", "model-a", [1.0])
        cache.put("hash1", "model-b", [2.0])
        assert cache.get("hash1", "model-a") == [1.0]
        assert cache.get("hash1", "model-b") == [2.0]


def test_batch_operations(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        cache.put_batch([
            ("h1", "m", [1.0]),
            ("h2", "m", [2.0]),
            ("h3", "m", [3.0]),
        ])
        results = cache.get_batch(["h1", "h2", "h3", "h4"], "m")
        assert results["h1"] == [1.0]
        assert results["h2"] == [2.0]
        assert results["h3"] == [3.0]
        assert results["h4"] is None


def test_clear(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        cache.put("h1", "m1", [1.0])
        cache.put("h2", "m2", [2.0])
        cleared = cache.clear(model="m1")
        assert cleared == 1
        assert cache.get("h1", "m1") is None
        assert cache.get("h2", "m2") == [2.0]


def test_upsert_overwrites(tmp_path: Path):
    db = tmp_path / "cache.db"
    with EmbeddingCache(db) as cache:
        cache.put("h1", "m", [1.0])
        cache.put("h1", "m", [9.0])
        assert cache.get("h1", "m") == [9.0]
