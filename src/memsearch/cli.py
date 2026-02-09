"""CLI interface for memsearch."""

from __future__ import annotations

import asyncio
import json
import sys

import click


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@click.group()
@click.version_option(package_name="memsearch")
def cli() -> None:
    """memsearch — semantic memory search for markdown knowledge bases."""


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--provider", "-p", default="openai", help="Embedding provider.")
@click.option("--model", "-m", default=None, help="Override embedding model.")
@click.option("--force", is_flag=True, help="Re-index all files.")
def index(paths: tuple[str, ...], provider: str, model: str | None, force: bool) -> None:
    """Index markdown files from PATHS."""
    from .core import MemSearch

    ms = MemSearch(list(paths), embedding_provider=provider, embedding_model=model)
    try:
        n = _run(ms.index(force=force))
        click.echo(f"Indexed {n} chunks.")
    finally:
        ms.close()


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results.")
@click.option("--provider", "-p", default="openai", help="Embedding provider.")
@click.option("--model", "-m", default=None, help="Override embedding model.")
@click.option("--doc-type", "-t", default=None, help="Filter by doc type.")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def search(
    query: str,
    top_k: int,
    provider: str,
    model: str | None,
    doc_type: str | None,
    json_output: bool,
) -> None:
    """Search indexed memory for QUERY."""
    from .core import MemSearch

    ms = MemSearch(embedding_provider=provider, embedding_model=model)
    try:
        results = _run(ms.search(query, top_k=top_k, doc_type=doc_type))
        if json_output:
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            if not results:
                click.echo("No results found.")
                return
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                source = r.get("source", "?")
                heading = r.get("heading", "")
                content = r.get("content", "")
                click.echo(f"\n--- Result {i} (score: {score:.4f}) ---")
                click.echo(f"Source: {source}")
                if heading:
                    click.echo(f"Heading: {heading}")
                click.echo(content[:500])
    finally:
        ms.close()


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--provider", "-p", default="openai", help="Embedding provider.")
@click.option("--model", "-m", default=None, help="Override embedding model.")
def watch(paths: tuple[str, ...], provider: str, model: str | None) -> None:
    """Watch PATHS for markdown changes and auto-index."""
    from .core import MemSearch
    from .watcher import FileWatcher

    ms = MemSearch(list(paths), embedding_provider=provider, embedding_model=model)

    def on_change(event_type: str, file_path) -> None:
        if event_type == "deleted":
            ms.store.delete_by_source(str(file_path))
            click.echo(f"Removed: {file_path}")
        else:
            n = _run(ms.index_file(file_path))
            click.echo(f"Indexed {n} chunks: {file_path}")

    watcher = FileWatcher(list(paths), on_change)
    click.echo(f"Watching {len(paths)} path(s) for changes... (Ctrl+C to stop)")
    try:
        watcher.start()
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher.")
    finally:
        watcher.stop()
        ms.close()


@cli.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--provider", "-p", default="openai", help="Embedding provider.")
@click.option("--model", "-m", default=None, help="Override embedding model.")
def ingest_session(session_path: str, provider: str, model: str | None) -> None:
    """Ingest a Claude JSONL session log."""
    from .core import MemSearch

    ms = MemSearch(embedding_provider=provider, embedding_model=model)
    try:
        n = _run(ms.index_session(session_path))
        click.echo(f"Indexed {n} chunks from session log.")
    finally:
        ms.close()


@cli.command()
@click.option("--source", "-s", default=None, help="Only flush chunks from this source.")
@click.option("--llm-provider", default="openai", help="LLM for summarization.")
@click.option("--llm-model", default=None, help="Override LLM model.")
@click.option("--provider", "-p", default="openai", help="Embedding provider.")
@click.option("--model", "-m", default=None, help="Override embedding model.")
def flush(
    source: str | None,
    llm_provider: str,
    llm_model: str | None,
    provider: str,
    model: str | None,
) -> None:
    """Compress stored memories into a summary."""
    from .core import MemSearch

    ms = MemSearch(embedding_provider=provider, embedding_model=model)
    try:
        summary = _run(ms.flush(source=source, llm_provider=llm_provider, llm_model=llm_model))
        if summary:
            click.echo("Flush complete. Summary:\n")
            click.echo(summary)
        else:
            click.echo("No chunks to flush.")
    finally:
        ms.close()


@cli.command()
def stats() -> None:
    """Show statistics about the index."""
    from .store import MilvusStore

    store = MilvusStore()
    try:
        count = store.count()
        click.echo(f"Total indexed chunks: {count}")
    finally:
        store.close()


@cli.command()
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset() -> None:
    """Drop all indexed data."""
    from .cache import EmbeddingCache
    from .store import MilvusStore

    store = MilvusStore()
    cache = EmbeddingCache()
    try:
        store.drop()
        cleared = cache.clear()
        click.echo(f"Dropped collection. Cleared {cleared} cached embeddings.")
    finally:
        store.close()
        cache.close()
