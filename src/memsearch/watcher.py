"""File watcher — monitors directories for markdown changes using watchdog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _MarkdownHandler(FileSystemEventHandler):
    """Dispatch markdown file events to a callback."""

    def __init__(
        self,
        callback: Callable[[str, Path], None],
        extensions: tuple[str, ...] = (".md", ".markdown"),
    ) -> None:
        self._callback = callback
        self._extensions = extensions

    def _is_markdown(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._extensions

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            logger.debug("File created: %s", event.src_path)
            self._callback("created", Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            logger.debug("File modified: %s", event.src_path)
            self._callback("modified", Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_markdown(event.src_path):
            logger.debug("File deleted: %s", event.src_path)
            self._callback("deleted", Path(event.src_path))


class FileWatcher:
    """Watch directories for markdown file changes.

    Parameters
    ----------
    paths:
        Directories to watch.
    callback:
        Called with ``(event_type, file_path)`` on change.
        ``event_type`` is one of ``"created"``, ``"modified"``, ``"deleted"``.
    """

    def __init__(
        self,
        paths: list[str | Path],
        callback: Callable[[str, Path], None],
    ) -> None:
        self._paths = [Path(p).expanduser().resolve() for p in paths]
        self._handler = _MarkdownHandler(callback)
        self._observer = Observer()

    def start(self) -> None:
        """Start watching in a background thread."""
        for p in self._paths:
            if p.is_dir():
                self._observer.schedule(self._handler, str(p), recursive=True)
                logger.info("Watching %s", p)
        self._observer.start()

    def stop(self) -> None:
        """Stop watching."""
        self._observer.stop()
        self._observer.join()

    def __enter__(self) -> FileWatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
