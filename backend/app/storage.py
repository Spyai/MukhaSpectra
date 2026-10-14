"""Object storage for captures, analysis artifacts and reports.

Keys are POSIX-style relative paths such as ``screenings/<id>/results/v1/sites/x/heatmap.png``.
Every key is validated strictly before it touches the filesystem: relative, already normalised,
no ``..`` or ``.`` segments, no backslashes and only ``[A-Za-z0-9._/-]``. Anything else raises
``ValueError`` (the API turns that into 400/404), so user-supplied artifact keys can never escape
the storage root.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

MAX_KEY_LENGTH = 512
_KEY_CHARS = re.compile(r"[A-Za-z0-9._/-]+")


class ObjectNotFoundError(FileNotFoundError):
    """No object is stored under the requested key."""


def validate_key(key: str) -> str:
    """Return ``key`` unchanged if it is a safe storage key, else raise ``ValueError``."""
    if not isinstance(key, str) or not key:
        raise ValueError("Storage key must be a non-empty string.")
    if len(key) > MAX_KEY_LENGTH:
        raise ValueError(f"Storage key is longer than {MAX_KEY_LENGTH} characters.")
    if "\\" in key:
        raise ValueError("Storage key must not contain backslashes.")
    if key.startswith("/"):
        raise ValueError("Storage key must be relative.")
    if not _KEY_CHARS.fullmatch(key):
        raise ValueError("Storage key may only contain letters, digits, '.', '_', '-' and '/'.")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("Storage key must be normalised (no empty, '.' or '..' segments).")
    return key


def validate_prefix(prefix: str) -> str:
    """A prefix is a valid key, optionally followed by a single trailing '/'."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Storage prefix must be a non-empty string.")
    validate_key(prefix[:-1] if prefix.endswith("/") else prefix)
    return prefix


def join_key(*parts: str) -> str:
    """Join key segments with '/' and validate the result."""
    return validate_key("/".join(p.strip("/") for p in parts if p))


@runtime_checkable
class Storage(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def delete_prefix(self, prefix: str) -> int: ...

    def list_prefix(self, prefix: str) -> list[str]: ...


class LocalStorage:
    """Filesystem-backed storage rooted at ``root`` (development and single-node deployments)."""

    def __init__(self, root: Path | str) -> None:
        # The root is created lazily on the first write, so constructing a storage object (for
        # example when importing app.main) has no filesystem side effects.
        self.root = Path(root).expanduser().resolve()

    # -- helpers -------------------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = self.root.joinpath(*key.split("/"))
        # Defence in depth: refuse anything that resolves outside the root (e.g. via a symlink).
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Storage key resolves outside the storage root.")
        return path

    def _key_for(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # -- API -----------------------------------------------------------------------------------

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(bytes(data))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_prefix(self, prefix: str) -> list[str]:
        validate_prefix(prefix)
        base_rel = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        base = self._path(base_rel) if base_rel else self.root
        if not base.is_dir():
            return []
        keys: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                if name.startswith(".") and name.endswith(".tmp"):
                    continue
                key = self._key_for(Path(dirpath) / name)
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_prefix(prefix)
        for key in keys:
            self._path(key).unlink(missing_ok=True)
        self._prune_empty_dirs(prefix)
        return len(keys)

    def _prune_empty_dirs(self, prefix: str) -> None:
        if prefix.endswith("/"):
            base_rel = prefix.rstrip("/")
        else:
            base_rel = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        if not base_rel:
            return
        start = self._path(base_rel)
        if not start.is_dir():
            return
        for dirpath, _dirnames, _filenames in os.walk(start, topdown=False):
            path = Path(dirpath)
            if path != self.root and not any(path.iterdir()):
                path.rmdir()
