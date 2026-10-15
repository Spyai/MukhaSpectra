"""LocalStorage and strict key validation (path traversal must be impossible)."""

from __future__ import annotations

import os

import pytest

from app.storage import (
    LocalStorage,
    ObjectNotFoundError,
    Storage,
    join_key,
    validate_key,
    validate_prefix,
)

BAD_KEYS = [
    "",
    "/etc/passwd",
    "../secret.txt",
    "a/../../secret.txt",
    "a/..",
    "..",
    ".",
    "./a",
    "a/./b",
    "a//b",
    "a/",
    "a\\..\\b",
    "a b",
    "a%2F..%2Fb",
    "C:/windows",
    "caf\u00e9.png",
    "a\x00b",
    "~/x",
    "x" * 513,
]


@pytest.mark.parametrize("key", BAD_KEYS)
def test_invalid_keys_are_rejected(key, tmp_path):
    with pytest.raises(ValueError):
        validate_key(key)
    storage = LocalStorage(tmp_path / "root")
    for op in (storage.get, storage.exists):
        with pytest.raises(ValueError):
            op(key)
    with pytest.raises(ValueError):
        storage.put(key, b"x")


@pytest.mark.parametrize(
    "key", ["a", "screenings/abc/inputs/1.png", "a.b-c_d/E.F", "x/.hidden", "v1/...dots"]
)
def test_valid_keys(key):
    assert validate_key(key) == key


def test_prefix_validation():
    assert validate_prefix("screenings/abc/") == "screenings/abc/"
    assert validate_prefix("screenings/ab") == "screenings/ab"
    for bad in ("", "/", "../", "a//", "a/../"):
        with pytest.raises(ValueError):
            validate_prefix(bad)


def test_join_key():
    assert join_key("screenings", "abc", "results", "v1") == "screenings/abc/results/v1"
    with pytest.raises(ValueError):
        join_key("screenings", "..", "x")


def test_put_get_exists_list_delete(tmp_path):
    storage = LocalStorage(tmp_path / "root")
    assert isinstance(storage, Storage)
    assert not (tmp_path / "root").exists()  # created lazily
    assert storage.list_prefix("screenings/") == []
    storage.put("screenings/abc/inputs/1.png", b"one")
    storage.put("screenings/abc/results/v1/sites/x/heat.png", b"two")
    storage.put("screenings/abd/inputs/2.png", b"three")
    assert storage.get("screenings/abc/inputs/1.png") == b"one"
    assert storage.exists("screenings/abc/inputs/1.png")
    assert not storage.exists("screenings/abc/inputs/missing.png")
    assert storage.list_prefix("screenings/abc/") == [
        "screenings/abc/inputs/1.png",
        "screenings/abc/results/v1/sites/x/heat.png",
    ]
    # S3-style string prefix: "screenings/ab" matches both screenings
    assert len(storage.list_prefix("screenings/ab")) == 3
    storage.put("screenings/abc/inputs/1.png", b"replaced")
    assert storage.get("screenings/abc/inputs/1.png") == b"replaced"
    assert not [p for p in (tmp_path / "root").rglob("*") if p.name.endswith(".tmp")]

    assert storage.delete_prefix("screenings/abc/") == 2
    assert not (tmp_path / "root" / "screenings" / "abc").exists()  # empty dirs pruned
    assert storage.list_prefix("screenings/") == ["screenings/abd/inputs/2.png"]
    with pytest.raises(ObjectNotFoundError):
        storage.get("screenings/abc/inputs/1.png")
    assert storage.delete_prefix("nothing/here/") == 0
    assert storage.delete_prefix("screenings/abd/inputs/2") == 1


def test_symlink_escape_is_blocked(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    root = tmp_path / "root"
    (root / "screenings").mkdir(parents=True)
    os.symlink(outside, root / "screenings" / "link")
    storage = LocalStorage(root)
    with pytest.raises(ValueError):
        storage.get("screenings/link/secret.txt")
    with pytest.raises(ValueError):
        storage.put("screenings/link/new.txt", b"x")
    assert not (outside / "new.txt").exists()
