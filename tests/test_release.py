"""The publication guard must detect unsafe content without exposing its value."""

import importlib.util
import io
from pathlib import Path
import stat
import subprocess
import tarfile
import zipfile
import pytest

spec = importlib.util.spec_from_file_location(
    "check_release", Path(__file__).resolve().parents[1] / "tools" / "check_release.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "name,content,rule",
    [
        ("../escape", b"", "unsafe path"),
        (".env", b"", "private or generated file"),
        (".venv/lib/example.py", b"", "private or generated file"),
        ("key.txt", ("-----BEGIN " + "PRIVATE KEY-----").encode(), "private key"),
        ("config", ("ghp_" + "a" * 36).encode(), "GitHub credential"),
        ("data", ("/" + "home" + "/example/project").encode(), "user directory"),
        ("config", ("https://user" + ":password@example.org").encode(), "credential URL"),
    ],
)
def test_rejects_sensitive_or_unsafe_content(name, content, rule):
    assert rule in guard.inspect_content(name, content)


def test_public_content_and_archive_paths(tmp_path):
    assert not guard.inspect_content(
        "README.md", b"https://github.com/teeratornk/deflation-example"
    )
    assert not guard.inspect_content("provenance.json", b"a1b2c3d4" * 8)
    archive = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("/absolute.txt", "example")
    with pytest.raises(ValueError, match="unsafe path"):
        list(guard.members(archive))


@pytest.mark.parametrize(
    "content",
    [
        ("sk-proj-" + "a" * 32).encode(),
        ("hf_" + "a" * 32).encode(),
        ('{"api_key": "' + "a" * 32 + '"}').encode(),
        ("C:" + "\\Users\\" + "example").encode(),
        ("host=192." + "168.1.2").encode(),
        b"\x00\xff",
    ],
)
def test_additional_sensitive_content(content):
    assert guard.inspect_content("config", content)


@pytest.mark.parametrize("name", ["../escape", "C:escape", "C:/escape", "folder\\escape"])
def test_archive_paths_on_both_platforms(tmp_path, name):
    archive = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "example")
    with pytest.raises(ValueError, match="unsafe path"):
        list(guard.members(archive))


def test_archive_links(tmp_path):
    archive = tmp_path / "link.whl"
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(info, "target")
    with pytest.raises(ValueError, match="link or special"):
        list(guard.members(archive))


def test_version_numbers_are_not_network_endpoints():
    assert not guard.inspect_content("uv.lock", b'version = "10.3.9.90"')


def test_only_numeric_reviewed_mesh_bundles_are_allowed():
    import io
    import numpy as np
    data = {
        "nodes": np.zeros((4, 3)), "cells": np.array([[0, 1, 2, 3]]),
        "materials": np.array([0]), "dirichlet": np.array([0]), "axisymmetric": np.array(False),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    name = "src/deflation_example/data/engine_3d/mesh.npz"
    assert not guard.inspect_content(name, stream.getvalue())
    assert guard.inspect_content("arbitrary.npz", stream.getvalue())
    stream = io.BytesIO()
    np.savez_compressed(stream, **{**data, "nodes": np.array(["hidden text"])})
    assert "nonnumeric mesh array" in guard.inspect_content(name, stream.getvalue())
    stream = io.BytesIO()
    np.savez_compressed(stream, **{**data, "extra": np.array([1])})
    assert "unexpected mesh arrays" in guard.inspect_content(name, stream.getvalue())


def test_history_finds_a_removed_credential(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Example")
    git("config", "user.email", "example@example.invalid")
    path = tmp_path / "config.txt"
    path.write_text("ghp_" + "a" * 36)
    git("add", "config.txt")
    git("commit", "-qm", "Initial fixture")
    path.write_text("credential removed")
    git("add", "config.txt")
    git("commit", "-qm", "Remove fixture credential")
    assert not guard.inspect_content("config.txt", path.read_bytes())
    assert any(guard.inspect_content(name, data) for name, data in guard.history_records(tmp_path))


def test_archive_duplicates_and_size_limits(tmp_path, monkeypatch):
    archive = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("a", "one")
        with pytest.warns(UserWarning):
            stream.writestr("a", "two")
    with pytest.raises(ValueError, match="duplicate"):
        list(guard.members(archive))
    archive = tmp_path / "large.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("file")
        info.size = 4
        stream.addfile(info, io.BytesIO(b"text"))
    monkeypatch.setattr(guard, "MAX_FILE_BYTES", 2)
    with pytest.raises(ValueError, match="size limits"):
        list(guard.members(archive))
