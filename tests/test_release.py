"""The publication guard must detect unsafe content without exposing its value."""

import importlib.util
from pathlib import Path
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


def test_allows_public_links_and_hashes():
    assert not guard.inspect_content(
        "README.md", b"https://github.com/teeratornk/deflation-example"
    )
    assert not guard.inspect_content("provenance.json", b"a1b2c3d4" * 8)
