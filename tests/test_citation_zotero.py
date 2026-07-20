import httpx
import pytest

from scholarmind.citations import zotero as zotero_module
from scholarmind.citations.metadata import NormalizedMetadata
from scholarmind.citations.zotero import ZoteroError, push_references


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self) -> dict:
        return self._payload


def _metadata(**overrides) -> NormalizedMetadata:
    defaults = dict(
        doi="10.1234/attention",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
        venue="NeurIPS",
        source="library",
    )
    defaults.update(overrides)
    return NormalizedMetadata(**defaults)


def test_push_references_without_config_raises_readable_error():
    with pytest.raises(ZoteroError, match="not configured"):
        push_references([_metadata()], api_key="", library_id="")


def test_push_references_empty_list_is_a_no_op():
    result = push_references([], api_key="key", library_id="123")
    assert result.pushed == 0
    assert result.failed == 0


def test_push_references_reports_success_and_failure_counts(monkeypatch):
    monkeypatch.setattr(
        zotero_module.httpx,
        "post",
        lambda *a, **kw: _FakeResponse({"successful": {"0": {}}, "failed": {"1": {"message": "bad item"}}}),
    )

    result = push_references([_metadata(), _metadata(title="Other")], api_key="key", library_id="123")

    assert result.pushed == 1
    assert result.failed == 1
    assert result.errors == ["bad item"]


def test_push_references_403_raises_readable_error(monkeypatch):
    monkeypatch.setattr(
        zotero_module.httpx, "post", lambda *a, **kw: _FakeResponse({}, status_code=403)
    )

    with pytest.raises(ZoteroError, match="403"):
        push_references([_metadata()], api_key="bad-key", library_id="123")


def test_push_references_429_raises_readable_error(monkeypatch):
    monkeypatch.setattr(
        zotero_module.httpx, "post", lambda *a, **kw: _FakeResponse({}, status_code=429)
    )

    with pytest.raises(ZoteroError, match="rate limited"):
        push_references([_metadata()], api_key="key", library_id="123")


def test_push_references_network_error_raises_readable_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(zotero_module.httpx, "post", _raise)

    with pytest.raises(ZoteroError):
        push_references([_metadata()], api_key="key", library_id="123")


def test_push_references_uses_group_url_for_group_library(monkeypatch):
    captured = {}

    def _fake_post(url, **kwargs):
        captured["url"] = url
        return _FakeResponse({"successful": {"0": {}}, "failed": {}})

    monkeypatch.setattr(zotero_module.httpx, "post", _fake_post)

    push_references([_metadata()], api_key="key", library_id="456", library_type="group")

    assert captured["url"] == "https://api.zotero.org/groups/456/items"


def test_push_references_batches_over_fifty_items(monkeypatch):
    calls = []

    def _fake_post(url, json, **kwargs):
        calls.append(len(json))
        return _FakeResponse({"successful": {str(i): {} for i in range(len(json))}, "failed": {}})

    monkeypatch.setattr(zotero_module.httpx, "post", _fake_post)

    references = [_metadata(title=f"Paper {i}") for i in range(120)]
    result = push_references(references, api_key="key", library_id="123")

    assert calls == [50, 50, 20]
    assert result.pushed == 120
