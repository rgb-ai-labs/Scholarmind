import httpx
import pytest

from scholarmind.discovery import arxiv as arxiv_module
from scholarmind.discovery.arxiv import search_arxiv
from scholarmind.discovery.models import DiscoverySourceError

_SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>  Attention Is All
      You Need  </title>
    <summary>  We propose a new architecture, the Transformer.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:doi>10.1234/attention</arxiv:doi>
    <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


def test_search_arxiv_parses_entry_fields(monkeypatch):
    monkeypatch.setattr(
        arxiv_module.httpx, "get", lambda *a, **kw: _FakeResponse(_SAMPLE_FEED)
    )

    [result] = search_arxiv("transformers", limit=5)

    assert result.title == "Attention Is All You Need"
    assert result.abstract == "We propose a new architecture, the Transformer."
    assert result.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert result.year == 2017
    assert result.doi == "10.1234/attention"
    assert result.venue == "arXiv"
    assert result.source == "arxiv"
    assert result.external_id == "1706.03762v5"
    assert result.pdf_url == "http://arxiv.org/pdf/1706.03762v5"


def test_search_arxiv_returns_empty_list_for_no_entries(monkeypatch):
    empty_feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    monkeypatch.setattr(arxiv_module.httpx, "get", lambda *a, **kw: _FakeResponse(empty_feed))

    assert search_arxiv("nothing matches this") == []


def test_search_arxiv_wraps_network_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(arxiv_module.httpx, "get", _raise)

    with pytest.raises(DiscoverySourceError) as excinfo:
        search_arxiv("anything")

    assert excinfo.value.source == "arxiv"


def test_search_arxiv_wraps_malformed_response(monkeypatch):
    monkeypatch.setattr(
        arxiv_module.httpx, "get", lambda *a, **kw: _FakeResponse("not valid xml <<<")
    )

    with pytest.raises(DiscoverySourceError):
        search_arxiv("anything")
