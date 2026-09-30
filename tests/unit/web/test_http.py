"""Tests for safe, non-browser HTML retrieval."""

from types import SimpleNamespace

import pytest

from researchflow.tools.web.http import SafeHttpClient, WebFetchError


class FakeResponse:
    def __init__(self, body: bytes, content_type: str, url: str) -> None:
        self._body = body
        self.headers = SimpleNamespace(get_content_type=lambda: content_type)
        self._url = url

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def public_resolver(host: str, port: int):
    return [(0, 0, 0, "", ("8.8.8.8", port))]


def test_fetch_accepts_public_html_and_extracts_title_and_text() -> None:
    client = SafeHttpClient(
        opener=lambda url, timeout: FakeResponse(
            b"<html><title>Example</title><body>Hello <b>world</b></body></html>",
            "text/html",
            url,
        ),
        resolver=public_resolver,
    )

    page = client.fetch("https://example.com/article")

    assert page.title == "Example"
    assert page.content == "Hello world"


def test_fetch_prefers_main_content_and_discards_page_chrome() -> None:
    client = SafeHttpClient(
        opener=lambda url, timeout: FakeResponse(
            (
                b"<html><head><title>Official guide</title>"
                b"<style>.ad{display:none}</style>"
                b'<script type="application/ld+json">{"name":"metadata"}</script>'
                b"</head><body><nav>Navigation links</nav>"
                b'<aside>Advertisement</aside><div class="advertisement">Buy now</div>'
                b"<main><article>"
                b"<h1>Python guide</h1><p>Official evidence belongs here.</p>"
                b"</article></main><footer>Copyright metadata</footer>"
                b"<script>window.tracker = true</script></body></html>"
            ),
            "text/html",
            url,
        ),
        resolver=public_resolver,
    )

    page = client.fetch("https://docs.python.org/guide")

    assert page.content == "Python guide Official evidence belongs here."
    assert "Navigation" not in page.content
    assert "Advertisement" not in page.content
    assert "Buy now" not in page.content
    assert "metadata" not in page.content


def test_fetch_discards_version_and_theme_switcher_noise() -> None:
    client = SafeHttpClient(
        opener=lambda url, timeout: FakeResponse(
            (
                b"<html><title>Python docs</title><body>"
                b'<div class="version-switcher">Python 3.13.12 documentation</div>'
                b'<div class="theme-switcher">Theme Light Dark</div>'
                b"<p>Python 3.13 adds supported free-threaded mode.</p>"
                b"</body></html>"
            ),
            "text/html",
            url,
        ),
        resolver=public_resolver,
    )

    page = client.fetch("https://docs.python.org/3.13/whatsnew/3.13.html")

    assert page.content == "Python 3.13 adds supported free-threaded mode."


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost/x"])
def test_fetch_rejects_unsafe_urls(url: str) -> None:
    client = SafeHttpClient(opener=lambda *_: None, resolver=public_resolver)

    with pytest.raises(WebFetchError, match="safe HTTP"):
        client.fetch(url)


def test_fetch_rejects_private_redirect_and_non_html_or_oversized_content() -> None:
    def redirecting_opener(url: str, timeout: float):
        return FakeResponse(b"ok", "text/html", "http://127.0.0.1/private")

    client = SafeHttpClient(opener=redirecting_opener, resolver=public_resolver)
    with pytest.raises(WebFetchError, match="safe HTTP"):
        client.fetch("https://example.com/redirect")

    non_html = SafeHttpClient(
        opener=lambda url, timeout: FakeResponse(b"%PDF", "application/pdf", url),
        resolver=public_resolver,
    )
    with pytest.raises(WebFetchError, match="HTML"):
        non_html.fetch("https://example.com/file")

    oversized = SafeHttpClient(
        opener=lambda url, timeout: FakeResponse(b"x" * 1025, "text/html", url),
        resolver=public_resolver,
        max_bytes=1024,
    )
    with pytest.raises(WebFetchError, match="too large"):
        oversized.fetch("https://example.com/large")
