"""Safe, bounded retrieval of ordinary public HTML pages."""

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener


class WebFetchError(RuntimeError):
    """Safe failure raised for rejected or unreadable web pages."""

    def __init__(self, message: str, error_type: str = "web_fetch_failed") -> None:
        super().__init__(message)
        self.error_type = error_type


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> object:
        raise WebFetchError("web redirects are not supported", "web_redirect_rejected")


def _default_opener(url: str, timeout: float) -> object:
    return build_opener(_RejectRedirect()).open(url, timeout=timeout)


@dataclass(frozen=True, slots=True)
class HtmlPage:
    title: str
    content: str
    url: str


class _TextExtractor(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "template", "svg"}
    _CHROME_TAGS = {"nav", "aside", "header", "footer", "form"}
    _PRIMARY_TAGS = {"main", "article"}
    _CHROME_TOKENS = {
        "ad",
        "ads",
        "advert",
        "advertisement",
        "banner",
        "cookie",
        "theme",
        "switcher",
        "version",
    }

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._title: list[str] = []
        self._fallback_text: list[str] = []
        self._primary_text: list[str] = []
        self._ignored_depth = 0
        self._chrome_depth = 0
        self._chrome_marker_tags: list[str] = []
        self._primary_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        self._in_title = normalized == "title"
        if normalized in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif normalized in self._CHROME_TAGS or self._has_chrome_marker(attrs):
            self._chrome_depth += 1
            self._chrome_marker_tags.append(normalized)
        elif normalized in self._PRIMARY_TAGS:
            self._primary_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = False
        if normalized in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._chrome_marker_tags and self._chrome_marker_tags[-1] == normalized:
            self._chrome_marker_tags.pop()
            self._chrome_depth -= 1
        elif normalized in self._PRIMARY_TAGS and self._primary_depth:
            self._primary_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title.append(data)
        elif not self._ignored_depth and not self._chrome_depth:
            target = self._primary_text if self._primary_depth else self._fallback_text
            target.append(data)

    @classmethod
    def _has_chrome_marker(cls, attrs: list[tuple[str, str | None]]) -> bool:
        for name, value in attrs:
            if name.lower() not in {"class", "id", "role"} or value is None:
                continue
            tokens = {token.casefold() for token in value.replace("-", " ").split()}
            if (
                tokens & cls._CHROME_TOKENS
                or {"navigation", "menu", "sidebar"} & tokens
            ):
                return True
        return False

    @property
    def title(self) -> str:
        return " ".join("".join(self._title).split()) or "Untitled page"

    @property
    def content(self) -> str:
        text = self._primary_text or self._fallback_text
        return " ".join(" ".join(text).split())


class SafeHttpClient:
    """Fetch a small HTML document after validating public URL endpoints."""

    def __init__(
        self,
        *,
        opener: Callable[..., object] = _default_opener,
        resolver: Callable[..., object] = socket.getaddrinfo,
        timeout: float = 10,
        max_bytes: int = 1_000_000,
    ) -> None:
        self._opener = opener
        self._resolver = resolver
        self._timeout = timeout
        self._max_bytes = max_bytes

    def fetch(self, url: str) -> HtmlPage:
        """Fetch public HTML, rejecting unsafe endpoints and unsupported content."""
        self._validate_url(url)
        try:
            with self._opener(url, timeout=self._timeout) as response:
                final_url = response.geturl()
                self._validate_url(final_url)
                content_type = response.headers.get_content_type()
                if content_type != "text/html":
                    raise WebFetchError("web response is not HTML", "web_content_type")
                charset = response.headers.get_content_charset()
                body = response.read(self._max_bytes + 1)
        except WebFetchError:
            raise
        except TimeoutError as exc:
            raise WebFetchError("web page request timed out", "web_timeout") from exc
        except HTTPError as exc:
            raise WebFetchError(
                "web page returned an HTTP error", "web_http_status"
            ) from exc
        except OSError as exc:
            raise WebFetchError("web page request failed", "web_network_error") from exc
        if len(body) > self._max_bytes:
            raise WebFetchError("web response is too large", "web_response_too_large")
        try:
            parser = _TextExtractor()
            parser.feed(self._decode_html(body, charset))
        except (ValueError, UnicodeError) as exc:
            raise WebFetchError(
                "web page could not be read", "web_content_decode"
            ) from exc
        if not parser.content:
            raise WebFetchError("web page has no readable text", "web_empty_content")
        return HtmlPage(title=parser.title, content=parser.content, url=final_url)

    @staticmethod
    def _decode_html(body: bytes, header_charset: str | None) -> str:
        """Decode header, then HTML declaration, then common Chinese encodings."""
        match = re.search(
            rb"<meta[^>]+charset=[\"']?([A-Za-z0-9_-]+)", body[:4096], re.I
        )
        meta_charset = match.group(1).decode("ascii", "ignore") if match else None
        for charset in (header_charset, meta_charset, "utf-8", "gb18030", "gbk"):
            if not charset:
                continue
            try:
                return body.decode(charset)
            except (LookupError, UnicodeDecodeError):
                continue
        return body.decode("utf-8", errors="replace")

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise WebFetchError("URL must be a safe HTTP or HTTPS address")
        hostname = parsed.hostname.lower()
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise WebFetchError("URL must be a safe HTTP or HTTPS address")
        try:
            literal_ip = ipaddress.ip_address(hostname)
        except ValueError:
            literal_ip = None
        if literal_ip is not None:
            addresses = [(0, 0, 0, "", (str(literal_ip), parsed.port or 443))]
        else:
            try:
                addresses = self._resolver(hostname, parsed.port or 443)
            except OSError as exc:
                raise WebFetchError("URL host could not be resolved") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise WebFetchError("URL must be a safe HTTP or HTTPS address")
