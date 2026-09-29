"""Domain allow-list validation for optional web sources."""

from urllib.parse import urlparse


def is_allowed_domain(url: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return True
    hostname = (urlparse(url).hostname or "").lower()
    return any(
        hostname == domain.lower() or hostname.endswith(f".{domain.lower()}")
        for domain in allowed_domains
    )
