from __future__ import annotations

from app.core.config import settings


def resolve_origin(req_origin: str | None) -> str:
    """Return the request origin if it's in the allow-list, else empty string."""
    if req_origin is not None and req_origin in settings.CORS_ALLOW_ORIGINS:
        return req_origin
    return ''


def header_builder(origin: str) -> dict[str, str]:
    """Build CORS headers for the given origin."""
    return {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Headers': _list_to_string(settings.CORS_ALLOW_HEADERS),
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '86400',
    }


def _list_to_string(items: list[str]) -> str:
    """Convert a list of strings to a comma-separated string."""
    return ', '.join(items)
