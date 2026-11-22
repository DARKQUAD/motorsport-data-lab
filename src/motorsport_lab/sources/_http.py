"""Small polite HTTP helper: HTTPS only, one user agent, a delay between calls, a disk cache."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

USER_AGENT = "motorsport-data-lab/0.1 (+https://github.com/DARKQUAD/motorsport-data-lab)"
_state = {"last_call": 0.0}


def _get(url: str, timeout: int) -> bytes:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"only https URLs are allowed: {url}")
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    for _ in range(5):  # follow a few redirects
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout)
        try:
            conn.request(
                "GET", urllib.parse.quote(path, safe="/?=&%:+,"), headers={"User-Agent": USER_AGENT}
            )
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                parts = urllib.parse.urlsplit(
                    urllib.parse.urljoin(url, response.getheader("Location", ""))
                )
                path = parts.path + (f"?{parts.query}" if parts.query else "")
                continue
            if response.status != 200:
                raise OSError(f"HTTP {response.status} for {url}")
            return response.read()
        finally:
            conn.close()
    raise OSError(f"too many redirects for {url}")


def fetch(
    url: str, cache_dir: Path | None = None, delay_s: float = 1.0, timeout: int = 120
) -> bytes:
    """GET ``url`` (cached by URL hash), waiting ``delay_s`` between network calls."""
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
        if cache_file.exists():
            return cache_file.read_bytes()
    wait = delay_s - (time.monotonic() - _state["last_call"])
    if wait > 0:
        time.sleep(wait)
    data = _get(url, timeout)
    _state["last_call"] = time.monotonic()
    if cache_file is not None:
        cache_file.write_bytes(data)
    return data


def fetch_json(url: str, cache_dir: Path | None = None, delay_s: float = 1.0) -> Any:
    """GET a JSON document (cached)."""
    return json.loads(fetch(url, cache_dir, delay_s).decode("utf-8"))
