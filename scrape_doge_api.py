#!/usr/bin/env python3
"""Scrape the DOGE API and export endpoint data to CSV files.

This script iterates through the ``grants``, ``contracts``, ``leases``
and ``payments`` endpoints exposed by ``https://api.doge.gov``.  It
paginates through the API responses, flattens the JSON payload for each
record and writes the consolidated rows to per-endpoint CSV files.

The script purposely avoids external dependencies so that it can run in
minimal Python environments.  It also introduces small delays between
requests to remain courteous to the remote API and help prevent rate
limits from triggering.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, build_opener

BASE_URL = "https://api.doge.gov/"
ENDPOINTS = ("grants", "contracts", "leases", "payments")
DEFAULT_PER_PAGE = 100
DEFAULT_SLEEP = 0.5  # seconds
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0

JsonObject = Mapping[str, Any]


@dataclass
class PageResult:
    """Container for a paginated response."""

    records: List[JsonObject]
    next_url: Optional[str]


class ScraperError(RuntimeError):
    """Base error raised for scraper failures."""


class ApiClient:
    """Minimal HTTP client around :mod:`urllib` with retry support."""

    def __init__(self, base_url: str, sleep: float = DEFAULT_SLEEP) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.sleep = sleep
        self._opener = build_opener()

    def request(self, url: str) -> Any:
        """Perform a GET request with basic retry logic.

        Args:
            url: The absolute or relative URL to fetch.

        Returns:
            The parsed JSON payload.
        """

        if not url.startswith("http"):
            url = urljoin(self.base_url, url)

        headers = {
            "Accept": "application/json",
            "User-Agent": "DOGEScraper/1.0 (+https://api.doge.gov/docs)",
        }

        for attempt in range(1, MAX_RETRIES + 1):
            req = Request(url, headers=headers)
            try:
                with self._opener.open(req, timeout=60) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read().decode(charset)
                    return json.loads(body)
            except HTTPError as exc:  # pragma: no cover - network dependent
                if exc.code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                    sleep_for = self.sleep * (RETRY_BACKOFF ** (attempt - 1))
                    time.sleep(sleep_for)
                    continue
                raise ScraperError(
                    f"HTTP error {exc.code} for {url}: {exc.reason}"
                ) from exc
            except URLError as exc:  # pragma: no cover - network dependent
                if attempt < MAX_RETRIES:
                    sleep_for = self.sleep * (RETRY_BACKOFF ** (attempt - 1))
                    time.sleep(sleep_for)
                    continue
                raise ScraperError(f"Network error retrieving {url}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ScraperError(f"Invalid JSON received from {url}: {exc}") from exc

        raise ScraperError(f"Unable to fetch {url} after {MAX_RETRIES} attempts")


def parse_page(payload: Any) -> PageResult:
    """Extract the list of records and the next URL from a payload."""

    if isinstance(payload, list):
        records = payload
        next_url = None
    elif isinstance(payload, Mapping):
        records = _extract_record_list(payload)
        next_url = _extract_next_link(payload)
    else:
        raise ScraperError(
            "Unexpected response format: expected list or object, "
            f"received {type(payload).__name__}"
        )

    if not isinstance(records, list):
        raise ScraperError(
            "Unexpected payload: could not locate list of records in response"
        )

    normalized_records: List[JsonObject] = []
    for record in records:
        if isinstance(record, Mapping):
            normalized_records.append(record)
        else:
            raise ScraperError(
                "Encountered non-object record in response payload"
            )

    return PageResult(records=normalized_records, next_url=next_url)


def _extract_record_list(payload: Mapping[str, Any]) -> List[Any]:
    for key in ("results", "data", "items", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if isinstance(payload.get("value"), list):
        return payload["value"]
    if isinstance(payload.get("response"), list):
        return payload["response"]
    raise ScraperError(
        "Unable to determine collection field in response payload. "
        "Expected one of 'results', 'data', 'items', 'records', 'value', or 'response'."
    )


def _extract_next_link(payload: Mapping[str, Any]) -> Optional[str]:
    links = payload.get("links")
    if isinstance(links, Mapping):
        next_link = links.get("next")
        if isinstance(next_link, str):
            return next_link
        if isinstance(next_link, Mapping) and isinstance(next_link.get("href"), str):
            return next_link["href"]
    next_link = payload.get("next")
    if isinstance(next_link, str):
        return next_link
    return None


def flatten_record(record: Mapping[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten nested JSON objects into a single-level mapping.

    Nested dictionary keys are concatenated with dot notation (``foo.bar``)
    while lists are either enumerated (when they contain dictionaries) or
    converted to a JSON string representation.
    """

    items: Dict[str, Any] = {}
    for key, value in record.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_record(value, new_key))
        elif isinstance(value, list):
            if value and all(isinstance(entry, Mapping) for entry in value):
                for idx, entry in enumerate(value):
                    indexed_key = f"{new_key}[{idx}]"
                    items.update(flatten_record(entry, indexed_key))
            else:
                items[new_key] = json.dumps(value, ensure_ascii=False)
        else:
            items[new_key] = value
    return items


def write_csv(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write flattened rows to ``path``.

    The CSV header is determined from the union of all keys present in the
    provided rows.
    """

    flattened_rows: List[Dict[str, Any]] = [dict(row) for row in rows]
    fieldnames: List[str] = sorted({key for row in flattened_rows for key in row})

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flattened_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def scrape_endpoint(
    client: ApiClient,
    endpoint: str,
    per_page: int = DEFAULT_PER_PAGE,
    sleep: float = DEFAULT_SLEEP,
) -> List[Dict[str, Any]]:
    """Scrape all pages for a specific endpoint."""

    results: List[Dict[str, Any]] = []
    page_number = 1
    next_url: Optional[str] = urljoin(
        client.base_url, f"{endpoint}?{urlencode({'page': page_number, 'per_page': per_page})}"
    )

    while next_url:
        current_url = next_url
        payload = client.request(current_url)
        page = parse_page(payload)

        for record in page.records:
            results.append(flatten_record(record))

        if page.next_url:
            next_url = page.next_url
            if not next_url.startswith("http"):
                next_url = urljoin(client.base_url, next_url)
        else:
            if len(page.records) < per_page:
                next_url = None
            else:
                page_number += 1
                query = urlencode({"page": page_number, "per_page": per_page})
                next_url = urljoin(client.base_url, f"{endpoint}?{query}")

        time.sleep(sleep)

    return results


def run(endpoints: Iterable[str], output_dir: str, per_page: int, sleep: float) -> None:
    client = ApiClient(BASE_URL, sleep=sleep)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for endpoint in endpoints:
        print(f"Scraping {endpoint}...", file=sys.stderr)
        rows = scrape_endpoint(client, endpoint, per_page=per_page, sleep=sleep)
        output_path = target_dir / f"{endpoint}.csv"
        write_csv(str(output_path), rows)
        print(f"Wrote {len(rows)} rows to {output_path}", file=sys.stderr)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where CSV files will be written (default: current directory)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Number of records to request per page (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Seconds to wait between requests (default: %(default)s)",
    )
    parser.add_argument(
        "endpoints",
        nargs="*",
        default=list(ENDPOINTS),
        help="Subset of endpoints to scrape (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(args.endpoints, args.output_dir, args.per_page, args.sleep)
    except ScraperError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
