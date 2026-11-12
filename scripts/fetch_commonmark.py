#!/usr/bin/env python3
"""Fetch the pinned official CommonMark 0.31.2 conformance corpus."""
from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request

# The upstream GitHub repository does not publish spec.json at its repository
# root. The versioned spec site provides the generated JSON corpus directly.
SPEC_VERSION = "0.31.2"
URL = f"https://spec.commonmark.org/{SPEC_VERSION}/spec.json"
DEST = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "commonmark_spec.json"
)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "text-chipper-validation/1.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except Exception as exc:
        raise SystemExit(
            f"Could not download CommonMark {SPEC_VERSION} corpus from {URL}: {exc}"
        ) from exc

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Downloaded corpus is not valid JSON: {exc}") from exc

    if not isinstance(parsed, list) or len(parsed) < 500:
        count = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        raise SystemExit(f"Unexpected CommonMark corpus shape/count: {count}")

    required = {"markdown", "html", "example", "section"}
    for index, item in enumerate(parsed):
        if not isinstance(item, dict) or not required.issubset(item):
            raise SystemExit(
                f"Unexpected corpus entry at index {index}; "
                f"required fields: {sorted(required)}"
            )

    DEST.write_bytes(data)
    print(f"saved={DEST}")
    print(
        f"version={SPEC_VERSION} examples={len(parsed)} bytes={len(data)} "
        f"sha256={hashlib.sha256(data).hexdigest()}"
    )
    print(f"source={URL}")


if __name__ == "__main__":
    main()
