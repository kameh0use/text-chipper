"""The single, library-independent reference tokenizer used to judge every chunker.

Every library is *configured* with cl100k_base, but none of them is trusted to report its own
token counts: all budget and quality numbers are measured here, with tiktoken, on the text the
library actually delivered.
"""
from __future__ import annotations

import re
from typing import Protocol


class RefTokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class TiktokenRef:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self.name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        # disallowed_special=() so that text such as "<|endoftext|>" is counted as ordinary text.
        return len(self._enc.encode(text, disallowed_special=()))


class RegexRef:
    """Crude word/punctuation tokenizer. ONLY for the self-tests; never for published numbers."""

    name = "regex-test"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._pattern.findall(text))


def get_reference(name: str) -> RefTokenizer:
    if name == "regex-test":
        return RegexRef()
    return TiktokenRef(name)
