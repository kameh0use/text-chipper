"""LangChain adapter for text_chipper.

Requires ``langchain-text-splitters`` and a built ``text_chipper`` (``maturin develop --release``).

    from integrations.langchain import TextChipperSplitter

    splitter = TextChipperSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents([markdown_text])

``chunk_size`` and ``chunk_overlap`` are measured in *tokens* of ``encoding``, not characters.
"""
from __future__ import annotations

from typing import Any, List

from langchain_text_splitters import TextSplitter
from text_chipper import TextChipper


class TextChipperSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding: str = "cl100k_base",
        contextualized: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            chunk_size: token budget per chunk (``max_length`` of ``TextChipper``).
            chunk_overlap: token overlap between chunks.
            encoding: tokenizer name, e.g. ``cl100k_base`` or ``o200k_base``.
            contextualized: return ``contextualized_text`` (with the ``[ctx:...]`` heading
                prefix) instead of the plain chunk text.
            **kwargs: passed to ``langchain_text_splitters.TextSplitter`` (e.g. ``add_start_index``).
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self._contextualized = contextualized
        self._chipper = TextChipper(
            max_length=chunk_size, overlap=chunk_overlap, encoding=encoding
        )

    def split_text(self, text: str) -> List[str]:
        chunks = self._chipper.chunk(text)
        if self._contextualized:
            return [c.contextualized_text for c in chunks]
        return [c.text for c in chunks]
