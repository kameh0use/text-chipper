"""LlamaIndex adapter for text_chipper.

Requires ``llama-index-core`` and a built ``text_chipper`` (``maturin develop --release``).

    from integrations.llama_index import TextChipperNodeParser

    parser = TextChipperNodeParser(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)

``chunk_size`` and ``chunk_overlap`` are measured in *tokens* of ``encoding``, not characters.
"""
from __future__ import annotations

from typing import Any, List

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.node_parser import TextSplitter
from text_chipper import TextChipper


class TextChipperNodeParser(TextSplitter):
    chunk_size: int = Field(default=512, gt=0, description="Token budget per chunk.")
    chunk_overlap: int = Field(default=50, ge=0, description="Token overlap between chunks.")
    encoding: str = Field(default="cl100k_base", description="Tokenizer name.")
    contextualized: bool = Field(
        default=False,
        description="Return the text with its [ctx:...] heading prefix.",
    )

    _chipper: Any = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding: str = "cl100k_base",
        contextualized: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding,
            contextualized=contextualized,
            **kwargs,
        )
        self._chipper = TextChipper(
            max_length=chunk_size, overlap=chunk_overlap, encoding=encoding
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextChipperNodeParser"

    def split_text(self, text: str) -> List[str]:
        chunks = self._chipper.chunk(text)
        if self.contextualized:
            return [c.contextualized_text for c in chunks]
        return [c.text for c in chunks]
