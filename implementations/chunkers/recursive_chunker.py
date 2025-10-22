from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.interfaces.chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """LangChain RecursiveCharacterTextSplitter 기반 청킹 구현체"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        RecursiveCharacterTextSplitter로 텍스트 청킹
        """
        if metadata is None:
            metadata = {}

        # LangChain splitter로 텍스트 분할
        chunks = self.splitter.split_text(text)

        # 각 청크에 메타데이터 추가
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            result.append({"text": chunk_text, "metadata": chunk_metadata})

        return result

    def get_chunker_info(self) -> Dict[str, Any]:
        """청킹 전략 정보 반환"""
        return {
            "strategy": "recursive",
            "description": "LangChain RecursiveCharacterTextSplitter 사용",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "config": self.config,
        }
