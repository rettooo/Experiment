from typing import List, Dict, Any, Optional

from core.interfaces.chunker import BaseChunker


class FixedChunker(BaseChunker):
    """고정 크기 청킹 구현체 (Fixed Size Chunking)"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        고정 크기로 텍스트 청킹 (간단한 sliding window 방식)
        """
        if metadata is None:
            metadata = {}

        # 텍스트를 고정 크기로 분할 (overlap 포함)
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # 빈 청크는 건너뛰기
            if chunk_text.strip():
                chunks.append(chunk_text)

            # 다음 청크의 시작 위치 (overlap 적용)
            start = end - self.chunk_overlap

            # 무한 루프 방지 (overlap이 chunk_size보다 클 경우)
            if self.chunk_overlap >= self.chunk_size:
                start = end

        # 각 청크에 메타데이터 추가
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "chunking_strategy": "fixed",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            result.append({"text": chunk_text, "metadata": chunk_metadata})

        return result

    def get_chunker_info(self) -> Dict[str, Any]:
        """청킹 전략 정보 반환"""
        return {
            "strategy": "fixed",
            "description": "Fixed Size Chunking with sliding window",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "config": self.config,
        }
