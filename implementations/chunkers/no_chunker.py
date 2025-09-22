from typing import List, Dict, Any, Optional

from core.interfaces.chunker import BaseChunker


class NoChunker(BaseChunker):
    """청킹을 하지 않는 구현체 (현재 서비스에서 사용 중)"""

    def __init__(self, **kwargs):
        # NoChunker는 chunk_size, chunk_overlap 사용하지 않음
        super().__init__(chunk_size=None, chunk_overlap=None, **kwargs)

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        텍스트를 청킹하지 않고 그대로 반환

        현재 서비스에서는 PDF 전체를 하나의 문서로 처리
        """
        if metadata is None:
            metadata = {}

        return [{
            "text": text,
            "metadata": {
                **metadata,
                "chunk_index": 0,
                "chunk_count": 1,
                "chunking_strategy": "no_chunk"
            }
        }]

    def get_chunker_info(self) -> Dict[str, Any]:
        """청킹 전략 정보 반환"""
        return {
            "strategy": "no_chunk",
            "description": "전체 문서를 하나의 청크로 처리",
            "chunk_size": None,
            "chunk_overlap": None,
            "config": self.config
        }