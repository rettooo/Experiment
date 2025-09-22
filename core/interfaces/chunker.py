from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseChunker(ABC):
    """텍스트 청킹 전략의 추상 인터페이스"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            metadata: 원본 문서의 메타데이터

        Returns:
            청크 리스트 (각 청크는 {'text': str, 'metadata': dict} 형태)
        """
        pass

    def get_chunker_info(self) -> Dict[str, Any]:
        """청킹 전략 정보 반환"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "config": self.config
        }