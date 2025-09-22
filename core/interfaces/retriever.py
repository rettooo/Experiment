from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRetriever(ABC):
    """문서 검색 시스템의 추상 인터페이스"""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        문서와 임베딩을 검색 시스템에 추가

        Args:
            documents: 문서 리스트 (text, metadata 포함)
            embeddings: 각 문서에 대응하는 임베딩 벡터
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 10, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """
        쿼리 임베딩으로 유사한 문서 검색

        Args:
            query_embedding: 쿼리의 임베딩 벡터
            top_k: 반환할 상위 문서 수
            **kwargs: 검색 알고리즘별 추가 파라미터

        Returns:
            (문서, 유사도 점수) 튜플의 리스트
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        pass

    def get_retriever_info(self) -> Dict[str, Any]:
        """검색기 정보 반환"""
        return {
            "config": self.config,
            "document_count": self.get_document_count()
        }