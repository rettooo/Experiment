from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEmbedder(ABC):
    """임베딩 모델의 추상 인터페이스"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환

        Args:
            texts: 임베딩할 텍스트 리스트
            **kwargs: 모델별 추가 파라미터

        Returns:
            임베딩 벡터 리스트
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 수 반환"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "embedding_dimension": self.get_embedding_dimension()
        }