import os
from typing import List, Optional
from openai import OpenAI

from core.interfaces.embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 임베딩 모델 구현체 (현재 서비스에서 사용 중)"""

    def __init__(self, model_name: str = "text-embedding-ada-002", batch_size: int = 5, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.batch_size = batch_size

        # 모델별 차원 수 매핑
        self._dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """텍스트 리스트를 임베딩 벡터로 변환"""
        if not texts:
            return []

        all_embeddings = []

        # 배치 단위로 처리 (현재 서비스와 동일)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"OpenAI 임베딩 API 호출 실패: {e}")
                raise

        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 수 반환"""
        return self._dimension_map.get(self.model_name, 1536)  # 기본값 1536