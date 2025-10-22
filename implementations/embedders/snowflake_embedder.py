import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from core.interfaces.embedder import BaseEmbedder


class SnowflakeEmbedder(BaseEmbedder):
    """Snowflake 임베딩 모델 구현체"""

    def __init__(
        self,
        model_name: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.batch_size = batch_size

        # apple silicon MPS 사용
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("🚀 Apple Silicon MPS 사용 활성화")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 CUDA 사용 활성화")
        else:
            self.device = "cpu"
            print("⚠️  CPU 모드 (느림)")

        # 모델 로드 (SentenceTransformer 사용)
        print(f"📥 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)
        print(f"✅ 모델 로드 완료 (device: {self.device})")

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """텍스트 리스트를 임베딩 벡터로 변환"""
        if not texts:
            return []
        # 배치 임베딩 (GPU 가속)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            device=self.device,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원 수 반환"""
        return 1024

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "embedding_dimension": self.get_embedding_dimension(),
        }
