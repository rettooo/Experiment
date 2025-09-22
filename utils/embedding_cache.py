"""
임베딩 캐싱 시스템
임베딩 모델과 청킹 전략 조합별로 임베딩 결과를 캐시
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from core.config import EmbedderConfig, ChunkerConfig


class EmbeddingCache:
    """임베딩 결과 캐싱 관리 클래스"""

    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(self, embedder_config: EmbedderConfig, chunker_config: ChunkerConfig) -> str:
        """
        임베딩 모델과 청킹 전략으로 캐시 키 생성

        Args:
            embedder_config: 임베딩 모델 설정
            chunker_config: 청킹 전략 설정

        Returns:
            캐시 키 문자열
        """
        # 임베딩 모델 부분
        model_part = embedder_config.model_name.replace("-", "_").replace(".", "_")

        # 청킹 전략 부분
        if chunker_config.type == "no_chunk":
            chunk_part = "no_chunk"
        else:
            chunk_part = f"{chunker_config.type}"
            if chunker_config.chunk_size is not None:
                chunk_part += f"_{chunker_config.chunk_size}"
            if chunker_config.chunk_overlap is not None:
                chunk_part += f"_{chunker_config.chunk_overlap}"

        cache_key = f"{model_part}_{chunk_part}"
        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """캐시 키에 해당하는 디렉토리 경로 반환"""
        return self.cache_dir / cache_key

    def exists(self, cache_key: str) -> bool:
        """캐시가 존재하는지 확인"""
        cache_path = self.get_cache_path(cache_key)
        required_files = [
            "processed_documents.pkl",
            "embeddings.npy",
            "metadata.json"
        ]

        return all((cache_path / file).exists() for file in required_files)

    def save(
        self,
        cache_key: str,
        processed_documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        additional_info: Dict[str, Any] = None
    ) -> None:
        """
        임베딩 결과를 캐시에 저장

        Args:
            cache_key: 캐시 키
            processed_documents: 청킹된 문서들
            embeddings: 임베딩 벡터들
            additional_info: 추가 정보
        """
        cache_path = self.get_cache_path(cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 처리된 문서들 저장
            with open(cache_path / "processed_documents.pkl", 'wb') as f:
                pickle.dump(processed_documents, f)

            # 2. 임베딩 벡터들 저장 (numpy 형태로)
            embeddings_array = np.array(embeddings)
            np.save(cache_path / "embeddings.npy", embeddings_array)

            # 3. 메타데이터 저장
            metadata = {
                "cache_key": cache_key,
                "document_count": len(processed_documents),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "created_at": datetime.now().isoformat(),
                "additional_info": additional_info or {}
            }

            with open(cache_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"✅ 임베딩 캐시 저장 완료: {cache_key}")
            print(f"   문서 수: {len(processed_documents)}")
            print(f"   임베딩 차원: {len(embeddings[0]) if embeddings else 0}")
            print(f"   저장 경로: {cache_path}")

        except Exception as e:
            print(f"❌ 캐시 저장 실패 ({cache_key}): {e}")
            # 실패시 부분적으로 생성된 파일들 정리
            self._cleanup_partial_cache(cache_path)
            raise

    def load(self, cache_key: str) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        캐시에서 임베딩 결과 로드

        Args:
            cache_key: 캐시 키

        Returns:
            (processed_documents, embeddings) 튜플
        """
        if not self.exists(cache_key):
            raise ValueError(f"캐시가 존재하지 않습니다: {cache_key}")

        cache_path = self.get_cache_path(cache_key)

        try:
            # 1. 처리된 문서들 로드
            with open(cache_path / "processed_documents.pkl", 'rb') as f:
                processed_documents = pickle.load(f)

            # 2. 임베딩 벡터들 로드
            embeddings_array = np.load(cache_path / "embeddings.npy")
            embeddings = embeddings_array.tolist()

            # 3. 메타데이터 로드 (확인용)
            with open(cache_path / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            print(f"✅ 임베딩 캐시 로드 완료: {cache_key}")
            print(f"   문서 수: {metadata['document_count']}")
            print(f"   임베딩 차원: {metadata['embedding_dimension']}")
            print(f"   생성일: {metadata['created_at']}")

            return processed_documents, embeddings

        except Exception as e:
            print(f"❌ 캐시 로드 실패 ({cache_key}): {e}")
            raise

    def get_metadata(self, cache_key: str) -> Dict[str, Any]:
        """캐시 메타데이터 조회"""
        if not self.exists(cache_key):
            return None

        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path / "metadata.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def list_caches(self) -> List[Dict[str, Any]]:
        """모든 캐시 목록 조회"""
        caches = []

        for cache_dir in self.cache_dir.iterdir():
            if cache_dir.is_dir():
                cache_key = cache_dir.name
                metadata = self.get_metadata(cache_key)
                if metadata:
                    caches.append({
                        "cache_key": cache_key,
                        "metadata": metadata,
                        "path": str(cache_dir)
                    })

        return sorted(caches, key=lambda x: x["metadata"]["created_at"], reverse=True)

    def delete_cache(self, cache_key: str) -> bool:
        """특정 캐시 삭제"""
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            print(f"⚠️  캐시가 존재하지 않습니다: {cache_key}")
            return False

        try:
            import shutil
            shutil.rmtree(cache_path)
            print(f"🗑️  캐시 삭제 완료: {cache_key}")
            return True
        except Exception as e:
            print(f"❌ 캐시 삭제 실패 ({cache_key}): {e}")
            return False

    def get_cache_size(self, cache_key: str) -> int:
        """캐시 크기 조회 (바이트)"""
        if not self.exists(cache_key):
            return 0

        cache_path = self.get_cache_path(cache_key)
        total_size = 0

        for file in cache_path.iterdir():
            if file.is_file():
                total_size += file.stat().st_size

        return total_size

    def _cleanup_partial_cache(self, cache_path: Path) -> None:
        """부분적으로 생성된 캐시 파일들 정리"""
        try:
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
        except Exception:
            pass

    def print_cache_stats(self) -> None:
        """캐시 통계 출력"""
        caches = self.list_caches()

        print("\n" + "="*60)
        print("임베딩 캐시 현황")
        print("="*60)

        if not caches:
            print("캐시된 임베딩이 없습니다.")
            return

        total_size = 0
        for cache_info in caches:
            cache_key = cache_info["cache_key"]
            metadata = cache_info["metadata"]
            size = self.get_cache_size(cache_key)
            total_size += size

            print(f"\n📦 {cache_key}")
            print(f"   문서 수: {metadata['document_count']}")
            print(f"   임베딩 차원: {metadata['embedding_dimension']}")
            print(f"   크기: {size / 1024 / 1024:.2f} MB")
            print(f"   생성일: {metadata['created_at']}")

        print(f"\n총 캐시 크기: {total_size / 1024 / 1024:.2f} MB")
        print(f"총 캐시 수: {len(caches)}")


# 전역 캐시 인스턴스
embedding_cache = EmbeddingCache()