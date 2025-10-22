"""
S3 문서 로드 캐싱 유틸리티

S3에서 로드한 원본 문서를 캐시하여 재사용
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional


class DocumentCache:
    """S3 문서 로드 캐시 관리"""

    def __init__(self, cache_dir: str = "cache/documents"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        s3_bucket: str,
        pdf_prefix: str,
        json_prefix: str,
        data_version: str = "v1",
    ) -> str:
        """S3 설정으로부터 캐시 키 생성"""
        config_str = f"{s3_bucket}_{pdf_prefix}_{json_prefix}"
        # 파일명으로 사용 가능한 형태로 해시
        hash_value = hashlib.md5(config_str.encode()).hexdigest()[:16]
        cache_key = f"s3_{hash_value}_{data_version}"
        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """캐시 디렉토리 경로 반환"""
        return self.cache_dir / cache_key

    def exists(self, cache_key: str) -> bool:
        """캐시 존재 여부 확인"""
        cache_path = self.get_cache_path(cache_key)
        documents_file = cache_path / "documents.pkl"
        metadata_file = cache_path / "metadata.json"

        return documents_file.exists() and metadata_file.exists()

    def save(
        self, cache_key: str, documents: List[Dict[str, Any]], s3_config: Dict[str, str]
    ) -> None:
        """문서 캐시 저장"""
        cache_path = self.get_cache_path(cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)

        # 문서 데이터 저장
        documents_file = cache_path / "documents.pkl"
        with open(documents_file, "wb") as f:
            pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 메타데이터 저장
        metadata = {
            "cache_key": cache_key,
            "document_count": len(documents),
            "s3_config": s3_config,
            "cache_created_at": str(Path(documents_file).stat().st_mtime),
        }

        metadata_file = cache_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ 문서 캐시 저장 완료: {cache_path}")
        print(f"   문서 수: {len(documents)}개")

    def load(self, cache_key: str) -> List[Dict[str, Any]]:
        """캐시된 문서 로드"""
        cache_path = self.get_cache_path(cache_key)
        documents_file = cache_path / "documents.pkl"

        if not documents_file.exists():
            raise FileNotFoundError(f"캐시 파일 없음: {documents_file}")

        with open(documents_file, "rb") as f:
            documents = pickle.load(f)

        print(f"✅ 문서 캐시 로드 완료: {cache_path}")
        print(f"   문서 수: {len(documents)}개")

        return documents

    def get_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """캐시 메타데이터 조회"""
        cache_path = self.get_cache_path(cache_key)
        metadata_file = cache_path / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, cache_key: str) -> bool:
        """캐시 삭제"""
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return False

        import shutil

        shutil.rmtree(cache_path)
        print(f"🗑️  문서 캐시 삭제: {cache_path}")
        return True

    def list_caches(self) -> List[Dict[str, Any]]:
        """모든 캐시 목록 조회"""
        caches = []

        if not self.cache_dir.exists():
            return caches

        for cache_path in self.cache_dir.iterdir():
            if cache_path.is_dir():
                metadata = self.get_metadata(cache_path.name)
                if metadata:
                    caches.append(metadata)

        return caches


# 전역 인스턴스
document_cache = DocumentCache()
