import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from core.interfaces.retriever import BaseRetriever


class FAISSRetriever(BaseRetriever):
    """FAISS 기반 검색 시스템 구현체 (ChromaDB와 동일한 구조)"""

    def __init__(
        self,
        collection_name: str = "job-postings",
        persist_directory: str = None,
        index_type: str = "flatip",
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "/tmp/faiss_experiment"
        self.index_type = index_type
        self.use_gpu = use_gpu

        # 저장 경로 생성
        os.makedirs(self.persist_directory, exist_ok=True)

        # FAISS 인덱스 및 문서 저장소
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None

        # 기존 인덱스 로드 시도
        self._try_load_index()

    def _try_load_index(self):
        """기존 인덱스 및 문서 로드 시도"""
        index_path = os.path.join(
            self.persist_directory, f"{self.collection_name}.index"
        )
        docs_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                # 인덱스 로드
                self.index = faiss.read_index(index_path)
                self.dimension = self.index.d

                # 문서 로드
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)

                print(f"✅ 기존 FAISS 인덱스 로드: {len(self.documents)}개 문서")
            except Exception as e:
                print(f"⚠️  인덱스 로드 실패: {e}")
                self.index = None
                self.documents = []

    def _create_index(self, dimension: int):
        """FAISS 인덱스 생성"""
        self.dimension = dimension

        if self.index_type == "flatl2":
            # Flat L2: 정확한 L2 거리 검색
            index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "flatip":
            # Flat Inner Product: 코사인 유사도 검색 (정규화된 벡터용)
            index = faiss.IndexFlatIP(dimension)
        else:
            # 기본값: flatip
            index = faiss.IndexFlatIP(dimension)

        # GPU 가속 (사용 가능하면)
        if self.use_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    print("🚀 GPU 가속 활성화")
            except Exception as e:
                print(f"⚠️  GPU 가속 실패 (CPU 사용): {e}")

        self.index = index

    def add_documents(
        self, documents: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """문서와 임베딩을 FAISS에 추가"""
        if len(documents) != len(embeddings):
            raise ValueError("문서 수와 임베딩 수가 일치하지 않습니다")

        if not documents:
            return

        # 임베딩을 numpy array로 변환
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # 인덱스 생성 (처음이면)
        if self.index is None:
            self._create_index(dimension=embeddings_np.shape[1])

        # 인덱스에 벡터 추가
        self.index.add(embeddings_np)

        # 문서 저장 (메타데이터 포함)
        self.documents.extend(documents)

    def search(
        self, query_embedding: List[float], top_k: int = 10, **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """쿼리 임베딩으로 유사한 문서 검색"""
        if self.index is None or len(self.documents) == 0:
            return []

        # 쿼리를 numpy array로 변환
        query_np = np.array([query_embedding], dtype=np.float32)

        # FAISS 검색 (거리와 인덱스 반환)
        distances, indices = self.index.search(query_np, top_k)

        # 결과 포맷팅 (ChromaDB와 동일한 형식)
        search_results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1 or idx >= len(self.documents):
                continue

            doc = self.documents[int(idx)]

            # 거리를 유사도 점수로 변환
            if self.index_type == "flatl2":
                # L2 거리: 작을수록 유사 → 유사도 변환
                similarity = 1.0 / (1.0 + float(distance))
            else:  # flatip
                # Inner Product: 클수록 유사 → 그대로 사용
                similarity = float(distance)

            search_results.append((doc, similarity))

        return search_results

    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return len(self.documents)

    def clear_collection(self) -> None:
        """컬렉션의 모든 문서 삭제 (실험용)"""
        self.index = None
        self.documents = []
        self.dimension = None

    def save_index(self) -> None:
        """인덱스를 디스크에 저장"""
        if self.index is None or len(self.documents) == 0:
            print("⚠️  저장할 인덱스가 없습니다.")
            return

        index_path = os.path.join(
            self.persist_directory, f"{self.collection_name}.index"
        )
        docs_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")

        try:
            # GPU 인덱스는 CPU로 변환 후 저장
            if (
                self.use_gpu
                and hasattr(self.index, "__class__")
                and "Gpu" in str(self.index.__class__)
            ):
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, index_path)
            else:
                faiss.write_index(self.index, index_path)

            # 문서 저장
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)

            print(f"💾 FAISS 인덱스 저장 완료: {index_path}")
        except Exception as e:
            print(f"❌ 인덱스 저장 실패: {e}")

    def get_retriever_info(self) -> Dict[str, Any]:
        """검색기 정보 반환"""
        return {
            "type": "FAISS",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "index_type": self.index_type,
            "document_count": self.get_document_count(),
            "dimension": self.dimension,
            "use_gpu": self.use_gpu,
            "config": self.config,
        }
