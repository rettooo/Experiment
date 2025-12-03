import os
import chromadb
from typing import List, Dict, Any, Tuple, Optional

from core.interfaces.retriever import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """ChromaDB 기반 검색 시스템 구현체 (현재 서비스에서 사용 중)"""

    def __init__(
        self,
        collection_name: str = "job-postings",
        persist_directory: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "/tmp/chroma_experiment"

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # 컬렉션이 없으면 새로 생성
            self.collection = self.client.create_collection(name=self.collection_name)

    def add_documents(
        self, documents: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """문서와 임베딩을 ChromaDB에 추가"""
        if len(documents) != len(embeddings):
            raise ValueError("문서 수와 임베딩 수가 일치하지 않습니다")

        if not documents:
            return

        # ChromaDB에 배치 추가 (배치 크기 제한 해결)
        batch_size = 5000  # ChromaDB 제한보다 작게 설정
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_docs = documents[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]

            # 배치별로 문서 ID, 텍스트, 메타데이터 준비
            batch_ids = []
            batch_texts = []
            batch_metadatas = []

            for j, doc in enumerate(batch_docs):
                # 고유 ID 생성
                metadata = doc.get("metadata", {})
                # rec_idx 또는 rec_id 확인 (StructuredDocumentLoader는 rec_id 사용)
                rec_idx = (
                    metadata.get("rec_idx") or metadata.get("rec_id") or f"doc_{i+j}"
                )

                # 고유 ID 생성 우선순위:
                # 1. chunk_id (StructuredDocumentLoader가 생성)
                # 2. recursive chunking이 적용된 경우: chunk_id + recursive_chunk_index
                # 3. chunk_index가 있는 경우: rec_idx_chunk_{chunk_index}
                # 4. 그 외: rec_idx

                if "chunk_id" in metadata:
                    chunk_id = metadata["chunk_id"]
                    # Recursive chunking이 적용된 경우 추가 인덱스 포함
                    if "recursive_chunk_index" in metadata:
                        doc_id = f"{chunk_id}_rec_{metadata['recursive_chunk_index']}"
                    else:
                        doc_id = chunk_id
                elif "chunk_index" in metadata:
                    doc_id = f"{rec_idx}_chunk_{metadata['chunk_index']}"
                else:
                    doc_id = str(rec_idx)

                # ChromaDB 메타데이터는 primitive 타입만 허용하므로 변환
                safe_metadata: Dict[str, Any] = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_metadata[key] = value
                    elif isinstance(value, list):
                        # 리스트는 쉼표로 join 해서 문자열로 저장 (예: 태그 리스트)
                        safe_metadata[key] = ", ".join(map(str, value))
                    else:
                        # 기타 타입(dict 등)은 문자열로 직렬화
                        safe_metadata[key] = str(value)

                # rec_idx가 없고 rec_id가 있는 경우, rec_idx도 메타데이터에 추가
                # (검색 시 일관성 있게 rec_idx로 접근할 수 있도록)
                if "rec_idx" not in safe_metadata and "rec_id" in safe_metadata:
                    safe_metadata["rec_idx"] = safe_metadata["rec_id"]

                batch_ids.append(doc_id)
                batch_texts.append(doc["text"])
                batch_metadatas.append(safe_metadata)

            # 배치 추가
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

            print(f"📦 ChromaDB 배치 추가: {i+1}-{end_idx}/{total_docs} 문서")

    def search(
        self, query_embedding: List[float], top_k: int = 10, **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """쿼리 임베딩으로 유사한 문서 검색"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 결과 포맷팅
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    ),
                }
                # ChromaDB는 거리를 반환하므로 유사도로 변환 (1 - normalized_distance)
                distance = results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)  # 거리를 유사도로 변환

                search_results.append((doc, similarity))

        return search_results

    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.count()

    def clear_collection(self) -> None:
        """컬렉션의 모든 문서 삭제 (실험용)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
        except Exception as e:
            print(f"컬렉션 초기화 실패: {e}")

    def get_metadata_by_rec_idx(self, rec_idx: str) -> Optional[Dict[str, Any]]:
        """
        rec_idx로 문서 메타데이터 조회

        Args:
            rec_idx: 문서 ID

        Returns:
            메타데이터 딕셔너리 (없으면 None)
        """
        try:
            # ChromaDB에서 rec_idx로 필터링하여 조회
            results = self.collection.get(
                where={"rec_idx": rec_idx},
                limit=1,
            )

            if results["metadatas"] and len(results["metadatas"]) > 0:
                # 첫 번째 결과의 메타데이터 반환
                return results["metadatas"][0]
            return None
        except Exception as e:
            print(f"⚠️  rec_idx {rec_idx} 메타데이터 조회 실패: {e}")
            return None

    def get_retriever_info(self) -> Dict[str, Any]:
        """검색기 정보 반환"""
        return {
            "type": "ChromaDB",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.get_document_count(),
            "config": self.config,
        }
