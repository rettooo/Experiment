import os
import chromadb
from typing import List, Dict, Any, Tuple, Optional

from core.interfaces.retriever import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """ChromaDB 기반 검색 시스템 구현체 (현재 서비스에서 사용 중)"""

    def __init__(self, collection_name: str = "job-postings", persist_directory: str = None, **kwargs):
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

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """문서와 임베딩을 ChromaDB에 추가"""
        if len(documents) != len(embeddings):
            raise ValueError("문서 수와 임베딩 수가 일치하지 않습니다")

        if not documents:
            return

        # ChromaDB에 저장할 데이터 준비
        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            # 고유 ID 생성
            metadata = doc.get("metadata", {})
            rec_idx = metadata.get("rec_idx", f"doc_{i}")

            # 청킹된 문서의 경우 chunk_index를 포함하여 고유 ID 생성
            if "chunk_index" in metadata:
                doc_id = f"{rec_idx}_chunk_{metadata['chunk_index']}"
            else:
                doc_id = str(rec_idx)

            ids.append(doc_id)
            texts.append(doc["text"])
            metadatas.append(doc.get("metadata", {}))

        # ChromaDB에 배치 추가
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding: List[float], top_k: int = 10, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """쿼리 임베딩으로 유사한 문서 검색"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # 결과 포맷팅
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = {
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                }
                # ChromaDB는 거리를 반환하므로 유사도로 변환 (1 - normalized_distance)
                distance = results['distances'][0][i]
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

    def get_retriever_info(self) -> Dict[str, Any]:
        """검색기 정보 반환"""
        return {
            "type": "ChromaDB",
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": self.get_document_count(),
            "config": self.config
        }