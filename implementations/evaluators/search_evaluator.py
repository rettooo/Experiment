import math
from typing import List, Dict, Any, Set
from collections import defaultdict

from core.interfaces.evaluator import SearchEvaluator, QueryResult, EvaluationResult


class SearchMetricsEvaluator(SearchEvaluator):
    """검색 성능 지표 계산 구현체"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, query_results: List[QueryResult]) -> List[EvaluationResult]:
        """전체 평가 지표 계산"""
        results = []

        # k 값들 (설정에서 가져오거나 기본값 사용)
        k_values = self.config.get('k_values', [1, 3, 5, 10])

        # Recall@k 계산
        for k in k_values:
            recall_k = self.calculate_recall_at_k(query_results, k)
            results.append(EvaluationResult(
                metric_name=f"recall@{k}",
                score=recall_k,
                details={"k": k, "total_queries": len(query_results)}
            ))

        # Precision@k 계산
        for k in k_values:
            precision_k = self.calculate_precision_at_k(query_results, k)
            results.append(EvaluationResult(
                metric_name=f"precision@{k}",
                score=precision_k,
                details={"k": k, "total_queries": len(query_results)}
            ))

        # MRR 계산
        mrr_score = self.calculate_mrr(query_results)
        results.append(EvaluationResult(
            metric_name="mrr",
            score=mrr_score,
            details={"total_queries": len(query_results)}
        ))

        # MAP 계산
        map_score = self.calculate_map(query_results)
        results.append(EvaluationResult(
            metric_name="map",
            score=map_score,
            details={"total_queries": len(query_results)}
        ))

        # nDCG@k 계산
        for k in k_values:
            ndcg_k = self.calculate_ndcg(query_results, k)
            results.append(EvaluationResult(
                metric_name=f"ndcg@{k}",
                score=ndcg_k,
                details={"k": k, "total_queries": len(query_results)}
            ))

        return results

    def calculate_recall_at_k(self, query_results: List[QueryResult], k: int) -> float:
        """
        Recall@k 계산
        Recall@k = (상위 k개 중 관련 문서 수) / (전체 관련 문서 수)
        """
        if not query_results:
            return 0.0

        total_recall = 0.0
        valid_queries = 0

        for query_result in query_results:
            if not query_result.ground_truth_docs:
                continue

            valid_queries += 1

            # 상위 k개 검색 결과에서 문서 ID 추출
            retrieved_k = []
            for i, doc in enumerate(query_result.retrieved_docs):
                if i >= k:
                    break
                doc_id = self._extract_doc_id(doc)
                if doc_id:
                    retrieved_k.append(doc_id)

            # Ground truth와 매칭
            ground_truth_set = set(query_result.ground_truth_docs)
            retrieved_set = set(retrieved_k)

            relevant_retrieved = len(ground_truth_set.intersection(retrieved_set))
            total_relevant = len(ground_truth_set)

            if total_relevant > 0:
                total_recall += relevant_retrieved / total_relevant

        return total_recall / valid_queries if valid_queries > 0 else 0.0

    def calculate_precision_at_k(self, query_results: List[QueryResult], k: int) -> float:
        """
        Precision@k 계산
        Precision@k = (상위 k개 중 관련 문서 수) / k
        """
        if not query_results:
            return 0.0

        total_precision = 0.0
        valid_queries = 0

        for query_result in query_results:
            if not query_result.ground_truth_docs:
                continue

            valid_queries += 1

            # 상위 k개 검색 결과에서 문서 ID 추출
            retrieved_k = []
            for i, doc in enumerate(query_result.retrieved_docs):
                if i >= k:
                    break
                doc_id = self._extract_doc_id(doc)
                if doc_id:
                    retrieved_k.append(doc_id)

            # Ground truth와 매칭
            ground_truth_set = set(query_result.ground_truth_docs)
            retrieved_set = set(retrieved_k)

            relevant_retrieved = len(ground_truth_set.intersection(retrieved_set))
            precision = relevant_retrieved / min(k, len(retrieved_k)) if retrieved_k else 0.0

            total_precision += precision

        return total_precision / valid_queries if valid_queries > 0 else 0.0

    def calculate_mrr(self, query_results: List[QueryResult]) -> float:
        """
        Mean Reciprocal Rank 계산
        MRR = (1/|Q|) * Σ(1/rank_i)
        """
        if not query_results:
            return 0.0

        total_reciprocal_rank = 0.0
        valid_queries = 0

        for query_result in query_results:
            if not query_result.ground_truth_docs:
                continue

            valid_queries += 1
            ground_truth_set = set(query_result.ground_truth_docs)

            # 첫 번째 관련 문서의 순위 찾기
            first_relevant_rank = None
            for rank, doc in enumerate(query_result.retrieved_docs, 1):
                doc_id = self._extract_doc_id(doc)
                if doc_id and doc_id in ground_truth_set:
                    first_relevant_rank = rank
                    break

            if first_relevant_rank:
                total_reciprocal_rank += 1.0 / first_relevant_rank

        return total_reciprocal_rank / valid_queries if valid_queries > 0 else 0.0

    def calculate_map(self, query_results: List[QueryResult]) -> float:
        """
        Mean Average Precision 계산
        MAP = (1/|Q|) * Σ(AP_i)
        """
        if not query_results:
            return 0.0

        total_ap = 0.0
        valid_queries = 0

        for query_result in query_results:
            if not query_result.ground_truth_docs:
                continue

            valid_queries += 1
            ap = self._calculate_average_precision(query_result)
            total_ap += ap

        return total_ap / valid_queries if valid_queries > 0 else 0.0

    def calculate_ndcg(self, query_results: List[QueryResult], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain@k 계산
        nDCG@k = DCG@k / IDCG@k
        """
        if not query_results:
            return 0.0

        total_ndcg = 0.0
        valid_queries = 0

        for query_result in query_results:
            if not query_result.ground_truth_docs:
                continue

            valid_queries += 1

            # DCG@k 계산
            dcg = self._calculate_dcg_at_k(query_result, k)

            # IDCG@k 계산 (이상적인 순서)
            idcg = self._calculate_ideal_dcg_at_k(query_result, k)

            # nDCG@k 계산
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg

        return total_ndcg / valid_queries if valid_queries > 0 else 0.0

    def _extract_doc_id(self, doc: Dict[str, Any]) -> str:
        """문서에서 고유 ID 추출"""
        metadata = doc.get('metadata', {})
        return metadata.get('rec_idx') or metadata.get('id') or metadata.get('filename', '')

    def _calculate_average_precision(self, query_result: QueryResult) -> float:
        """단일 쿼리에 대한 Average Precision 계산"""
        ground_truth_set = set(query_result.ground_truth_docs)
        relevant_retrieved = 0
        total_precision = 0.0

        for rank, doc in enumerate(query_result.retrieved_docs, 1):
            doc_id = self._extract_doc_id(doc)
            if doc_id and doc_id in ground_truth_set:
                relevant_retrieved += 1
                precision_at_rank = relevant_retrieved / rank
                total_precision += precision_at_rank

        total_relevant = len(ground_truth_set)
        return total_precision / total_relevant if total_relevant > 0 else 0.0

    def _calculate_dcg_at_k(self, query_result: QueryResult, k: int) -> float:
        """DCG@k 계산"""
        ground_truth_set = set(query_result.ground_truth_docs)
        dcg = 0.0

        for rank, doc in enumerate(query_result.retrieved_docs, 1):
            if rank > k:
                break

            doc_id = self._extract_doc_id(doc)
            # 관련 문서면 relevance = 1, 아니면 0 (이진 relevance)
            relevance = 1 if (doc_id and doc_id in ground_truth_set) else 0

            if rank == 1:
                dcg += relevance
            else:
                dcg += relevance / math.log2(rank)

        return dcg

    def _calculate_ideal_dcg_at_k(self, query_result: QueryResult, k: int) -> float:
        """이상적인 DCG@k 계산 (모든 관련 문서가 상위에 있을 때)"""
        num_relevant = len(query_result.ground_truth_docs)
        ideal_k = min(k, num_relevant)

        idcg = 0.0
        for rank in range(1, ideal_k + 1):
            if rank == 1:
                idcg += 1.0
            else:
                idcg += 1.0 / math.log2(rank)

        return idcg

    def get_supported_metrics(self) -> List[str]:
        """지원하는 평가 지표 목록 반환"""
        return ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]