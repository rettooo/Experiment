import math
from typing import List, Dict, Any, Set
from core.interfaces.evaluator import QueryResult, EvaluationResult


class RetrieverEvaluator:
    """검색 성능 지표 계산 class (평가지표 수정!!)"""

    def __init__(self, ground_truth_size: int = 5):
        # 주의: ground_truth_size는 현재 사용되지 않음
        # 실제 평가는 ground_truth_ids의 실제 개수를 사용
        self.gt_size = ground_truth_size

    def evaluate_query(
        self,
        retrieved_rec_idxs: List[str],
        ground_truth_rec_idxs: List[str],
        search_time: float = None,
    ) -> Dict[str, float]:
        """
        단일 쿼리에 대한 모든 지표 계산

        Args:
            retrieved_rec_idxs: 검색된 rec_idx 리스트 (순서대로, 최소 20개)
            ground_truth_rec_idxs: 정답 rec_idx 리스트 (개수는 가변적)
            search_time: 검색 시간 (초, 선택)
        Returns:
            {'ndcg@10': 0.72, 'recall@20': 0.8, 'mrr@10': 0.5, 'search_time': 0.421, ...}
        """
        # 사용자 요청 순서대로 지표 구성
        metrics = {
            # 1. NDCG@10
            "ndcg@10": self.calculate_ndcg_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=10
            ),
            # 2. MRR@10
            "mrr@10": self.calculate_mrr_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=10
            ),
            # 3. Precision@3
            "precision@3": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=3
            ),
            # 4. Precision@5
            "precision@5": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=5
            ),
            # 5. Precision@10
            "precision@10": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=10
            ),
            # 6. Precision@20
            "precision@20": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=20
            ),
            # 7. Hit@10_count: 상위 10개 중 맞은 개수
            "hit@10_count": len(
                set(retrieved_rec_idxs[:10]) & set(ground_truth_rec_idxs)
            ),
        }

        # 8. R-recall: recall@(정답개수)
        gt_count = len(ground_truth_rec_idxs)
        if gt_count > 0:
            metrics["r_recall"] = self.calculate_recall_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=gt_count
            )
        else:
            metrics["r_recall"] = 0.0

        # 추가 정보 (하위 호환성 및 디버깅용)
        metrics["recall@10"] = self.calculate_recall_at_k(
            retrieved_rec_idxs, ground_truth_rec_idxs, k=10
        )
        metrics["recall@20"] = self.calculate_recall_at_k(
            retrieved_rec_idxs, ground_truth_rec_idxs, k=20
        )
        metrics["hit@20_count"] = len(
            set(retrieved_rec_idxs[:20]) & set(ground_truth_rec_idxs)
        )
        metrics["hits@20"] = metrics["hit@20_count"]  # 하위 호환성
        metrics["total_gt"] = gt_count

        # 검색 시간 추가 (제공된 경우)
        if search_time is not None:
            metrics["search_time"] = search_time

        return metrics

    def calculate_ndcg_at_k(
        self, retrieved_ids: List[str], ground_truth_ids: List[str], k=10
    ) -> float:
        """

        NDCG@10 : 순위 품질 측정
        - DCG: 각 위치의 관련성 / log2(rank+1)
        - IDCG: 최적 순위의 DCG
        - NDCG: DCG / IDCG (0-1 정규화)

        Args:
            retrieved_ids: 검색된 rec_idx 순위
            ground_truth_ids: 정답
            k: 상위 k개 평가
        Return:
            0.0~ 1.0 (1.0이 가장 좋음)
        """
        gt_set = set(ground_truth_ids)

        # DCG 계산 (표준 공식: relevance / log2(rank + 1))
        # rank 1일 때 log2(2), rank 2일 때 log2(3), ...
        dcg = 0.0
        for i, rec_idx in enumerate(retrieved_ids[:k], start=1):
            relevance = 1.0 if rec_idx in gt_set else 0.0
            dcg += relevance / math.log2(i + 1)

        # IDCG 계산 (이상적 순서: 모든 정답이 맨 앞에)
        idcg = 0.0
        num_relevant = min(len(ground_truth_ids), k)
        for rank in range(1, num_relevant + 1):
            idcg += 1.0 / math.log2(rank + 1)
        # NDCG 정규화
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def calculate_recall_at_k(
        self, retrieved_ids: List[str], ground_truth_ids: List[str], k=20
    ) -> float:
        """
        Recall@k : 정답 재현율
        - 수식: (상위 k개에 포함된 정답 개수) / (전체 정답 개수)

        - 분모: 실제 ground_truth_ids의 개수 (가변적)
        - 분자: 상위 k개에 포함된 정답 개수
        - recall@k = hits / len(ground_truth_ids)

        Args:
            retrieved_ids: 검색된 rec_idx 결과
            ground_truth_ids: 정답 rec_idx 리스트 (개수는 가변적)
            k: 상위 k개 평가
        Returns:
            0.0~ 1.0 (1.0이 가장 좋음)
        Example:
            GT = [A, B, C, D, E] (5개)
            Retrieved@20 = [X, A, Y, B, Z, ..., C, ...]
            Hits = 3 (A, B, C)
            Recall@20 = 3 / 5 = 0.6

            GT = [A, B, C] (3개)
            Retrieved@20 = [X, A, Y, B, Z, ..., C, ...]
            Hits = 3 (A, B, C)
            Recall@20 = 3 / 3 = 1.0
        """
        if len(ground_truth_ids) == 0:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        gt_set = set(ground_truth_ids)

        hits = len(retrieved_set & gt_set)

        # 분모: 실제 ground_truth_ids의 개수 (가변적)
        return hits / len(ground_truth_ids)

    def calculate_mrr_at_k(
        self, retrieved_ids: List[str], ground_truth_ids: List[str], k=10
    ) -> float:
        """
        MRR@k : 첫 정답 순위의 역수
        수식 :
            - MRR = 1/ rank (첫번째 정답 위치)
        Args:
            retrieved_ids: 검색 결과 (순서 중요)
            ground_truth_ids: 정답 rec_idx 리스트 (개수는 가변적)
            k: 상위 k개에서만 찾기
        Returns:
            0.0~ 1.0 (1.0이 가장 좋음)
        Example:
            GT = [A, B, C, D, E] (5개)
            Retrieved@10 = [X, A, Y, B, Z, ..., C, ...]
            First relevant rank = 2 (A)
            MRR@10 = 1/2 = 0.5
        """
        gt_set = set(ground_truth_ids)

        for rank, rec_idx in enumerate(retrieved_ids[:k], start=1):
            if rec_idx in gt_set:
                return 1.0 / rank
        # 상위 k개 안에 정답 없음
        return 0.0

    def calculate_precision_at_k(
        self, retrieved_ids: List[str], ground_truth_ids: List[str], k: int
    ) -> float:
        """
        precision@k : 상위 k개 중 정답의 비율
        수식:
        - precision@k = (상위 k개에 포함된 정답 개수) / k
        Args:
            retrieved_ids: 검색된 rec_idx 결과
            ground_truth_ids: 정답 rec_idx 리스트 (개수는 가변적)
            k: 상위 k개 평가
        Returns:
            0.0~ 1.0 (1.0이 가장 좋음)
        Example:
            GT = [A, B, C, D, E] (5개)
            Retrieved@3 = [A, X, B]
            Hits = 2 (A, B)
            Precision@3 = 2 / 3 = 0.67
        """
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        gt_set = set(ground_truth_ids)

        hits = len(retrieved_set & gt_set)

        return hits / k

    def evaluate_all_queries(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        여러 쿼리의 평가 결과 집계
        Args: results:[{
            "query_id": "437" ,
            "retrieved_rec_idxs": ["123", "456", "789"],
            "ground_truth_rec_idxs": ["123", "456", "789"]
            "metrics":{...}
        }, ...]
        Returns:
            {
                "total_queries": 115,
                "average_metrics": {
                    "ndcg@10": 0.72,
                    ...
                },
                per_query_metrics: {... }
            }
        """
        total_queries = len(results)

        if total_queries == 0:
            return {"error": "No queries to evaluate"}

        # 각 지표별 합산 (사용자 요청 순서대로)
        metric_sums = {
            "ndcg@10": 0.0,
            "mrr@10": 0.0,
            "precision@3": 0.0,
            "precision@5": 0.0,
            "precision@10": 0.0,
            "precision@20": 0.0,
            "hit@10_count": 0.0,
            "r_recall": 0.0,
            # 하위 호환성을 위한 추가 지표
            "recall@10": 0.0,
            "recall@20": 0.0,
            "hit@20_count": 0.0,
        }
        per_query_metrics = {}

        for result in results:
            query_id = result["query_id"]
            metrics = result.get("metrics", {})

            per_query_metrics[query_id] = metrics

            for metric_name in metric_sums.keys():
                metric_sums[metric_name] += metrics.get(metric_name, 0.0)

        # 평균 계산
        average_metrics = {
            metric_name: total / total_queries
            for metric_name, total in metric_sums.items()
        }

        return {
            "total_queries": total_queries,
            "average_metrics": average_metrics,
            "per_query_metrics": per_query_metrics,
        }

    def evaluate(self, query_results: List[QueryResult]) -> List[EvaluationResult]:
        """
        QueryResult 리스트를 받아서 평가 지표를 계산하고 EvaluationResult 리스트 반환

        Args:
            query_results: QueryResult 객체 리스트

        Returns:
            EvaluationResult 리스트 (각 지표별)
        """
        if not query_results:
            return []

        # 각 쿼리별 지표 수집 (사용자 요청 순서대로)
        all_metrics = {
            "ndcg@10": [],
            "mrr@10": [],
            "precision@3": [],
            "precision@5": [],
            "precision@10": [],
            "precision@20": [],
            "hit@10_count": [],
            "r_recall": [],
            # 하위 호환성을 위한 추가 지표
            "recall@10": [],
            "recall@20": [],
            "hit@20_count": [],
        }

        evaluated_count = 0
        skipped_count = 0
        missing_rec_idx_count = 0

        for idx, query_result in enumerate(query_results):
            # retrieved_docs에서 rec_idx 추출
            retrieved_rec_idxs = []
            missing_in_retrieved = 0

            for doc in query_result.retrieved_docs:
                if isinstance(doc, dict):
                    metadata = doc.get("metadata", {})
                    if not metadata:
                        missing_in_retrieved += 1
                        continue

                    rec_idx = metadata.get("rec_idx") or metadata.get("rec_id")
                    if rec_idx:
                        retrieved_rec_idxs.append(str(rec_idx))
                    else:
                        missing_in_retrieved += 1
                else:
                    missing_in_retrieved += 1

            # ground_truth_docs에서 rec_idx 추출
            ground_truth_rec_idxs = []
            for gt in query_result.ground_truth_docs:
                if isinstance(gt, dict):
                    rec_idx = gt.get("rec_idx") or gt.get("rec_id")
                    if rec_idx:
                        ground_truth_rec_idxs.append(str(rec_idx))
                elif isinstance(gt, str):
                    ground_truth_rec_idxs.append(gt)

            # 각 쿼리 평가
            if not retrieved_rec_idxs:
                skipped_count += 1
                if idx < 3:  # 처음 3개만 디버깅 정보 출력
                    print(
                        f"⚠️  쿼리 {idx} 스킵: retrieved_docs에서 rec_idx 추출 실패 (총 {len(query_result.retrieved_docs)}개 중 {missing_in_retrieved}개 실패)"
                    )
                continue

            if not ground_truth_rec_idxs:
                skipped_count += 1
                if idx < 3:
                    print(f"⚠️  쿼리 {idx} 스킵: ground_truth_docs가 비어있음")
                continue

            # 평가 수행
            metrics = self.evaluate_query(retrieved_rec_idxs, ground_truth_rec_idxs)
            evaluated_count += 1

            # 지표별로 수집
            for metric_name in all_metrics.keys():
                if metric_name in metrics:
                    all_metrics[metric_name].append(metrics[metric_name])

        # 통계 정보
        if skipped_count > 0:
            print(
                f"📊 평가 통계: {evaluated_count}개 평가 완료, {skipped_count}개 스킵"
            )

        # 평균 계산 및 EvaluationResult 생성
        evaluation_results = []
        for metric_name, values in all_metrics.items():
            if values:
                avg_score = sum(values) / len(values)
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        score=avg_score,
                        details={
                            "total_queries": len(query_results),
                            "evaluated_queries": len(values),
                            "skipped_queries": skipped_count,
                        },
                    )
                )
            else:
                # 평가된 쿼리가 없으면 0.0 반환
                evaluation_results.append(
                    EvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        details={
                            "total_queries": len(query_results),
                            "evaluated_queries": 0,
                            "skipped_queries": skipped_count,
                            "error": "No valid queries to evaluate",
                        },
                    )
                )

        return evaluation_results


# ========================================
# 유틸리티 함수
# ========================================


def print_evaluation_summary(summary: Dict[str, Any]):
    """평가 결과 요약 출력"""
    print(f"\n{'='*60}")
    print(f"📊 검색 성능 평가 결과")
    print(f"{'='*60}")
    print(f"총 쿼리: {summary['total_queries']}개\n")

    avg_metrics = summary["average_metrics"]

    print("평균 지표 (사용자 요청 순서):")
    print(f"  1. NDCG@10:        {avg_metrics.get('ndcg@10', 0.0):.4f}")
    print(f"  2. MRR@10:         {avg_metrics.get('mrr@10', 0.0):.4f}")
    print(f"  3. Precision@3:     {avg_metrics.get('precision@3', 0.0):.4f}")
    print(f"  4. Precision@5:    {avg_metrics.get('precision@5', 0.0):.4f}")
    print(f"  5. Precision@10:   {avg_metrics.get('precision@10', 0.0):.4f}")
    print(f"  6. Precision@20:   {avg_metrics.get('precision@20', 0.0):.4f}")
    print(f"  7. Hit@10_count:   {avg_metrics.get('hit@10_count', 0.0):.2f}")
    print(f"  8. R-recall:       {avg_metrics.get('r_recall', 0.0):.4f}")
    print(f"\n추가 지표 (하위 호환성):")
    print(f"  Recall@10:        {avg_metrics.get('recall@10', 0.0):.4f}")
    print(f"  Recall@20:        {avg_metrics.get('recall@20', 0.0):.4f}")
    print(f"  Hit@20_count:     {avg_metrics.get('hit@20_count', 0.0):.2f}")
    print(f"{'='*60}\n")
