import math
from typing import List, Dict, Any, Set


class RetrieverEvaluator:
    """검색 성능 지표 계산 class (평가지표 수정!!)"""

    def __init__(self, ground_truth_size: int = 5):
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
            ground_truth_rec_idxs : 정답 rec_idx 리스트 (5개)
            search_time: 검색 시간 (초, 선택)
        Returns:
            {'ndcg@10': 0.72, 'recall@20': 0.8, 'mrr@10': 0.5, 'search_time': 0.421, ...}
        """
        metrics = {
            "ndcg@10": self.calculate_ndcg_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=10
            ),
            "recall@20": self.calculate_recall_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=20
            ),
            "mrr@10": self.calculate_mrr_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=10
            ),
            "precision@3": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=3
            ),
            "precision@5": self.calculate_precision_at_k(
                retrieved_rec_idxs, ground_truth_rec_idxs, k=5
            ),
        }
        # 추가 정보
        metrics["hits@20"] = len(
            set(retrieved_rec_idxs[:20]) & set(ground_truth_rec_idxs)
        )
        metrics["total_gt"] = len(ground_truth_rec_idxs)

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

        # DCG 계산
        # DCG 계산
        dcg = 0.0
        for i, rec_idx in enumerate(retrieved_ids[:k], start=1):
            relevance = 1.0 if rec_idx in gt_set else 0.0
            dcg += relevance / math.log2(i + 1)
        # IDCG 계산 (이상적 순서: 모든 정답이 맨 앞에)
        idcg = 0.0
        for i in range(1, min(len(ground_truth_ids), k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        # NDCG 정규화
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def calculate_recall_at_k(
        self, retrieved_ids: List[str], ground_truth_ids: List[str], k=20
    ) -> float:
        """
        Recall@20 : 정답 재현율
        - 수식: (상위 k개에 포함된 정답 개수) / (전체 정답 개수)

        - 분모 : 항상 gt 의 개수 (5개)
        - 분자 : 상위 k개에 포함된 정답 개수
        - recall@20 = hits / 5

        Args:
            retreived_ids: 검색된 rec_idx 결과
            ground_truth_ids: 정답 rec_idx 리스트
            k : 상위 k개 vudrk
        Returns:
            0.0~ 1.0 (1.0이 가장 좋음)
        Example:
            GT = [A, B, C, D, E] (5개)
            Retrieved@20 = [X, A, Y, B, Z, ..., C, ...]
            Hits = 3 (A, B, C)
            Recall@20 = 3 / 5 = 0.6
        """
        if len(ground_truth_ids) == 0:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        gt_set = set(ground_truth_ids)

        hits = len(retrieved_set & gt_set)

        # 분모: 전체 정답 개수 5개로 고정
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
            ground_truth_ids: 정답 rec_idx 리스트
            K: 상위 k개에서만 찾기
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
        precision@k : 상위 K개중 정답의 비율
        수식:
        - precision@k = (상위 k개에 포함된 정답 개수) / k
        Args:
            retrieved_ids: 검색된 rec_idx 결과
            ground_truth_ids: 정답 rec_idx 리스트
            k: 상위 k개 vudrk
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

        return hits / k  # k 3,5고정

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

        # 각 지표별 합산
        metric_sums = {
            "ndcg@10": 0.0,
            "recall@20": 0.0,
            "mrr@10": 0.0,
            "precision@3": 0.0,
            "precision@5": 0.0,
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

    print("평균 지표:")
    print(f"  NDCG@10:      {avg_metrics['ndcg@10']:.4f}")
    print(f"  Recall@20:    {avg_metrics['recall@20']:.4f}")
    print(f"  MRR@10:       {avg_metrics['mrr@10']:.4f}")
    print(f"  Precision@3:  {avg_metrics['precision@3']:.4f}")
    print(f"  Precision@5:  {avg_metrics['precision@5']:.4f}")
    print(f"{'='*60}\n")


# ========================================
# 테스트 코드 (실제 evaluation_queries.jsonl 형식)
# ========================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 RetrieverEvaluator 테스트 (실제 데이터 형식)")
    print("=" * 70)

    evaluator = RetrieverEvaluator(ground_truth_size=5)

    # 📊 실제 evaluation_queries.jsonl 형식 시뮬레이션
    query_data = {
        "query_id": "437",
        "query_text": "전공: 생명공학\n관심 직무: 생명공학 연구원...",
        "ground_truth": [
            {"rec_idx": "50436465", "job_title": "[한국콜마] 연구전략", "url": "..."},
            {
                "rec_idx": "50436592",
                "job_title": "[한국콜마] 컴플라이언스",
                "url": "...",
            },
            {"rec_idx": "50436291", "job_title": "[한국콜마] 마케팅", "url": "..."},
            {"rec_idx": "50436627", "job_title": "[한국콜마] 생산관리", "url": "..."},
            {"rec_idx": "50436344", "job_title": "[한국콜마] 소재개발", "url": "..."},
        ],
    }

    # ✅ Pipeline에서 하는 것처럼 rec_idx 추출
    gt_rec_idxs = [str(gt["rec_idx"]) for gt in query_data["ground_truth"]]
    print(f"\n📌 Query ID: {query_data['query_id']}")
    print(f"📌 Ground Truth (GT): {len(gt_rec_idxs)}개")
    print(f"   GT rec_idx: {gt_rec_idxs}\n")

    # ========================================
    # 시나리오 1: 완벽한 검색 (모든 정답이 상위 5개에)
    # ========================================
    print("\n" + "=" * 70)
    print("✅ 시나리오 1: 완벽한 검색 (상위 5개에 모든 GT 포함)")
    print("=" * 70)

    perfect_retrieval = gt_rec_idxs + [f"9999{i}" for i in range(15)]  # 20개
    print(f"검색 결과 (상위 10개): {perfect_retrieval[:10]}")

    metrics1 = evaluator.evaluate_query(perfect_retrieval, gt_rec_idxs)
    print("\n📊 평가 지표:")
    for metric, value in metrics1.items():
        print(f"  {metric:15s}: {value}")

    # ========================================
    # 시나리오 2: 일부 정답만 검색 (3개만 상위 10개에)
    # ========================================
    print("\n" + "=" * 70)
    print("⚠️  시나리오 2: 일부 정답만 검색 (상위 10개에 3개만)")
    print("=" * 70)

    # 1위: 오답, 2위: GT[0], 4위: GT[1], 8위: GT[2], 나머지는 하위
    partial_retrieval = [
        "88888888",  # 1위: 오답
        gt_rec_idxs[0],  # 2위: 50436465 ✅
        "99999999",  # 3위: 오답
        gt_rec_idxs[1],  # 4위: 50436592 ✅
        "77777777",  # 5위: 오답
        "66666666",  # 6위: 오답
        "55555555",  # 7위: 오답
        gt_rec_idxs[2],  # 8위: 50436291 ✅
        "44444444",  # 9위: 오답
        "33333333",  # 10위: 오답
    ] + [
        f"1111{i}" for i in range(10)
    ]  # 11-20위: 오답

    print(f"검색 결과 (상위 10개):")
    for i, rec_idx in enumerate(partial_retrieval[:10], start=1):
        is_gt = "✅ GT" if rec_idx in gt_rec_idxs else "❌"
        print(f"  {i:2d}위: {rec_idx} {is_gt}")

    metrics2 = evaluator.evaluate_query(partial_retrieval, gt_rec_idxs)
    print("\n📊 평가 지표:")
    for metric, value in metrics2.items():
        print(f"  {metric:15s}: {value}")

    # ========================================
    # 시나리오 3: 정답이 하위에 (15위 이후)
    # ========================================
    print("\n" + "=" * 70)
    print("❌ 시나리오 3: 정답이 하위에 (15-20위)")
    print("=" * 70)

    poor_retrieval = [
        f"9999{i}" for i in range(15)
    ] + gt_rec_idxs  # 1-15위 오답, 16-20위 정답
    print(f"검색 결과 (상위 10개): {poor_retrieval[:10]}")
    print(f"검색 결과 (16-20위): {poor_retrieval[15:20]}")

    metrics3 = evaluator.evaluate_query(poor_retrieval, gt_rec_idxs)
    print("\n📊 평가 지표:")
    for metric, value in metrics3.items():
        print(f"  {metric:15s}: {value}")

    # ========================================
    # 전체 결과 비교
    # ========================================
    print("\n" + "=" * 70)
    print("📊 전체 시나리오 비교")
    print("=" * 70)

    results = [
        {"query_id": "시나리오1_완벽", "metrics": metrics1},
        {"query_id": "시나리오2_일부", "metrics": metrics2},
        {"query_id": "시나리오3_하위", "metrics": metrics3},
    ]

    summary = evaluator.evaluate_all_queries(results)
    print_evaluation_summary(summary)

    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)
