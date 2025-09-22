from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    metric_name: str
    score: float
    details: Dict[str, Any] = None


@dataclass
class QueryResult:
    """쿼리 결과 데이터 클래스"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    ground_truth_docs: List[str]  # 정답 문서 ID들
    llm_response: str = None


class BaseEvaluator(ABC):
    """평가 지표 계산의 추상 인터페이스"""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def evaluate(self, query_results: List[QueryResult]) -> List[EvaluationResult]:
        """
        쿼리 결과들에 대한 평가 수행

        Args:
            query_results: 쿼리 결과 리스트

        Returns:
            평가 결과 리스트
        """
        pass

    def get_supported_metrics(self) -> List[str]:
        """지원하는 평가 지표 목록 반환"""
        return []


class SearchEvaluator(BaseEvaluator):
    """검색 성능 평가기"""

    @abstractmethod
    def calculate_recall_at_k(self, query_results: List[QueryResult], k: int) -> float:
        """Recall@k 계산"""
        pass

    @abstractmethod
    def calculate_precision_at_k(self, query_results: List[QueryResult], k: int) -> float:
        """Precision@k 계산"""
        pass

    @abstractmethod
    def calculate_mrr(self, query_results: List[QueryResult]) -> float:
        """Mean Reciprocal Rank 계산"""
        pass

    @abstractmethod
    def calculate_map(self, query_results: List[QueryResult]) -> float:
        """Mean Average Precision 계산"""
        pass

    @abstractmethod
    def calculate_ndcg(self, query_results: List[QueryResult], k: int) -> float:
        """Normalized Discounted Cumulative Gain@k 계산"""
        pass


class LLMEvaluator(BaseEvaluator):
    """LLM 생성 품질 평가기"""

    @abstractmethod
    def evaluate_relevance(self, query_results: List[QueryResult]) -> List[EvaluationResult]:
        """프로필/사용자 입력 적합성 평가"""
        pass

    @abstractmethod
    def evaluate_requirement_satisfaction(self, query_results: List[QueryResult]) -> List[EvaluationResult]:
        """요구사항 충족도 평가"""
        pass