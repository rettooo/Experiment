"""
응답 생성기 인터페이스

검색된 문서들과 사용자 프로필을 바탕으로
실제 서비스와 동일한 방식의 응답을 생성하는 추상 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class RecommendedJob(BaseModel):
    """추천된 채용공고 (실제 서비스와 동일한 구조)"""
    rec_idx: Optional[str] = None
    title: str
    url: str
    deadline: Optional[str] = None
    start_date: Optional[str] = None
    crawling_time: Optional[str] = None
    recommendation_reason: str


class GeneratedResponse(BaseModel):
    """생성된 응답 (실제 서비스와 동일한 구조)"""
    content: str
    recommended_jobs: List[RecommendedJob] = []


# 실제 서비스와 동일한 구조화된 응답 모델 (LangChain with_structured_output용)
class JobRecommendationResponse(BaseModel):
    """채용공고 추천 응답 구조 (실제 서비스와 동일)"""

    recommended_job_indices: List[int] = Field(
        description="추천하는 채용공고의 번호 (1-10), 채용공고 추천이 불필요한 경우 빈 배열",
        max_items=3,
        min_items=0,  # 실험에서는 추천이 없을 수도 있으므로 0으로 설정
    )
    overall_advice: str = Field(
        description="전반적인 취업 준비 방향성과 조언, 또는 질문에 대한 답변"
    )
    recommendation_reasons: List[str] = Field(
        description="각 추천 채용공고의 추천 이유 설명 자세히, 채용공고 추천이 없으면 빈 배열",
        max_items=3,
        min_items=0,  # 실험에서는 추천이 없을 수도 있으므로 0으로 설정
    )
    practical_tips: str = Field(
        description="지원 시 도움이 될 수 있는 구체적인 팁, 또는 추가 조언"
    )


class BaseResponseGenerator(ABC):
    """응답 생성기 기본 인터페이스"""

    @abstractmethod
    async def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> GeneratedResponse:
        """
        사용자 쿼리와 검색된 문서들을 바탕으로 응답 생성

        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 채용공고 문서들
            user_profile: 사용자 프로필 (전공, 수강과목, 자격증 등)
            chat_history: 대화 이력 (선택사항)

        Returns:
            GeneratedResponse: 생성된 응답 (content + recommended_jobs)
        """
        pass