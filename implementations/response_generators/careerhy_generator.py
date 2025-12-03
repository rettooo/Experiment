"""
Career-HY 스타일 응답 생성기

실제 서비스와 동일한 방식으로 구조화된 응답을 생성
LangChain with_structured_output 방식 사용
"""

import json
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI

from core.interfaces.response_generator import (
    BaseResponseGenerator,
    GeneratedResponse,
    RecommendedJob,
    JobRecommendationResponse,
)
from services.prompt_builder import CareerHYPromptBuilder

logger = logging.getLogger(__name__)


class CareerHYResponseGenerator(BaseResponseGenerator):
    """Career-HY 실제 서비스와 동일한 응답 생성 (LangChain structured_output 방식)"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        # LangChain ChatOpenAI 초기화 (실제 서비스와 동일)
        self.llm = ChatOpenAI(
            model=model_name, temperature=temperature, max_tokens=max_tokens
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = CareerHYPromptBuilder()

    async def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
        config_tags: List[str] = None,  # 추가
    ) -> GeneratedResponse:
        """
        실제 서비스와 동일한 방식으로 응답 생성 (LangChain structured_output)

        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 채용공고 문서들
            user_profile: 사용자 프로필
            chat_history: 대화 이력

        Returns:
            GeneratedResponse: 생성된 구조화된 응답
        """

        try:
            # 1. 프롬프트 구성 (실제 서비스와 동일)
            prompt = self.prompt_builder.build_recommendation_prompt(
                query=query,
                retrieved_docs=retrieved_docs,
                user_profile=user_profile,
                chat_history=chat_history,
            )

            logger.debug(f"Generated prompt length: {len(prompt)} characters")

            # 2. 동적 태그 생성
            tags = (config_tags or []) + [
                "response-generation",
                f"retrieved-docs-{len(retrieved_docs)}",
            ]
            # 2. LangChain with_structured_output 방식 (실제 서비스와 동일)
            structured_llm = self.llm.with_structured_output(JobRecommendationResponse)
            result: JobRecommendationResponse = await structured_llm.ainvoke(
                prompt, config={"tags": tags}
            )

            logger.info(f"🤖 LLM Structured Output 결과:")
            logger.info(
                f"  - recommended_job_indices: {result.recommended_job_indices}"
            )
            logger.info(f"  - overall_advice 길이: {len(result.overall_advice)}")
            logger.info(
                f"  - recommendation_reasons 개수: {len(result.recommendation_reasons)}"
            )
            logger.info(f"  - practical_tips 길이: {len(result.practical_tips)}")

            # 3. 실제 서비스와 동일한 방식으로 응답 변환
            return self._convert_to_experiment_response(result, retrieved_docs)

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # 폴백: 기본 응답 생성
            return self._create_fallback_response(query, retrieved_docs)

    def _convert_to_experiment_response(
        self,
        structured_result: JobRecommendationResponse,
        retrieved_docs: List[Dict[str, Any]],
    ) -> GeneratedResponse:
        """
        LangChain structured_output 결과를 실험용 응답 형식으로 변환
        (실제 서비스와 동일한 로직)
        """

        # 전체 응답 텍스트 생성 (실제 서비스와 동일)
        if structured_result.recommended_job_indices:
            # 채용공고 추천이 있는 경우
            content = f"""
{structured_result.overall_advice}

실무 팁:
{structured_result.practical_tips}
""".strip()
        else:
            # 일반 상담/조언인 경우 (채용공고 추천 없음)
            content = f"""
{structured_result.overall_advice}

{structured_result.practical_tips}
""".strip()

        # 추천된 채용공고 파싱 (인덱스 → 실제 JobPosting 객체)
        recommended_jobs = []

        for i, job_index in enumerate(structured_result.recommended_job_indices):
            try:
                # 1-based index를 0-based로 변환
                doc_index = job_index - 1

                if 0 <= doc_index < len(retrieved_docs):
                    doc = retrieved_docs[doc_index]
                    metadata = doc.get("metadata", {})

                    # 추천 이유 가져오기 (인덱스에 맞춰)
                    reason = ""
                    if i < len(structured_result.recommendation_reasons):
                        reason = structured_result.recommendation_reasons[i]

                    rec_idx = metadata.get("rec_idx")
                    
                    # 디버깅: 첫 번째 추천 공고의 메타데이터 확인
                    if i == 0:
                        logger.info(f"🔍 [응답 생성] 첫 번째 추천 공고 메타데이터:")
                        logger.info(f"   - rec_idx: {rec_idx}")
                        logger.info(f"   - deadline: {metadata.get('deadline', 'N/A')}")
                        logger.info(f"   - start_date: {metadata.get('start_date', 'N/A')}")
                        logger.info(f"   - crawling_time: {metadata.get('crawling_time', 'N/A')}")
                        logger.info(f"   - 전체 메타데이터 키: {list(metadata.keys())[:15]}")
                    
                    recommended_job = RecommendedJob(
                        rec_idx=rec_idx,
                        title=metadata.get(
                            "title", metadata.get("post_title", "제목 없음")
                        ),
                        url=(
                            metadata.get("url")
                            or metadata.get("detail_url")
                            or (f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_idx}" if rec_idx else "")
                        ),
                        deadline=metadata.get("deadline"),
                        start_date=metadata.get("start_date"),
                        crawling_time=metadata.get("crawling_time"),
                        recommendation_reason=reason,
                    )
                    recommended_jobs.append(recommended_job)

                    logger.info(f"✅ 추천 공고 {job_index}: {recommended_job.title}")
                else:
                    logger.warning(
                        f"❌ 잘못된 추천 인덱스: {job_index} (범위: 1-{len(retrieved_docs)})"
                    )

            except Exception as e:
                logger.warning(f"Failed to process job recommendation {job_index}: {e}")
                continue

        return GeneratedResponse(content=content, recommended_jobs=recommended_jobs)

    def _create_fallback_response(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> GeneratedResponse:
        """오류 발생 시 기본 응답 생성"""

        content = f"""죄송합니다. 요청하신 '{query}'에 대한 상세한 분석이 어려웠습니다.
하지만 검색된 채용공고들을 바탕으로 관련성이 높은 공고들을 추천드립니다."""

        # 검색된 문서 중 상위 3개를 기본 추천으로 제공
        recommended_jobs = []
        for i, doc in enumerate(retrieved_docs[:3]):
            metadata = doc.get("metadata", {})
            rec_idx = metadata.get("rec_idx", f"fallback_{i}")
            recommended_job = RecommendedJob(
                rec_idx=rec_idx,
                title=metadata.get(
                    "title", metadata.get("post_title", f"채용공고 {i+1}")
                ),
                url=(
                    metadata.get("url")
                    or metadata.get("detail_url")
                    or (f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_idx}" if rec_idx and not rec_idx.startswith("fallback_") else "")
                ),
                deadline=metadata.get("deadline"),
                start_date=metadata.get("start_date"),
                crawling_time=metadata.get("crawling_time"),
                recommendation_reason=f"검색 결과 상위 {i+1}번째로 관련성이 높은 공고입니다.",
            )
            recommended_jobs.append(recommended_job)

        return GeneratedResponse(content=content, recommended_jobs=recommended_jobs)
