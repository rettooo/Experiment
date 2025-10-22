"""
LangSmith 기반 고품질 응답 평가

LLM-as-Judge 방식을 사용하여 응답 생성 품질을 정교하게 평가
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI


@dataclass
class LangSmithEvaluationResult:
    """LangSmith 평가 결과"""

    metric_name: str
    score: float
    reasoning: str
    details: Dict[str, Any]


class CareerHYLangSmithEvaluator:
    """Career-HY 맞춤형 LangSmith 평가기"""

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        project_name: str = "career-hy-rag-evaluation",
        base_tags: List[str] = None,
    ):
        self.judge_model = judge_model
        self.project_name = project_name
        self.base_tags = base_tags or []
        self.client = Client()
        self.llm = ChatOpenAI(model=judge_model, temperature=0.1)

        # 평가 지표들
        self.metrics = [
            "recommendation_quality",
            "personalization_score",
            "response_helpfulness",
            "profile_alignment",
        ]

    async def evaluate_batch(
        self, query_results: List[Dict[str, Any]], experiment_name: str
    ) -> List[LangSmithEvaluationResult]:
        """
        여러 쿼리 결과에 대한 LangSmith 배치 평가

        Args:
            query_results: 쿼리별 결과 리스트
            experiment_name: 실험명 (LangSmith 추적용)

        Returns:
            평가 결과 리스트
        """

        print(f"🔍 LangSmith 평가 시작: {len(query_results)}개 쿼리")
        print(f"🤖 Judge 모델: {self.judge_model}")
        print(f"📊 평가 지표: {self.metrics}")

        # 직접 평가 실행
        final_results = []

        for metric_name in self.metrics:
            print(f"\n📈 {metric_name} 평가 중...")

            try:
                scores = []
                reasonings = []

                for i, result in enumerate(query_results):
                    evaluation_result = await self._evaluate_single_query(
                        result, metric_name
                    )
                    scores.append(evaluation_result["score"])
                    reasonings.append(evaluation_result["reasoning"])

                avg_score = sum(scores) / len(scores) if scores else 0.0

                final_results.append(
                    LangSmithEvaluationResult(
                        metric_name=metric_name,
                        score=avg_score,
                        reasoning=f"평균 점수: {avg_score:.3f}",
                        details={
                            "individual_scores": scores,
                            "individual_reasonings": reasonings,
                            "total_queries": len(query_results),
                        },
                    )
                )

                print(f"✅ {metric_name} 평가 완료: {avg_score:.3f}")

            except Exception as e:
                print(f"❌ {metric_name} 평가 실패: {e}")
                final_results.append(
                    LangSmithEvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        reasoning=f"평가 실패: {e}",
                        details={"error": str(e)},
                    )
                )

        print(f"\n🎉 LangSmith 평가 완료!")
        for result in final_results:
            print(f"  {result.metric_name}: {result.score:.3f}")

        return final_results

    async def _evaluate_single_query(
        self, query_result: Dict[str, Any], metric_name: str
    ) -> Dict[str, Any]:
        """단일 쿼리에 대한 특정 지표 평가"""

        try:
            query = query_result.get("query", "")
            user_profile = query_result.get("user_profile", {})
            generated_response = query_result.get("generated_response", {})
            alternative_query = query_result.get("alternative_query", "")

            # 메트릭별 평가 프롬프트 생성
            if metric_name == "recommendation_quality":
                prompt = self._create_recommendation_quality_prompt(
                    query, user_profile, generated_response, alternative_query
                )
            elif metric_name == "personalization_score":
                prompt = self._create_personalization_prompt(
                    query, user_profile, generated_response, alternative_query
                )
            elif metric_name == "response_helpfulness":
                prompt = self._create_helpfulness_prompt(
                    query, user_profile, generated_response, alternative_query
                )
            elif metric_name == "profile_alignment":
                prompt = self._create_profile_alignment_prompt(
                    user_profile, generated_response
                )
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

            # LLM으로 평가 실행
            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            # 점수와 이유 추출
            score = self._extract_score(response_text)
            reasoning = self._extract_reasoning(response_text)

            return {
                "score": score,
                "reasoning": reasoning,
                "full_response": response_text,
            }

        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"평가 실패: {e}",
                "full_response": str(e),
            }

    # 추천 품질 평가 프롬프트 생성
    def _create_recommendation_quality_prompt(
        self,
        query: str,
        user_profile: Dict,
        generated_response: Dict,
        alternative_query: str = "",
    ) -> str:
        """추천 품질 평가 프롬프트 생성"""
        # alternative_query가 있으면 사용, 없으면 query 사용
        display_query = alternative_query if alternative_query else query
        # 평가 대상 : 추천 공고 리스트, 조언 / 설명 텍스트
        recommended_jobs = json.dumps(
            generated_response.get("recommended_jobs", []), ensure_ascii=False, indent=2
        )
        recommendation_content = generated_response.get("content", "")
        return f"""
다음 채용공고 추천의 품질을 1-5점으로 평가해주세요.

[정보]
사용자 의도(최종 질문): {display_query}

사용자 프로필: 
- 전공: {user_profile.get('major', 'N/A')}
- 관심직무: {user_profile.get('interest_job', 'N/A')}


[평가 대상]
1. 추천 공고 리스트: {recommended_jobs}
2. 생성된 조언 및 설명: {recommendation_content}

평가 기준:
- 추천된 공고(1)가 '사용자 의도' 및 '프로필'과 얼마나 일치하는지 
- 생성된 조언(2)이 '사용자 의도'에 대해 얼마나 구체적이고 유용한 답변을 하는지 종합적으로 평가합니다.

[점수 기준]
- 5점( 매우우수): 5점 (매우 우수): 추천 공고가 '사용자 의도' 및 '프로필'과 완벽히 일치합니다. '생성된 조언'은 매우 구체적이고, 추천 이유가 명확하며 실질적으로 유용합니다.
- 4점 (우수): 대부분의 공고가 일치하며 조언도 유용합니다.
    **(중요!) 만약 추천 공고가 사용자 의도와 다소 다르더라도, 사용자 프로필에 더 적합하다고 판단하여 '생성된 조언'에서 그 이유를 명확히 설명했다면 4-5점을 부여할 수 있습니다. 
- 3점 (보통): 일부 공고만 일치하거나 조언이 부족합니다.
- 2점 (부족):대부분의 공고가 관련 없거나, 조언이 질문과 동떨어져 있습니다.
- 1점 (미흡): 완전히 부적절한 추천입니다.

점수 (1-5점):
이유:
"""

    def _create_personalization_prompt(
        self,
        query: str,
        user_profile: Dict,
        generated_response: Dict,
        alternative_query: str = "",
    ) -> str:
        """개인화 평가 프롬프트 생성"""
        display_query = alternative_query if alternative_query else query
        recommendation_content = generated_response.get("content", "")
        return f"""
다음 '추천 조언'이 사용자에게 얼마나 '개인화'되어 있는지 1-5점으로 평가해주세요.
(이 평가는 추천된 공고 리스트가 아닌, '생성된 조언' 텍스트 자체에 집중합니다.)

[정보]
사용자 의도(최종 질문): {display_query}

사용자 프로필:
- 전공: {user_profile.get('major', 'N/A')}
- 관심 직무: {user_profile.get('interest_job', 'N/A')}

[평가대상]
- 생성된 조언:
{recommendation_content}

[평가 기준]
- 이 조언이 '사용자 프로필'(전공, 관심 직무)의 요소를 얼마나 구체적으로 '언급'하고 '반영'하여 맞춤형 조언을 제공하는지 평가합니다.

[점수 기준]
- 5점 (매우 우수): 사용자의 전공, 관심 직무를 명확히 언급하며, 이와 직접적으로 연결된 구체적인 행동(예: '경영학 전공이시니 OOO 경험을 강조하세요')을 제안합니다.
- 3점 (보통): 프로필 요소를 언급하기는 하나, '관련 경험을 쌓으세요'처럼 일반적인 수준의 조언에 그칩니다.
- 1점 (매우 미흡): 프로필을 전혀 반영하지 않은(예: "당신에게 맞는 공고를 추천합니다") 템플릿 형태의 일반적인 조언입니다.

점수 (1-5점):
이유:
"""

    def _create_helpfulness_prompt(
        self,
        query: str,
        user_profile: Dict,
        generated_response: Dict,
        alternative_query: str = "",
    ) -> str:
        """도움이 되는 정도 평가 프롬프트 생성"""
        display_query = alternative_query if alternative_query else query
        recommendation_content = generated_response.get("content", "")
        return f"""
다음 '추천 조언'이 사용자에게 '실질적으로 얼마나 도움이 되는지' 1-5점으로 평가해주세요.
(이 평가는 추천된 공고 리스트가 아닌, '생성된 조언' 텍스트 자체에 집중합니다.)

[정보]
- 사용자 의도 (최종 질문): {display_query}

[평가대상]
- 생성된 조언:
{recommendation_content}

[평가 기준]
- 이 조언이 사용자의 질문과 프로필을 바탕으로, 취업 준비에 실질적인 도움이 되는 '구체적이고 실행 가능한(actionable)' 정보를 제공하는지 평가합니다.

[점수 기준]
- 5점 (매우 우수): '인턴십 지원', '포트폴리오에 OOO 프로젝트 추가', '관련 자격증 X, Y 취득' 등 매우 구체적이고, 당장 실행 가능한 실용적인 조언을 포함합니다.
- 3점 (보통): '경험을 쌓는 것이 좋습니다', '자격증을 알아보세요'처럼 방향성은 맞지만 다소 일반적이고 원론적인 조언입니다.
- 1점 (매우 미흡): 도움이 되지 않거나, 질문과 동문서답하는 내용입니다.

점수 (1-5점):
이유:
"""

    def _create_profile_alignment_prompt(
        self, user_profile: Dict, generated_response: Dict
    ) -> str:
        """프로필 일치도 평가 프롬프트 생성"""
        recommended_jobs = json.dumps(
            generated_response.get("recommended_jobs", []), ensure_ascii=False, indent=2
        )
        return f"""
추천된 '채용 공고 리스트'가 '사용자 프로필'과 얼마나 일치하는지 1-5점으로 평가해주세요.
(이 평가는 '사용자 질문'이나 '생성된 조언'을 **무시**하고, 오직 추천된 공고가 '사용자 프로필'과 얼마나 일치하는지에 집중합니다.)

[정보]
사용자 프로필:
- 전공: {user_profile.get('major', 'N/A')}
- 수강이력: {user_profile.get('courses', [])}

[평가 대상]
추천된 공고들:
{recommended_jobs}

[점수 기준]
- 5점 (매우 우수): 모든 추천 공고가 사용자의 전공 또는 수강이력 명확하게 일치합니다.
- 3점 (보통): 일부 공고는 일치하지만, 관련성이 낮은 공고가 20-40% 포함되어 있습니다.
- 1점 (매우 미흡): 추천된 공고가 프로필과 거의 또는 전혀 일치하지 않습니다.

점수 (1-5점):
이유:
"""

    def _extract_score(self, response_text: str) -> float:
        """응답에서 점수 추출"""
        try:
            lines = response_text.strip().split("\n")
            for line in lines:
                if "점수" in line and ":" in line:
                    score_part = line.split(":")[1].strip()
                    import re

                    numbers = re.findall(r"\d+\.?\d*", score_part)
                    if numbers:
                        score = float(numbers[0])
                        return max(1.0, min(5.0, score))  # 1-5 범위로 제한
            return 3.0  # 기본값
        except:
            return 3.0

    def _extract_reasoning(self, response_text: str) -> str:
        """응답에서 이유 추출"""
        try:
            lines = response_text.strip().split("\n")
            for line in lines:
                if "이유" in line and ":" in line:
                    return line.split(":")[1].strip()
            return "이유를 추출할 수 없음"
        except:
            return "이유 추출 실패"

    async def _create_langsmith_dataset(
        self, query_results: List[Dict[str, Any]], dataset_name: str
    ) -> str:
        """LangSmith 데이터셋 생성"""

        try:
            # 기존 데이터셋 삭제 (있다면)
            try:
                self.client.delete_dataset(dataset_name=dataset_name)
            except:
                pass

            # 새 데이터셋 생성
            dataset = self.client.create_dataset(
                dataset_name=dataset_name, description=f"Career-HY RAG 평가용 데이터셋"
            )

            # 데이터 추가 (새로운 API 방식)
            examples = []
            for i, result in enumerate(query_results):
                example = self.client.create_example(
                    dataset_id=dataset.id,
                    inputs={
                        "query": result.get("query", ""),
                        "user_profile": result.get("user_profile", {}),
                        "retrieved_docs": result.get("retrieved_docs", []),
                        "generated_response": result.get("generated_response", {}),
                        "ground_truth_docs": result.get("ground_truth_docs", []),
                    },
                    outputs={"expected_quality": "high"},  # 기본값
                )
                examples.append(example)

            print(
                f"📁 LangSmith 데이터셋 생성: {dataset_name} ({len(examples)}개 항목)"
            )
            return dataset_name

        except Exception as e:
            print(f"❌ 데이터셋 생성 실패: {e}")
            raise

    def _format_response_for_evaluation(self, inputs: Dict[str, Any]) -> str:
        """평가를 위한 응답 포맷팅"""
        generated_response = inputs.get("generated_response", {})
        return json.dumps(generated_response, ensure_ascii=False)

    def _aggregate_evaluation_results(
        self, evaluation_results: Dict[str, Any]
    ) -> List[LangSmithEvaluationResult]:
        """평가 결과 집계"""

        final_results = []

        for metric_name, results in evaluation_results.items():
            if results is None:
                # 평가 실패한 경우 기본값
                final_results.append(
                    LangSmithEvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        reasoning="평가 실패",
                        details={"error": "evaluation_failed"},
                    )
                )
                continue

            # 결과에서 평균 점수 계산
            try:
                scores = []
                reasoning_samples = []

                for result in results:
                    if hasattr(result, "results") and result.results:
                        for eval_result in result.results:
                            if (
                                hasattr(eval_result, "score")
                                and eval_result.score is not None
                            ):
                                scores.append(float(eval_result.score))

                            if hasattr(eval_result, "comment") and eval_result.comment:
                                reasoning_samples.append(eval_result.comment)

                avg_score = sum(scores) / len(scores) if scores else 0.0
                sample_reasoning = (
                    reasoning_samples[0] if reasoning_samples else "평가 완료"
                )

                final_results.append(
                    LangSmithEvaluationResult(
                        metric_name=metric_name,
                        score=avg_score,
                        reasoning=sample_reasoning,
                        details={
                            "total_evaluations": len(scores),
                            "score_distribution": {
                                "min": min(scores) if scores else 0,
                                "max": max(scores) if scores else 0,
                                "avg": avg_score,
                            },
                        },
                    )
                )

            except Exception as e:
                print(f"⚠️  {metric_name} 결과 집계 실패: {e}")
                final_results.append(
                    LangSmithEvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        reasoning=f"집계 실패: {e}",
                        details={"error": "aggregation_failed"},
                    )
                )

        return final_results
