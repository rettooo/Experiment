"""
자동화된 응답 생성 품질 평가

LLM Judge 없이 자동화된 지표들로 응답 생성 품질을 측정
비용 효율적이면서도 의미 있는 평가를 제공
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationEvaluationResult:
    """생성 평가 결과"""
    metric_name: str
    score: float
    details: Dict[str, Any]


class GenerationEvaluator:
    """자동화된 응답 생성 품질 평가기"""

    def __init__(self):
        self.metrics = [
            "recommendation_accuracy",
            "profile_utilization",
            "response_completeness",
            "structure_quality"
        ]

    def evaluate_batch(
        self,
        query_results: List[Dict[str, Any]]
    ) -> List[GenerationEvaluationResult]:
        """
        여러 쿼리 결과에 대한 배치 평가

        Args:
            query_results: 쿼리별 결과 리스트
                각 항목: {
                    'query': str,
                    'user_profile': dict,
                    'ground_truth_docs': list,
                    'generated_response': dict,
                    'retrieved_docs': list
                }

        Returns:
            평가 결과 리스트
        """

        print(f"🔍 생성 품질 평가 시작: {len(query_results)}개 쿼리")

        # 각 지표별 점수 수집
        metric_scores = {metric: [] for metric in self.metrics}
        detailed_results = []

        for i, result in enumerate(query_results):
            try:
                # 개별 쿼리 평가
                individual_scores = self._evaluate_single_query(result)

                # 점수 수집
                for metric, score in individual_scores.items():
                    if metric in metric_scores:
                        metric_scores[metric].append(score)

                detailed_results.append({
                    'query_index': i,
                    'query': result['query'][:50] + '...',
                    'scores': individual_scores
                })

                if (i + 1) % 10 == 0:
                    print(f"  평가 진행률: {i + 1}/{len(query_results)}")

            except Exception as e:
                print(f"  ⚠️  쿼리 {i} 평가 실패: {e}")
                # 기본 점수로 채움
                for metric in self.metrics:
                    metric_scores[metric].append(0.0)

        # 평균 점수 계산 및 결과 생성
        evaluation_results = []
        for metric in self.metrics:
            scores = metric_scores[metric]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            evaluation_results.append(GenerationEvaluationResult(
                metric_name=metric,
                score=avg_score,
                details={
                    'total_queries': len(query_results),
                    'successful_evaluations': len([s for s in scores if s > 0]),
                    'score_distribution': {
                        'min': min(scores) if scores else 0,
                        'max': max(scores) if scores else 0,
                        'avg': avg_score
                    }
                }
            ))

        print(f"✅ 생성 품질 평가 완료")
        for result in evaluation_results:
            print(f"  {result.metric_name}: {result.score:.4f}")

        return evaluation_results

    def _evaluate_single_query(self, result: Dict[str, Any]) -> Dict[str, float]:
        """단일 쿼리에 대한 평가"""

        query = result.get('query', '')
        user_profile = result.get('user_profile', {})
        ground_truth_docs = result.get('ground_truth_docs', [])
        generated_response = result.get('generated_response', {})
        retrieved_docs = result.get('retrieved_docs', [])

        scores = {}

        # 1. 추천 정확도
        scores['recommendation_accuracy'] = self._calculate_recommendation_accuracy(
            generated_response, ground_truth_docs
        )

        # 2. 프로필 활용도
        scores['profile_utilization'] = self._calculate_profile_utilization(
            generated_response, user_profile
        )

        # 3. 응답 완성도
        scores['response_completeness'] = self._calculate_response_completeness(
            generated_response, query
        )

        # 4. 구조 품질
        scores['structure_quality'] = self._calculate_structure_quality(
            generated_response
        )

        return scores

    def _calculate_recommendation_accuracy(
        self,
        generated_response: Dict[str, Any],
        ground_truth_docs: List[str]
    ) -> float:
        """추천 정확도: 추천된 공고 중 실제 관련 공고 비율"""

        try:
            recommended_jobs = generated_response.get('recommended_jobs', [])

            if not recommended_jobs:
                return 0.0

            if not ground_truth_docs:
                return 0.5  # GT가 없으면 중립 점수

            # 추천된 공고의 rec_idx 추출
            recommended_rec_ids = []
            for job in recommended_jobs:
                rec_idx = job.get('rec_idx')
                if rec_idx:
                    recommended_rec_ids.append(str(rec_idx))

            if not recommended_rec_ids:
                return 0.0

            # GT와 추천 결과의 교집합 계산
            gt_set = set(str(doc) for doc in ground_truth_docs)
            recommended_set = set(recommended_rec_ids)

            intersection = gt_set & recommended_set
            accuracy = len(intersection) / len(recommended_set)

            return accuracy

        except Exception as e:
            print(f"추천 정확도 계산 실패: {e}")
            return 0.0

    def _calculate_profile_utilization(
        self,
        generated_response: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        """프로필 활용도: 응답에서 사용자 프로필 요소 언급 비율"""

        try:
            response_content = generated_response.get('content', '')
            if not response_content:
                return 0.0

            response_lower = response_content.lower()

            # 체크할 프로필 요소들
            profile_elements = []

            # 전공
            major = user_profile.get('major', '')
            if major:
                profile_elements.append(('major', major))

            # 관심 직무
            interest_jobs = user_profile.get('interest_job', [])
            if isinstance(interest_jobs, list):
                for job in interest_jobs:
                    profile_elements.append(('interest_job', job))
            elif interest_jobs:
                profile_elements.append(('interest_job', interest_jobs))

            # 자격증
            certifications = user_profile.get('certification', [])
            if isinstance(certifications, list):
                for cert in certifications:
                    profile_elements.append(('certification', cert))
            elif certifications:
                profile_elements.append(('certification', certifications))

            # 수강 과목 (catalogs에서 추출)
            catalogs = user_profile.get('catalogs', [])
            if isinstance(catalogs, list):
                for catalog in catalogs[:5]:  # 상위 5개만 체크
                    if isinstance(catalog, dict):
                        course_name = catalog.get('course_name', '')
                        if course_name:
                            profile_elements.append(('course', course_name))

            if not profile_elements:
                return 0.5  # 프로필 정보가 없으면 중립 점수

            # 언급된 요소 카운트
            mentioned_count = 0
            for element_type, element_value in profile_elements:
                if self._is_mentioned_in_text(element_value, response_lower):
                    mentioned_count += 1

            utilization_score = mentioned_count / len(profile_elements)
            return utilization_score

        except Exception as e:
            print(f"프로필 활용도 계산 실패: {e}")
            return 0.0

    def _calculate_response_completeness(
        self,
        generated_response: Dict[str, Any],
        query: str
    ) -> float:
        """응답 완성도: 응답이 얼마나 완전한가"""

        try:
            content = generated_response.get('content', '')
            recommended_jobs = generated_response.get('recommended_jobs', [])

            score_components = []

            # 1. 내용 존재 여부 (0.3)
            if content and len(content.strip()) >= 20:
                score_components.append(0.3)
            else:
                score_components.append(0.0)

            # 2. 추천 공고 존재 여부 (0.3)
            if recommended_jobs and len(recommended_jobs) > 0:
                score_components.append(0.3)
            else:
                score_components.append(0.0)

            # 3. 추천 이유 존재 여부 (0.4)
            reason_score = 0.0
            for job in recommended_jobs:
                reason = job.get('recommendation_reason', '')
                if reason and len(reason.strip()) >= 10:
                    reason_score += 0.4 / len(recommended_jobs)

            score_components.append(min(0.4, reason_score))

            return sum(score_components)

        except Exception as e:
            print(f"응답 완성도 계산 실패: {e}")
            return 0.0

    def _calculate_structure_quality(self, generated_response: Dict[str, Any]) -> float:
        """구조 품질: JSON 응답 구조의 완성도"""

        try:
            score = 0.0

            # 1. 기본 필드 존재 (0.5)
            if 'content' in generated_response:
                score += 0.25
            if 'recommended_jobs' in generated_response:
                score += 0.25

            # 2. recommended_jobs 구조 품질 (0.5)
            recommended_jobs = generated_response.get('recommended_jobs', [])
            if recommended_jobs:
                valid_jobs = 0
                required_fields = ['rec_idx', 'title', 'url', 'recommendation_reason']

                for job in recommended_jobs:
                    if isinstance(job, dict):
                        field_count = sum(1 for field in required_fields if field in job and job[field])
                        if field_count >= 3:  # 최소 3개 필드는 있어야 함
                            valid_jobs += 1

                if recommended_jobs:
                    job_structure_score = valid_jobs / len(recommended_jobs)
                    score += 0.5 * job_structure_score

            return score

        except Exception as e:
            print(f"구조 품질 계산 실패: {e}")
            return 0.0

    def _is_mentioned_in_text(self, element: str, text: str) -> bool:
        """텍스트에서 특정 요소가 언급되었는지 확인"""

        if not element or not text:
            return False

        element_lower = element.lower()

        # 정확한 매치
        if element_lower in text:
            return True

        # 키워드 매치 (공백 제거 후)
        element_keywords = element_lower.replace(' ', '').replace('-', '')
        text_normalized = text.replace(' ', '').replace('-', '')

        if element_keywords in text_normalized:
            return True

        # 부분 매치 (길이가 3자 이상인 경우)
        if len(element_keywords) >= 3:
            words = element_keywords.split()
            for word in words:
                if len(word) >= 3 and word in text:
                    return True

        return False