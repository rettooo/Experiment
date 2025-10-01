"""
실제 Career-HY 서비스와 동일한 프롬프트 구성

실제 서비스의 prompt_templates.py와 llm_prompting.py의 로직을
실험 환경에 맞게 복제하여 구현
"""

from typing import Dict, List, Any, Optional


# 실제 서비스와 동일한 프롬프트 템플릿들
_demo_profile = (
    "전공: 컴퓨터공학\n"
    "관심 직무: 백엔드, 클라우드\n"
    "자격증: SQLD\n"
    "수강 과목: 자료구조, 운영체제, 네트워크, 데이터베이스, 클라우드컴퓨팅"
)

CONSULTATION_EXAMPLES = (
    "[예시: 일반 상담]\n"
    f"<사용자 프로필>\n{_demo_profile}\n\n"
    "user: 이 프로필을 가진 학생이 면접 준비를 어떻게 하면 좋을까요?\n"
    "assistant:\n"
    "1) 전공 과목(운영체제, 네트워크)을 기반으로 시스템 이해도를 강조하세요.\n"
    "2) SQLD는 이미 취득하셨으니 정보처리기사 자격증을 준비하시는 것도 좋게 작용할수 있습니다.\n"
    "3) 기술 면접 대비로 자료구조·DB 질문 리스트를 만들어 답변을 연습하세요.\n"
    "4) 팀을 꾸려서 프로젝트를 진행하고 실제로 배포하고 운영경험을 쌓아보는 것도 좋습니다.\n"
)

RECOMMENDATION_EXAMPLES = (
    "[예시: 채용공고 추천 + 후속 질문]\n"
    f"<사용자 프로필>\n{_demo_profile}\n\n"
    "user: 클라우드 관련 인턴 채용공고 3개만 추천해줘\n"
    "assistant: 1) AWS 클라우드 인턴 (마감 4/10) ... 2) Azure 백엔드 인턴 (마감 4/18) ... 3) GCP DevOps 인턴 (마감 4/25) ...\n"
    "user: 방금 추천한 공고 중 기술 스택이 가장 다양한 곳을 알려줘\n"
    "assistant: 세 공고 중 가장 다양한 스택을 다루는 곳은 GCP DevOps 인턴입니다. 이유는 ...\n"
)

RECOMMENDATION_GUIDANCE = (
    "\n\n**중요 지침:**\n"
    "1. 사용자의 질문 의도를 정확히 파악하세요.\n"
    "2. 사용자의 프로필(전공·수강 과목·자격증·관심 직무)을 적극 활용하여 추천 이유를 작성하세요.\n"
)

BASE_PROMPT_STATIC = (
    "당신은 Career-HY의 AI 커리어 어시스턴트입니다.\n"
    "Career-HY는 한양대학교 학생들을 대상으로 하는 채용공고 추천 챗봇 서비스입니다.\n\n"
    "우리의 서비스에 대해 묻는 질문이 있다면, 학생들의 수강 이력, 자격증, 관심사 등의 프로필 정보를 활용해 개인 맞춤형 채용공고를 추천할수 있다는 점을 꼭 드러내야해\n"
    "사용자가 아직 프로필 정보를 등록하지 않은 상태라면, 프로필 정보를 등록할 것을 요구해줘 (이미 프로필 정보가 등록이 되어있다면, 굳이 언급할 필요는 없어)\n"
    "사용자의 프로필 정보, 이전 대화 기록을 바탕으로 사용자의 질문에 적절한 답변을 해주세요.\n\n"
)


class CareerHYPromptBuilder:
    """실제 Career-HY 서비스와 동일한 프롬프트 구성"""

    def __init__(self):
        self.base_prompt = BASE_PROMPT_STATIC
        self.recommendation_guidance = RECOMMENDATION_GUIDANCE
        self.recommendation_examples = RECOMMENDATION_EXAMPLES

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        실제 서비스와 동일한 프롬프트 구성

        Args:
            query: 사용자 질문
            retrieved_docs: 검색된 채용공고 문서들
            user_profile: 사용자 프로필
            chat_history: 대화 이력

        Returns:
            str: 완성된 프롬프트
        """

        prompt_parts = []

        # 1. 기본 시스템 프롬프트
        prompt_parts.append(self.base_prompt)

        # 2. 사용자 프로필 포맷팅 (실제 서비스와 동일)
        profile_text = self._format_user_profile(user_profile)
        prompt_parts.append(f"<사용자 프로필>\n{profile_text}\n")

        # 3. 추천 가이드라인 및 예시 (채용공고 추천 시나리오 가정)
        prompt_parts.append(self.recommendation_guidance)
        prompt_parts.append(self.recommendation_examples)

        # 4. 검색된 문서들 (핵심!)
        if retrieved_docs:
            docs_text = self._format_retrieved_documents(retrieved_docs)
            prompt_parts.append(f"<검색된 채용공고들>\n{docs_text}\n")

        # 5. 대화 이력 (있다면)
        if chat_history:
            history_text = self._format_chat_history(chat_history)
            prompt_parts.append(f"<대화 이력>\n{history_text}\n")

        # 6. 응답 형식 지정
        response_format = self._get_response_format()
        prompt_parts.append(response_format)

        # 7. 현재 질문
        prompt_parts.append(f"user: {query}\nassistant:")

        return "\n\n".join(prompt_parts)

    def build_recommendation_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        실제 서비스와 동일한 추천용 프롬프트 구성
        (LangChain with_structured_output용 - JSON 형식 요구 없음)
        """

        # 1. Base prompt
        prompt_parts = [BASE_PROMPT_STATIC]

        # 2. 사용자 프로필
        profile_text = self._format_user_profile(user_profile)
        prompt_parts.append(f"<사용자 프로필>\n{profile_text}\n")

        # 3. 대화 이력 (있다면)
        if chat_history:
            history_text = self._format_chat_history(chat_history)
            prompt_parts.append(f"<이전 대화 내용>\n{history_text}\n")

        # 4. 검색된 채용공고들 (실제 서비스와 동일한 형식)
        doc_texts = [
            f"채용공고 {i + 1}:\n제목: {doc.get('metadata', {}).get('title', doc.get('metadata', {}).get('post_title', 'N/A'))}\nURL: {doc.get('metadata', {}).get('url', 'N/A')}\n내용: {doc.get('text', '')[:300]}..."
            for i, doc in enumerate(retrieved_docs[:10])  # 최대 10개
        ]
        formatted_docs = "\n\n".join(doc_texts)

        prompt_parts.append(f"다음은 사용자의 프로필과 관심사에 맞춰 검색된 {len(retrieved_docs[:10])}개 채용공고입니다:\n{formatted_docs}")

        # 5. 추천 가이드라인
        prompt_parts.append(RECOMMENDATION_GUIDANCE)

        # 6. 현재 질문
        prompt_parts.append(f"사용자 질문: {query}")

        return "\n\n".join(prompt_parts)

    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """사용자 프로필을 실제 서비스와 동일한 형태로 포맷팅"""

        formatted_parts = []

        # 전공
        if profile.get('major'):
            formatted_parts.append(f"전공: {profile['major']}")

        # 관심 직무
        if profile.get('interest_job'):
            interest_jobs = profile['interest_job']
            if isinstance(interest_jobs, list):
                formatted_parts.append(f"관심 직무: {', '.join(interest_jobs)}")
            else:
                formatted_parts.append(f"관심 직무: {interest_jobs}")

        # 자격증
        if profile.get('certification'):
            certifications = profile['certification']
            if isinstance(certifications, list):
                formatted_parts.append(f"자격증: {', '.join(certifications)}")
            else:
                formatted_parts.append(f"자격증: {certifications}")

        # 수강 과목 (catalogs에서 추출)
        if profile.get('catalogs'):
            course_names = []
            for catalog in profile['catalogs']:
                if isinstance(catalog, dict) and 'course_name' in catalog:
                    course_names.append(catalog['course_name'])
                elif isinstance(catalog, str):
                    course_names.append(catalog)

            if course_names:
                formatted_parts.append(f"수강 과목: {', '.join(course_names)}")

        # 동아리/대외활동
        if profile.get('club_activities'):
            activities = profile['club_activities']
            if isinstance(activities, list):
                formatted_parts.append(f"동아리/대외활동: {', '.join(activities)}")
            else:
                formatted_parts.append(f"동아리/대외활동: {activities}")

        return "\n".join(formatted_parts)

    def _format_retrieved_documents(self, docs: List[Dict[str, Any]]) -> str:
        """검색된 문서들을 프롬프트에 포함시키는 실제 서비스 방식"""

        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # 문서 메타데이터 추출
            metadata = doc.get('metadata', {})

            doc_text = f"""[채용공고 {i}]
- 공고 ID: {metadata.get('rec_idx', 'N/A')}
- 제목: {metadata.get('title', metadata.get('post_title', 'N/A'))}
- 회사: {metadata.get('company', metadata.get('company_name', 'N/A'))}
- 마감일: {metadata.get('deadline', 'N/A')}
- URL: {metadata.get('url', 'N/A')}

채용 내용:
{doc.get('text', '')[:1000]}..."""

            formatted_docs.append(doc_text.strip())

        return "\n\n".join(formatted_docs)

    def _format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """대화 이력 포맷팅"""

        formatted_history = []
        for message in chat_history[-3:]:  # 최근 3개만
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_history.append(f"{role}: {content}")

        return "\n".join(formatted_history)

    def _get_response_format(self) -> str:
        """응답 형식 지정 (JSON 구조화된 출력 요구)"""

        return """응답은 반드시 다음 JSON 형식으로 제공해주세요:

{
  "content": "사용자에게 제공할 조언 및 설명 텍스트",
  "recommended_jobs": [
    {
      "rec_idx": "채용공고 ID",
      "title": "채용공고 제목",
      "url": "채용공고 URL",
      "deadline": "마감일 (있다면)",
      "recommendation_reason": "이 공고를 추천하는 구체적인 이유 (사용자 프로필과 연결)"
    }
  ]
}

중요: 추천하는 채용공고는 반드시 위에서 제공된 <검색된 채용공고들> 중에서만 선택하세요."""