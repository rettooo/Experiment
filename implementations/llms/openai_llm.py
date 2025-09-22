import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

from core.interfaces.llm import BaseLLM, ChatMessage, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI LLM 모델 구현체 (현재 서비스에서 사용 중)"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """채팅 메시지에 대한 응답 생성"""

        # 파라미터 설정 (인자로 받은 값이 우선, 없으면 인스턴스 기본값 사용)
        final_max_tokens = max_tokens or self.max_tokens
        final_temperature = temperature if temperature is not None else self.temperature

        # ChatMessage를 OpenAI 형식으로 변환
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=final_max_tokens,
                temperature=final_temperature,
                timeout=self.timeout,
                **kwargs
            )

            content = response.choices[0].message.content
            metadata = {
                "model": self.model_name,
                "usage": response.usage.dict() if response.usage else {},
                "finish_reason": response.choices[0].finish_reason
            }

            return LLMResponse(content=content, metadata=metadata)

        except Exception as e:
            print(f"OpenAI LLM API 호출 실패: {e}")
            raise

    def generate_structured(
        self,
        messages: List[ChatMessage],
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        구조화된 출력 생성 (JSON 스키마 기반)

        Note: 현재 서비스에서는 LangChain의 with_structured_output을 사용하지만,
        실험에서는 간단한 JSON 모드를 사용
        """
        # 시스템 메시지에 JSON 형식 요청 추가
        system_message = ChatMessage(
            role="system",
            content=f"다음 JSON 스키마에 맞춰 응답해주세요: {schema}"
        )

        all_messages = [system_message] + messages

        try:
            response = self.generate(
                all_messages,
                response_format={"type": "json_object"},
                **kwargs
            )

            # JSON 파싱 시도
            import json
            return json.loads(response.content)

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            print(f"응답 내용: {response.content}")
            raise
        except Exception as e:
            print(f"구조화된 출력 생성 실패: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "config": self.config
        }