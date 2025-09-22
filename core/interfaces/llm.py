from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """채팅 메시지 데이터 클래스"""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """LLM 응답 데이터 클래스"""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """LLM 모델의 추상 인터페이스"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        채팅 메시지에 대한 응답 생성

        Args:
            messages: 채팅 메시지 리스트
            max_tokens: 최대 토큰 수
            temperature: 온도 파라미터
            **kwargs: 모델별 추가 파라미터

        Returns:
            LLM 응답
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[ChatMessage],
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        구조화된 출력 생성 (예: JSON 스키마 기반)

        Args:
            messages: 채팅 메시지 리스트
            schema: 출력 스키마
            **kwargs: 모델별 추가 파라미터

        Returns:
            구조화된 응답
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "config": self.config
        }