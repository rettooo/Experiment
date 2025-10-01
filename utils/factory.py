"""
컴포넌트 팩토리: 설정에 따라 적절한 구현체 인스턴스 생성
"""

from typing import Type, Dict, Any

from core.interfaces import BaseEmbedder, BaseChunker, BaseRetriever, BaseLLM
from core.interfaces.response_generator import BaseResponseGenerator
from core.config import EmbedderConfig, ChunkerConfig, RetrieverConfig, LLMConfig

# 임베딩 모델 구현체들
from implementations.embedders import OpenAIEmbedder

# 청킹 전략 구현체들
from implementations.chunkers import NoChunker, RecursiveChunker

# 검색 시스템 구현체들
from implementations.retrievers import ChromaRetriever

# LLM 모델 구현체들
from implementations.llms import OpenAILLM

# 응답 생성기 구현체들
from implementations.response_generators.careerhy_generator import CareerHYResponseGenerator


class ComponentFactory:
    """컴포넌트 팩토리 클래스"""

    # 등록된 구현체들
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "openai": OpenAIEmbedder,
    }

    _chunkers: Dict[str, Type[BaseChunker]] = {
        "no_chunk": NoChunker,
        "recursive": RecursiveChunker,
    }

    _retrievers: Dict[str, Type[BaseRetriever]] = {
        "chroma": ChromaRetriever,
    }

    _llms: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
    }

    _response_generators: Dict[str, Type[BaseResponseGenerator]] = {
        "careerhy": CareerHYResponseGenerator,
    }

    @classmethod
    def create_embedder(cls, config: EmbedderConfig) -> BaseEmbedder:
        """임베딩 모델 인스턴스 생성"""
        if config.type not in cls._embedders:
            available = list(cls._embedders.keys())
            raise ValueError(f"지원하지 않는 임베딩 타입: {config.type}. 지원 타입: {available}")

        embedder_class = cls._embedders[config.type]

        # 설정을 인스턴스 생성 파라미터로 변환
        kwargs = {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            **config.params
        }

        return embedder_class(**kwargs)

    @classmethod
    def create_chunker(cls, config: ChunkerConfig) -> BaseChunker:
        """청킹 전략 인스턴스 생성"""
        if config.type not in cls._chunkers:
            available = list(cls._chunkers.keys())
            raise ValueError(f"지원하지 않는 청킹 타입: {config.type}. 지원 타입: {available}")

        chunker_class = cls._chunkers[config.type]

        # 설정을 인스턴스 생성 파라미터로 변환
        kwargs = {**config.params}

        if config.chunk_size is not None:
            kwargs["chunk_size"] = config.chunk_size
        if config.chunk_overlap is not None:
            kwargs["chunk_overlap"] = config.chunk_overlap

        return chunker_class(**kwargs)

    @classmethod
    def create_retriever(cls, config: RetrieverConfig) -> BaseRetriever:
        """검색 시스템 인스턴스 생성"""
        if config.type not in cls._retrievers:
            available = list(cls._retrievers.keys())
            raise ValueError(f"지원하지 않는 검색 타입: {config.type}. 지원 타입: {available}")

        retriever_class = cls._retrievers[config.type]

        # 설정을 인스턴스 생성 파라미터로 변환
        kwargs = {
            "collection_name": config.collection_name,
            "persist_directory": config.persist_directory,
            **config.params
        }

        return retriever_class(**kwargs)

    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseLLM:
        """LLM 모델 인스턴스 생성"""
        if config.type not in cls._llms:
            available = list(cls._llms.keys())
            raise ValueError(f"지원하지 않는 LLM 타입: {config.type}. 지원 타입: {available}")

        llm_class = cls._llms[config.type]

        # 설정을 인스턴스 생성 파라미터로 변환
        kwargs = {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            **config.params
        }

        return llm_class(**kwargs)

    @classmethod
    def create_response_generator(cls, config) -> BaseResponseGenerator:
        """응답 생성기 인스턴스 생성"""
        # Dict 형태와 Config 객체 둘 다 지원
        if hasattr(config, 'type'):
            # LLMConfig 객체인 경우
            generator_type = config.type
            model_name = config.model_name
            temperature = config.temperature
            max_tokens = config.max_tokens
            params = config.params
        else:
            # Dict인 경우 (기존 호환성)
            generator_type = config.get('type', 'careerhy')
            model_name = config.get('model_name', 'gpt-4o-mini')
            temperature = config.get('temperature', 0.7)
            max_tokens = config.get('max_tokens', 1000)
            params = config.get('params', {})

        if generator_type not in cls._response_generators:
            available = list(cls._response_generators.keys())
            raise ValueError(f"지원하지 않는 응답 생성기 타입: {generator_type}. 지원 타입: {available}")

        generator_class = cls._response_generators[generator_type]

        # 설정을 인스턴스 생성 파라미터로 변환
        kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **params
        }

        return generator_class(**kwargs)

    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """새로운 임베딩 모델 등록"""
        cls._embedders[name] = embedder_class

    @classmethod
    def register_chunker(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        """새로운 청킹 전략 등록"""
        cls._chunkers[name] = chunker_class

    @classmethod
    def register_retriever(cls, name: str, retriever_class: Type[BaseRetriever]) -> None:
        """새로운 검색 시스템 등록"""
        cls._retrievers[name] = retriever_class

    @classmethod
    def register_llm(cls, name: str, llm_class: Type[BaseLLM]) -> None:
        """새로운 LLM 모델 등록"""
        cls._llms[name] = llm_class

    @classmethod
    def register_response_generator(cls, name: str, generator_class: Type[BaseResponseGenerator]) -> None:
        """새로운 응답 생성기 등록"""
        cls._response_generators[name] = generator_class

    @classmethod
    def get_available_components(cls) -> Dict[str, list]:
        """사용 가능한 모든 컴포넌트 목록 반환"""
        return {
            "embedders": list(cls._embedders.keys()),
            "chunkers": list(cls._chunkers.keys()),
            "retrievers": list(cls._retrievers.keys()),
            "llms": list(cls._llms.keys()),
            "response_generators": list(cls._response_generators.keys())
        }