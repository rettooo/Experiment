import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class EmbedderConfig:
    """임베딩 모델 설정"""
    type: str  # "openai", "huggingface", etc.
    model_name: str
    batch_size: int = 5
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ChunkerConfig:
    """청킹 전략 설정"""
    type: str  # "no_chunk", "recursive", "semantic", etc.
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class RetrieverConfig:
    """검색 시스템 설정"""
    type: str  # "chroma", "faiss", "elasticsearch", etc.
    collection_name: str = "job-postings"
    persist_directory: str = "/tmp/chroma_experiment"
    top_k: int = 10
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class LLMConfig:
    """LLM 모델 설정"""
    type: str  # "openai", "anthropic", "local", etc.
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class DataConfig:
    """데이터 설정"""
    s3_bucket: str = None
    pdf_prefix: str = "initial-dataset/pdf/"
    json_prefix: str = "initial-dataset/json/"
    ground_truth_path: str = "data/ground_truth.jsonl"
    test_queries_path: str = "data/test_queries.jsonl"

    def __post_init__(self):
        # S3 버킷이 설정되지 않은 경우 환경변수에서 가져오기
        if self.s3_bucket is None:
            import os
            self.s3_bucket = os.getenv('S3_BUCKET_NAME', 'career-hi')


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: list = None  # ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]
    k_values: list = None  # [1, 3, 5, 10]
    use_langsmith: bool = True
    langsmith_project: str = "Career-HY-Experiment"

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]


@dataclass
class ExperimentConfig:
    """전체 실험 설정"""
    experiment_name: str
    description: str = ""
    embedder: EmbedderConfig = None
    chunker: ChunkerConfig = None
    retriever: RetrieverConfig = None
    llm: LLMConfig = None
    data: DataConfig = None
    evaluation: EvaluationConfig = None
    output_dir: str = "results"

    def __post_init__(self):
        # 기본값 설정
        if self.embedder is None:
            self.embedder = EmbedderConfig(type="openai", model_name="text-embedding-ada-002")
        if self.chunker is None:
            self.chunker = ChunkerConfig(type="no_chunk")
        if self.retriever is None:
            self.retriever = RetrieverConfig(type="chroma")
        if self.llm is None:
            self.llm = LLMConfig(type="openai", model_name="gpt-4o-mini")
        if self.data is None:
            self.data = DataConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 중첩된 딕셔너리를 dataclass로 변환
        config_data = {}

        config_data['experiment_name'] = data['experiment_name']
        config_data['description'] = data.get('description', '')
        config_data['output_dir'] = data.get('output_dir', 'results')

        # 각 컴포넌트 설정 변환
        if 'embedder' in data:
            config_data['embedder'] = EmbedderConfig(**data['embedder'])

        if 'chunker' in data:
            config_data['chunker'] = ChunkerConfig(**data['chunker'])

        if 'retriever' in data:
            config_data['retriever'] = RetrieverConfig(**data['retriever'])

        if 'llm' in data:
            config_data['llm'] = LLMConfig(**data['llm'])

        if 'data' in data:
            config_data['data'] = DataConfig(**data['data'])

        if 'evaluation' in data:
            config_data['evaluation'] = EvaluationConfig(**data['evaluation'])

        return cls(**config_data)

    def to_yaml(self, yaml_path: str) -> None:
        """설정을 YAML 파일로 저장"""
        config_dict = asdict(self)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def get_experiment_id(self) -> str:
        """실험 고유 ID 생성"""
        components = [
            f"emb_{self.embedder.model_name.replace('-', '_')}",
            f"chunk_{self.chunker.type}",
            f"retr_{self.retriever.type}",
            f"llm_{self.llm.model_name.replace('-', '_')}"
        ]
        return "_".join(components)

    def get_output_path(self, filename: str = None) -> Path:
        """출력 경로 생성"""
        base_path = Path(self.output_dir) / self.experiment_name
        base_path.mkdir(parents=True, exist_ok=True)

        if filename:
            return base_path / filename
        return base_path