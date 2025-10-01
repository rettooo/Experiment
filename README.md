# Career-HY RAG 실험 파이프라인

Career-HY RAG 시스템의 다양한 파라미터를 체계적으로 실험하여 검색 성능을 최적화하기 위한 파이프라인입니다.

## 🎯 프로젝트 개요

### 목적
- **검색 성능 최적화**: 다양한 임베딩 모델, 청킹 전략, 검색 알고리즘 비교
- **체계적 실험**: YAML 기반 설정으로 재현 가능한 실험 환경 제공

### 주요 특징
- 🐳 **Docker 기반**: 일관된 실험 환경 보장
- 💾 **임베딩 캐싱**: 동일 설정 재실험 시 API 비용 절약
- 📊 **다양한 평가 지표**: Recall@k, Precision@k, MRR, MAP, nDCG@k
- 🔧 **모듈형 아키텍처**: 쉬운 확장성과 유지보수
- 📝 **YAML 설정**: 코드 수정 없이 실험 파라미터 조정

## 📁 디렉토리 구조

```
Experiment/
├── configs/                 # 실험 설정 파일들
│   ├── baseline.yaml       # 베이스라인 설정 (현재 서비스)
│   └── chunking_test.yaml  # 청킹 전략 실험 설정
├── core/                   # 핵심 파이프라인
│   ├── interfaces/        # 추상 인터페이스 (ABC)
│   ├── pipeline.py        # 메인 실험 파이프라인
│   └── config.py         # 설정 관리
├── implementations/       # 구현체들
│   ├── embedders/        # 임베딩 모델 (OpenAI 등)
│   ├── chunkers/         # 청킹 전략 (RecursiveCharacterTextSplitter 등)
│   ├── retrievers/       # 검색 시스템 (ChromaDB 등)
│   └── evaluators/       # 평가 지표 계산
├── utils/                # 유틸리티
│   ├── data_loader.py    # S3 데이터 로드
│   ├── embedding_cache.py # 임베딩 캐싱 시스템
│   ├── gt_converter.py   # Ground Truth CSV→JSONL 변환기
│   └── factory.py        # 컴포넌트 팩토리
├── data/                 # Ground Truth 데이터
│   ├── test_queries.jsonl      # 테스트 쿼리 (575개)
│   ├── test_queries_small.jsonl # 작은 테스트 셋 (3개)
│   └── ground_truth.jsonl      # Ground Truth 데이터
├── cache/                # 임베딩 캐시 저장소
├── results/              # 실험 결과
├── run_experiment.sh     # 메인 실험 실행 스크립트
├── docker-compose.yml    # Docker 구성
├── Dockerfile           # Docker 이미지 정의
└── requirements.txt     # Python 의존성
```

## 실험 방법

### 1. yaml 파일 설정

confgis 디렉토리에 실험하고자 하는 옵션을 지정하여 yaml 파일 작성 (baseline.yaml 및 아래 실허 설정 참고)

### 2. 실험 실행

```bash
# 베이스라인 실험 (현재 서비스와 동일한 설정)
./run_experiment.sh configs/baseline.yaml
```

### 3. 결과 확인

results 디렉토리에서 json 결과 파일 확인


## 📊 실험 설정

### YAML 설정 파일 구조

```yaml
# 실험 기본 정보
experiment_name: "baseline"
description: "현재 서비스와 동일한 베이스라인 설정"
output_dir: "results"

# 임베딩 설정
embedder:
  type: "openai"
  model_name: "text-embedding-ada-002"
  batch_size: 5

# 청킹 설정
chunker:
  type: "no_chunk"
  chunk_size: null
  chunk_overlap: null

# 검색 시스템 설정
retriever:
  type: "chroma"
  collection_name: "job-postings-baseline"
  persist_directory: "/tmp/chroma_baseline"
  top_k: 10

# LLM 설정
llm:
  type: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000

# 데이터 설정
data:
  s3_bucket: "career-hi"
  pdf_prefix: "initial-dataset/pdf/"
  json_prefix: "initial-dataset/json/"
  test_queries_path: "data/test_queries.jsonl"

# 평가 설정
evaluation:
  metrics: ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]
  k_values: [1, 3, 5, 10]
```

### 평가 지표

#### Recall@k (재현율)
- **정의**: 전체 관련 문서 중 상위 k개 검색 결과에 포함된 관련 문서의 비율
- **계산**: `Recall@k = (상위 k개 중 관련 문서 수) / (전체 관련 문서 수)`
- **의미**: 놓친 관련 문서가 얼마나 적은지 측정 (높을수록 좋음)
- **예시**: 관련 문서 10개 중 상위 5개에서 3개 발견 → Recall@5 = 0.3

#### Precision@k (정밀도)
- **정의**: 상위 k개 검색 결과 중 실제로 관련 있는 문서의 비율
- **계산**: `Precision@k = (상위 k개 중 관련 문서 수) / k`
- **의미**: 검색 결과의 정확성 측정 (높을수록 좋음)
- **예시**: 상위 5개 중 3개가 관련 있음 → Precision@5 = 0.6

#### MRR (Mean Reciprocal Rank)
- **정의**: 각 쿼리의 첫 번째 관련 문서 순위의 역수 평균
- **계산**: `MRR = (1/|Q|) × Σ(1/rank_i)` (rank_i = 첫 번째 관련 문서 순위)
- **의미**: 관련 문서가 얼마나 상위에 위치하는지 측정 (높을수록 좋음)
- **예시**: 첫 관련 문서가 3번째 → RR = 1/3 = 0.333

#### MAP (Mean Average Precision)
- **정의**: 각 쿼리의 Average Precision의 평균값
- **계산**: `MAP = (1/|Q|) × Σ(AP_i)` (AP = 관련 문서별 Precision 평균)
- **의미**: 모든 관련 문서 순위를 고려한 종합적 성능 (높을수록 좋음)
- **특징**: Precision과 Recall을 모두 반영한 균형 잡힌 지표

#### nDCG@k (Normalized Discounted Cumulative Gain)
- **정의**: 상위 k개 결과의 순위별 가중 점수를 이상적 순위와 비교한 정규화 점수
- **계산**: `nDCG@k = DCG@k / IDCG@k`
- **의미**: 순위가 높을수록 더 중요하다고 가정한 성능 측정 (높을수록 좋음)
- **특징**: 상위 순위에 있는 관련 문서에 더 높은 가중치 부여


## 🎭 LangSmith 정성평가 (LLM-as-Judge)

실험 파이프라인은 검색 성능 지표 외에도 생성된 응답의 품질을 평가하기 위한 LangSmith 기반 정성평가를 지원합니다.

### 정성평가 개요

#### 평가 방식: 이중 평가 시스템
1. **검색 평가**: 전체 575개 쿼리에 대한 검색 성능 측정
2. **생성 평가**: 15개 샘플 쿼리에 대한 응답 생성 및 정성평가

#### 샘플링 전략: Profile-based Sampling
- **목적**: 비용 효율적이면서도 대표성 있는 평가
- **방법**: MD5 해시 기반 고유 사용자 프로필 식별
- **결과**: 575개 쿼리 중 15개 고유 프로필 선택

### 4가지 정성평가 지표

#### 1. Recommendation Quality (추천 품질)
- **평가 대상**: 생성된 채용공고 추천의 전반적 품질
- **평가 기준**:
  - 관련성: 사용자 질문과 추천 공고의 연관성
  - 개인화: 사용자 프로필과 추천의 맞춤성
  - 구체성: 추천 이유의 상세함과 유용성
- **점수 범위**: 1-5점 (높을수록 좋음)

#### 2. Personalization Score (개인화 점수)
- **평가 대상**: 응답이 사용자에게 얼마나 개인화되어 있는가
- **평가 기준**: 프로필 요소(전공, 관심직무, 자격증 등)가 추천에 반영된 정도
- **점수 범위**: 1-5점 (높을수록 좋음)

#### 3. Response Helpfulness (응답 도움 정도)
- **평가 대상**: 취업 준비생에게 얼마나 실용적인 도움이 되는가
- **평가 기준**: 실용적이고 구체적인 조언 제공 여부
- **점수 범위**: 1-5점 (높을수록 좋음)

#### 4. Profile Alignment (프로필 일치도)
- **평가 대상**: 추천된 공고들이 사용자 배경과 얼마나 잘 맞는가
- **평가 기준**: 전공, 관심직무, 경험과 추천 공고의 적합성
- **점수 범위**: 1-5점 (높을수록 좋음)

### 평가 설정

#### YAML 설정에서 LangSmith 활성화
```yaml
# LangSmith 고품질 정성 평가
langsmith:
  enabled: true
  project_name: "career-hy-rag-evaluation"
  judge_model: "gpt-4o-mini"
  metrics:
    - "recommendation_quality"
    - "personalization_score"
    - "response_helpfulness"
    - "profile_alignment"
  max_concurrency: 3
  evaluation_timeout: 300

# 생성 품질 평가 (샘플링 설정)
evaluation:
  generation:
    target: "sample"
    sample_size: 15
    sample_strategy: "profile_based"
    sample_seed: null
```

### LangSmith 웹 인터페이스

평가 실행 중 LangSmith 웹사이트(https://smith.langchain.com)에서 실시간 추적 가능:
- 프로젝트: `career-hy-rag-evaluation`
- 개별 평가 실행 과정 및 결과 확인
- 평가 프롬프트와 응답 상세 분석
- 

## 임베딩 캐싱 시스템

### 캐시 동작 원리
1. **캐시 키 생성**: `{embedding_model}_{chunking_strategy}`
2. **첫 실행**: OpenAI API 호출 → 임베딩 생성 → 캐시 저장
3. **재실행**: 캐시 확인 → 기존 임베딩 로드 (API 호출 없음)

### 캐시 파일 구조
```
cache/embeddings/{cache_key}/
├── embeddings.npy          # NumPy 배열 (1536차원 벡터들)
├── processed_documents.pkl # 처리된 문서 텍스트
└── metadata.json          # 캐시 메타데이터
```

### 캐시 관리

```bash
# 캐시 목록 확인
ls cache/embeddings/

# 특정 캐시 삭제 (새로운 임베딩 생성하려면)
rm -rf cache/embeddings/text_embedding_ada_002_no_chunk

# 전체 캐시 삭제
rm -rf cache/embeddings/*
```

## 실험 결과 파일

### results_*.json - 종합 결과 요약
  - 전체 실험의 핵심 지표들을 요약한 메인 결과 파일
  - 검색 성능 지표 (recall@k, precision@k, MRR, MAP,
   NDCG@k)
  - LangSmith 정성평가 평균 점수 (4개 지표)
  - 실험 설정 정보 및 소요 시간

### generated_responses_*.json - 생성된 응답 모음
  - 샘플링된 15개 쿼리에 대한 실제 LLM 응답들
  - 각 응답의 추천 채용공고 목록 (1-10번 인덱스)
  - 개인화된 추천 이유 및 설명
  - 사용자 맞춤형 조언

### retrieval_detailed_*.jsonl - 검색 상세 결과
  - 575개 전체 쿼리의 검색 성능 상세 데이터 (JSONL
  형식)
  - 각 쿼리별 검색된 문서 목록, relevance 점수
  - 개별 쿼리의 recall, precision, MRR 등 세부 지표
  - 검색 실패 케이스 분석용 데이터

### generation_detailed_*.jsonl - 생성 상세 결과
  - 15개 샘플 쿼리의 생성 평가 상세 데이터 (JSONL
  형식)
  - LangSmith 정성평가 개별 점수 및 평가 이유
  - 각 지표별 상세 평가 결과
  (recommendation_quality, personalization_score 등)
  - 품질 개선을 위한 분석용 데이터

## 📊 데이터 소스

### S3 데이터
- **버킷**: `career-hi`
- **PDF 경로**: `initial-dataset/pdf/` (1,473개 파일)
- **JSON 경로**: `initial-dataset/json/` (1,473개 파일)

### Ground Truth
- **쿼리**: 575개 (GT 버전 3 기준)

## 🔄 Ground Truth 데이터 관리

### 새로운 GT CSV → JSONL 변환

새로운 Ground Truth CSV 파일을 받았을 때 실험에 사용할 JSONL 형태로 변환하는 유틸리티:

```bash
# 대화형 변환
python utils/gt_converter.py new_ground_truth.csv data/test_queries_new.jsonl
```


## 📚 참고 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [RAG 평가 지표 가이드](https://docs.ragas.io/en/stable/concepts/metrics/)

## 📝 라이센스

이 프로젝트는 Career-HY 팀의 내부 실험용으로 개발되었습니다.
