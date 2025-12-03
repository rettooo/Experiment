# Career-HY RAG 실험 파이프라인

Career-HY RAG 시스템의 다양한 파라미터를 체계적으로 실험하여 검색 성능을 최적화하기 위한 파이프라인입니다.

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [주요 기능](#2-주요-기능)
3. [설치 및 설정](#3-설치-및-설정)
4. [사용 방법](#4-사용-방법)
5. [아키텍처](#5-아키텍처)
6. [실제 서비스 통합 가이드](#6-실제-서비스-통합-가이드)
7. [실험 결과 및 분석](#55-실험-결과-및-분석)
8. [참고 자료](#7-참고-자료)

---

## 1. 프로젝트 개요

### 목적
- **검색 성능 최적화**: 다양한 임베딩 모델, 청킹 전략, 검색 알고리즘 비교
- **체계적 실험**: YAML 기반 설정으로 재현 가능한 실험 환경 제공
- **GT 데이터셋 생성**: 클러스터링 기반 Ground Truth 데이터셋 자동 생성
- **실제 서비스 적용**: StructuredDocumentLoader 및 메타데이터 확장 기능 제공

### 주요 특징
- 🐳 **Docker 기반**: 일관된 실험 환경 보장
- 💾 **임베딩 캐싱**: 동일 설정 재실험 시 API 비용 절약
- 📊 **다양한 평가 지표**: Recall@k, Precision@k, MRR, MAP, nDCG@k
- 🔧 **모듈형 아키텍처**: 쉬운 확장성과 유지보수
- 📝 **YAML 설정**: 코드 수정 없이 실험 파라미터 조정
- 🎯 **섹션별 청킹**: 채용공고의 구조화된 정보(우대사항, 자격요건, 주요업무) 활용
- 📈 **GT 자동 생성**: 클러스터링 기반 Ground Truth 데이터셋 생성
- 🔄 **통합 파이프라인**: GT 생성 관련 기능을 하나의 파이프라인으로 통합
- ⚡ **동적 평가**: 정답 개수에 따른 자동 top_k 조정
- 📅 **메타데이터 확장**: deadline, start_date, crawling_time 지원

---

## 2. 주요 기능

### 2.1 RAG 실험 파이프라인

체계적이고 재현 가능한 RAG 실험을 위한 종합 파이프라인입니다.

#### 핵심 기능
- **다양한 임베딩 모델 지원**: OpenAI, Snowflake 등
- **다양한 청킹 전략**: Recursive, Fixed, No Chunk
- **벡터 검색 시스템**: ChromaDB, FAISS
- **종합 평가 시스템**: 검색 성능 + LangSmith 정성평가

#### 평가 지표
- **Recall@k**: 전체 관련 문서 중 상위 k개 검색 결과에 포함된 관련 문서의 비율
- **Precision@k**: 상위 k개 검색 결과 중 실제로 관련 있는 문서의 비율
- **MRR@k**: 각 쿼리의 첫 번째 관련 문서 순위의 역수 평균 (상위 k개 내)
- **MAP**: 모든 관련 문서 순위를 고려한 종합적 성능
- **nDCG@k**: 순위가 높을수록 더 중요하다고 가정한 성능 측정
- **R-recall**: Recall@(정답개수) - 정답 문서 개수만큼의 recall 계산
- **Hit@k_count**: 상위 k개 검색 결과 중 정답 문서의 개수

#### 평가 시스템 개선 사항
- **동적 top_k 지원**: R-recall 계산을 위해 정답 개수에 따라 top_k 자동 조정
  - `evaluation_top_k = min(max(base_top_k, gt_count), 60)`
  - 정답이 많은 쿼리에서도 정확한 평가 가능

#### LangSmith 정성평가
- **Recommendation Quality**: 추천 품질 전반
- **Personalization Score**: 개인화 수준
- **Response Helpfulness**: 도움 정도
- **Profile Alignment**: 프로필 일치도

### 2.2 StructuredDocumentLoader

채용공고의 구조화된 정보를 활용한 섹션별 청킹 시스템입니다.

#### 주요 특징
- **섹션별 청킹**: 우대사항(preferred), 자격요건(qualifications), 주요업무(job_duties) 별도 처리
- **JobPostParser**: `unstructured` 라이브러리 기반 PDF 파싱
- **메타데이터 보존**: 원본 JSON 메타데이터 전체 보존 (deadline, start_date, crawling_time 포함)
- **Context 주입**: 각 청크에 `[회사: ...] [직무: ...]` 자동 추가

#### 통계
- 총 청크: 2,039개 (1,473개 문서)
- 섹션별 분포:
  - `preferred`: 1,036개 (50.8%)
  - `qualifications`: 398개 (19.5%)
  - `job_duties`: 279개 (13.7%)
  - `full_text` (fallback): 326개 (16.0%)

#### 사용 예시
```python
from implementations.loaders.structured_loader import StructuredDocumentLoader

loader = StructuredDocumentLoader(
    strategy="fast",  # 또는 "hi_res"
    target_sections=["preferred", "qualifications", "job_duties"],
    include_context=True
)
chunks = loader.load_from_documents(documents)
```

#### 2.2.1 메타데이터 확장

채용공고의 시간 정보를 포함한 확장된 메타데이터를 지원합니다.

##### 지원 메타데이터 필드
- **deadline**: 채용공고 마감일
- **start_date**: 채용공고 시작일
- **crawling_time**: 데이터 크롤링 시각
- **기본 필드**: rec_idx, title, company, url, tags 등

##### 메타데이터 보존
- **StructuredDocumentLoader**: 모든 청크에 원본 메타데이터 전체 보존
- **Fallback 청크**: 경량/일반 fallback 모두 메타데이터 보존
- **벡터 DB 저장**: ChromaDB에 메타데이터 전체 저장
- **응답 생성**: Response Generator에서 메타데이터 활용

##### 사용 예시
```python
# 청크 메타데이터 확인
chunk = chunks[0]
print(chunk.metadata.get("deadline"))      # 마감일
print(chunk.metadata.get("start_date"))   # 시작일
print(chunk.metadata.get("crawling_time")) # 크롤링 시각
```

##### 프롬프트 활용
- 프롬프트에 마감일 정보 자동 포함
- 응답 생성 시 시간 정보 활용 가능

### 2.3 GT 생성 파이프라인

클러스터링 기반 Ground Truth 데이터셋 자동 생성 시스템입니다.

#### 파이프라인 단계
1. **Phase 1**: 중분류 기반 초기 클러스터 생성
2. **Phase 2**: 유사 중분류 병합 (규칙 기반)
3. **Phase 3**: 대표 문서 선택 (쿼리-문서 유사도 기반)
4. **Phase 4**: 통계 및 결과 저장

#### 입력 데이터
- `clustering_results_tag_based/*_classification.json`: 대분류별 분류 결과
- `similarity_rules_template.json`: 유사 중분류 병합 규칙

#### 출력 데이터
- `gt_generation_results/gt_clusters.json`: 클러스터 정보 및 대표 문서
- `gt_generation_results/gt_clusters_summary.csv`: 클러스터 요약 정보
- `gt_generation_results/gt_generation_statistics.txt`: 통계 정보

#### 실행 방법
```bash
# 전체 파이프라인 실행 (기본)
python gt_generation_pipeline.py

# CSV → JSONL 변환
python gt_generation_pipeline.py --convert-csv data/gt.csv data/output.jsonl

# 평가용 데이터 생성
python gt_generation_pipeline.py --create-eval data/gt_analysis.csv data/eval.jsonl

# 규칙 검증만 실행
python gt_generation_pipeline.py --validate-rules
```

#### 통합된 기능
- **유사도 규칙 검증**: 규칙 파일의 유효성 검사 및 커버리지 분석
- **CSV → JSONL 변환**: GT CSV를 평가 파이프라인 형식으로 변환
- **평가용 데이터 생성**: GT Analysis CSV를 평가용 JSONL로 변환
- **대표 문서 선택**: SentenceTransformer 기반 쿼리-문서 유사도 계산

### 2.4 클러스터링 파이프라인

태그 + 제목 기반으로 채용공고를 직무별로 클러스터링하는 시스템입니다.

#### 주요 기능
- **태그 기반 분류**: 대분류/중분류 자동 할당
- **다중 카테고리 지원**: 하나의 문서가 여러 대분류에 속할 수 있음
- **UMAP + HDBSCAN**: 차원 축소 및 클러스터링

#### 실행 방법
```bash
python job_clustering_pipeline.py
```

#### 출력 결과
- `clustering_results_tag_based/`: 대분류별 분류 결과
- `clustering_results/`: 클러스터링 결과 및 시각화

---

## 3. 설치 및 설정

### 3.1 필수 요구사항
- Python 3.8 이상
- Docker 및 Docker Compose
- AWS 자격 증명 (S3 접근용)
- OpenAI API Key (임베딩 및 LLM용)

### 3.2 설치

#### 1. 저장소 클론
```bash
git clone <repository-url>
cd Experiment
```

#### 2. 환경 변수 설정
`.env` 파일 생성:
```bash
# OpenAI API
OPENAI_API_KEY=your_api_key

# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-northeast-2

# LangSmith (선택)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

#### 3. Docker 빌드 및 실행
```bash
docker-compose build
docker-compose up -d
```

#### 4. 의존성 설치 (로컬 실행 시)
```bash
pip install -r requirements.txt
# 클러스터링/GT 생성용 추가 의존성
pip install -r requirements_clustering.txt
```

### 3.3 디렉토리 구조

```
Experiment/
├── configs/                 # 실험 설정 파일들
│   ├── baseline_search.yaml
│   ├── new_eval_baseline.yaml
│   └── new_eval_baseline_recursive.yaml
├── core/                   # 핵심 파이프라인
│   ├── interfaces/        # 추상 인터페이스 (ABC)
│   ├── pipeline.py        # 메인 실험 파이프라인
│   └── config.py         # 설정 관리
├── implementations/       # 구현체들
│   ├── embedders/        # 임베딩 모델
│   ├── chunkers/         # 청킹 전략
│   ├── retrievers/       # 검색 시스템
│   ├── evaluators/       # 평가 지표 계산
│   ├── loaders/          # StructuredDocumentLoader
│   └── parsers/          # JobPostParser
├── utils/                # 유틸리티
│   ├── data_loader.py    # S3 데이터 로드
│   ├── embedding_cache.py # 임베딩 캐싱 시스템
│   └── factory.py        # 컴포넌트 팩토리
├── data/                 # Ground Truth 데이터
│   └── gt_eval_fullquery_cluster_ids.jsonl
├── cache/                # 임베딩 캐시 저장소
├── results/              # 실험 결과
├── run_experiment.sh     # 메인 실험 실행 스크립트
├── docker-compose.yml    # Docker 구성
├── Dockerfile           # Docker 이미지 정의
└── requirements.txt     # Python 의존성
```

---

## 4. 사용 방법

### 4.1 실험 실행

#### 1. YAML 설정 파일 작성
`configs/` 디렉토리에 실험 설정 파일 작성:

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
  test_queries_path: "data/gt_eval_fullquery_cluster_ids.jsonl"
  use_structured_loader: false  # StructuredDocumentLoader 사용 여부

# 평가 설정
evaluation:
  mode: "retrieval_only"  # 또는 "dual"
  metrics: ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]
  k_values: [1, 3, 5, 10]
```

#### 2. 실험 실행
```bash
# Docker 환경
./run_experiment.sh configs/baseline_search.yaml

# 로컬 환경
python run_experiment.py configs/baseline_search.yaml
```

#### 3. 결과 확인
- `results/`: 실험 결과 JSON 파일
- LangSmith 웹 인터페이스: https://smith.langchain.com

### 4.1.1 설정 파일 설명

프로젝트에는 다양한 실험 시나리오를 위한 설정 파일이 포함되어 있습니다.

#### baseline_search.yaml
- **용도**: 기본 검색 성능 평가 실험
- **특징**:
  - Recursive Chunking 방식 사용 (chunk_size: 700, chunk_overlap: 100)
  - StructuredDocumentLoader 미사용 (전체 텍스트 파싱)
  - 데이터 버전: v3
- **사용 시나리오**: 기본 검색 성능 벤치마크 측정

#### new_eval_baseline.yaml
- **용도**: StructuredDocumentLoader 기반 섹션별 청킹 실험
- **특징**:
  - StructuredDocumentLoader 사용 (섹션별 청킹)
  - 청킹 방식: no_chunk (섹션 단위로만 분할)
  - 타겟 섹션: preferred, qualifications, job_duties
  - 데이터 버전: v4
- **사용 시나리오**: 섹션별 구조화된 청킹의 효과 측정

#### new_eval_baseline_recursive.yaml
- **용도**: StructuredDocumentLoader + Recursive Chunking 하이브리드 실험
- **특징**:
  - StructuredDocumentLoader 사용 (섹션별 청킹)
  - Recursive Chunking 추가 적용 (chunk_size: 500, chunk_overlap: 75)
  - 타겟 섹션: preferred, qualifications, job_duties
  - 데이터 버전: v4
- **사용 시나리오**: 하이브리드 청킹 전략 성능 평가

### 4.2 GT 데이터셋 생성

#### 1. 클러스터링 실행
```bash
python job_clustering_pipeline.py
```

#### 2. GT 생성 파이프라인 실행
```bash
python gt_generation_pipeline.py
```

#### 3. 결과 확인
- `gt_generation_results/gt_clusters.json`: 클러스터 정보
- `gt_generation_results/gt_clusters_summary.csv`: 요약 정보
- `gt_generation_results/gt_generation_statistics.txt`: 통계

### 4.3 클러스터링 실행

#### 기본 실행
```bash
python job_clustering_pipeline.py
```

#### 예상 소요 시간
- 데이터 로드: ~5초
- 임베딩 생성: ~2-3분 (첫 실행 시 모델 다운로드: +30초)
- UMAP 차원 축소: ~30초
- HDBSCAN 클러스터링: ~10초
- **총 예상 시간: 약 3-4분**

#### 출력 결과
- `clustering_results_tag_based/`: 대분류별 분류 결과
- `clustering_results/`: 클러스터링 결과 및 시각화

---

## 5. 아키텍처

### 5.1 모듈 구조

#### Core 모듈
- `core/pipeline.py`: 메인 실험 파이프라인
- `core/config.py`: 설정 관리
- `core/interfaces/`: 추상 인터페이스 정의

#### 구현 모듈
- `implementations/embedders/`: 임베딩 모델 구현
- `implementations/chunkers/`: 청킹 전략 구현
- `implementations/retrievers/`: 검색 시스템 구현
- `implementations/evaluators/`: 평가 지표 계산
- `implementations/loaders/`: StructuredDocumentLoader
- `implementations/parsers/`: JobPostParser

#### 유틸리티 모듈
- `utils/data_loader.py`: S3 데이터 로드
- `utils/embedding_cache.py`: 임베딩 캐싱
- `utils/factory.py`: 컴포넌트 팩토리

### 5.2 데이터 흐름

```
S3 데이터 로드
    ↓
StructuredDocumentLoader (섹션별 청킹)
    ↓
임베딩 생성 (캐싱)
    ↓
벡터 DB 저장 (ChromaDB/FAISS)
    ↓
검색 및 평가
    ↓
결과 저장
```

### 5.3 평가 시스템

#### Retrieval 평가
- 전체 쿼리에 대한 검색 성능 측정
- **지원 지표**: Recall@k, Precision@k, MRR@k, MAP, nDCG@k, R-recall, Hit@k_count
- **동적 top_k**: 정답 개수에 따라 검색 범위 자동 조정 (최대 60개)
- **평가 순서**: 사용자 요청 순서대로 지표 계산 및 출력

#### Generation 평가 (LangSmith)
- 샘플 쿼리에 대한 응답 생성 및 정성평가
- Profile-based Sampling (15개 고유 프로필)
- **4가지 정성 평가 지표**:
  - Recommendation Quality: 추천 품질 전반
  - Personalization Score: 개인화 수준
  - Response Helpfulness: 도움 정도
  - Profile Alignment: 프로필 일치도

---

## 5.5 실험 결과 및 분석

### 5.5.1 문제점 및 개선 방안

#### 기존 문제점
1. **GT 신뢰성 부족**: 기존 GT는 정답 데이터가 5개로 제한되어 신뢰성이 낮음
2. **노이즈 문제**: 단순 텍스트 청킹으로 무의미한 텍스트 포함 → 노이즈가 많음

#### 개선 방안
1. **태그 기반 클러스터링**: 텍스트에서 tag 정보를 이용해서 정답 군집 생성
2. **StructuredDocumentLoader 구현**: JobPostParser를 활용해 섹션별 청킹
   - 섹션 타입별로 청크 분리: `preferred`, `qualifications`, `job_duties`
   - 각 chunk에 섹션 타입 메타데이터 포함

### 5.5.2 실험 설정

#### 1) Baseline: Recursive Chunking
- **설정 파일**: `configs/baseline_search.yaml`
- **Chunker**: `recursive` (chunk_size: 700, chunk_overlap: 100)
- **StructuredDocumentLoader**: 사용 안함
- **목적**: 기본 검색 성능 벤치마크 측정
- **데이터 버전**: v3
- **처리된 문서 수**: 4,121개

#### 2) StructuredDocumentLoader (섹션별 청킹)
- **설정 파일**: `configs/new_eval_baseline.yaml`
- **Chunker**: `no_chunk` (StructuredDocumentLoader가 수행)
- **StructuredDocumentLoader**: 사용 (`use_structured_loader: true`)
- **Target Section**: `preferred`, `qualifications`, `job_duties`
- **목적**: 섹션별 청킹 효과 측정
- **데이터 버전**: v4
- **처리된 문서 수**: 2,503개

#### 3) StructuredDocumentLoader + Recursive Chunking
- **설정 파일**: `configs/new_eval_baseline_recursive.yaml`
- **Chunker**: `recursive` (chunk_size: 500, chunk_overlap: 75)
- **StructuredDocumentLoader**: 사용 (`use_structured_loader: true`)
- **Target Section**: `preferred`, `qualifications`, `job_duties`
- **목적**: 섹션별 청킹 + Recursive chunking 하이브리드 효과 측정
- **데이터 버전**: v4
- **처리된 문서 수**: 3,124개

### 5.5.3 검색 성능 비교

| Metric | Baseline | 섹션 청킹 | 섹션+Recursive | 최우수 |
|--------|----------|-----------|----------------|--------|
| **ndcg@10** | 0.2205 | **0.2591** | 0.2547 | ✅ 섹션 청킹 |
| **mrr@10** | 0.4227 | **0.4606** | 0.4468 | ✅ 섹션 청킹 |
| **precision@3** | **0.2479** | 0.2308 | 0.2350 | ✅ Baseline |
| **precision@5** | 0.2128 | **0.2205** | 0.2077 | ✅ 섹션 청킹 |
| **precision@10** | 0.1628 | **0.1679** | 0.1667 | ✅ 섹션 청킹 |
| **precision@20** | 0.1314 | **0.1353** | 0.1321 | ✅ 섹션 청킹 |
| **recall@10** | 0.1061 | 0.1086 | **0.1090** | ✅ 섹션+Recursive |
| **recall@20** | 0.1547 | **0.1609** | 0.1591 | ✅ 섹션 청킹 |
| **r_recall** | 0.1204 | **0.1345** | 0.1297 | ✅ 섹션 청킹 |
| **hit@10_count** | 1.628 | **1.680** | 1.667 | ✅ 섹션 청킹 |
| **hit@20_count** | 2.628 | **2.705** | 2.641 | ✅ 섹션 청킹 |

**결론**: 섹션별 청킹이 대부분의 검색 성능 지표에서 최우수 성능을 보임

### 5.5.4 생성 품질 비교 (LangSmith 평가)

| Metric | Baseline | 섹션 청킹 | 섹션+Recursive | 최우수 |
|--------|----------|-----------|----------------|--------|
| **recommendation_quality** | 4.2 | 4.2 | **4.3** | ✅ 섹션+Recursive |
| **personalization_score** | **4.5** | **4.5** | 4.4 | ✅ Baseline/섹션 청킹 |
| **response_helpfulness** | **4.4** | 4.2 | 4.3 | ✅ Baseline |
| **profile_alignment** | 4.0 | 4.0 | **4.1** | ✅ 섹션+Recursive |

**결론**: 생성 품질은 세 방법 모두 유사하나, 섹션+Recursive가 약간 우수 (차이는 매우 작음)

### 5.5.5 GT 데이터셋 통계

#### GT 버전 설명

**기존 GT (v1)**:
RAG 정량적 평가를 위해 Agent로 생성한 GT입니다. GT 하나당 초기 수집해놓은 채용공고 데이터셋에서 채용 공고를 랜덤으로 선택한 후 유사도가 높은 4개의 채용 공고를 추가하여 총 5개의 채용공고를 선별합니다. Agent는 5개의 채용 공고를 확인한 후 해당 채용 공고에 지원했을 법한 학생 프로필 데이터를 생성합니다. 이때 한양대학교 수강편람 데이터를 검색할 수 있는 API를 tool로 이용합니다.

**새로운 GT (v2)**:
태그 기반 클러스터링을 통해 생성된 GT입니다. 같은 클러스터에 속한 모든 채용공고를 정답으로 사용하여 가변 크기의 GT를 생성합니다. 평균 19.8개의 정답 문서를 포함하며, 최대 57개까지 포함할 수 있습니다. 클러스터 전체를 포함하므로 기존 GT보다 신뢰성이 높습니다.

#### 데이터셋 구조
- **파일**: `data/gt_eval_fullquery_cluster_ids.jsonl`
- **총 쿼리 수**: 79개
- **평균 GT 문서 수**: 19.8개
- **최대 GT 문서 수**: 57개
- **최소 GT 문서 수**: 1개

#### GT 특징
- **클러스터 기반**: 같은 클러스터의 모든 문서를 GT로 사용
- **가변 크기**: 쿼리별 GT 문서수가 다름
- **중복 허용**: 같은 문서가 여러 쿼리의 GT에 포함될 수 있음

#### 쿼리 텍스트 구조
```markdown
질문: [사용자 질문]
전공: [전공 정보]
관심 직무: [관심 직무]
자격증: [자격증 목록]
동아리/대외활동: [활동 내역]
수강 이력:
[강의명] | [핵심 역량] | [강의 개요] | [학습 목표]
```

### 5.5.6 동적 top_k 설정

#### 문제점
- GT 문서 수가 top_k보다 큰 경우, R-recall 계산 불가
- 예: GT 32개, top_k=20 → 최대 recall 20/32 = 0.625

#### 해결 방법
```python
gt_count = len(ground_truth)
evaluation_top_k = min(max(base_top_k, gt_count), 60)
# 최소 top_k와 GT 개수 중 큰 값 사용, 최대 60개로 제한
```

이를 통해 가변 크기 GT에 대해 정확한 R-recall 계산이 가능합니다.

---

## 6. 실제 서비스 통합 가이드

### 6.1 통합 개요

이 섹션은 ForkExperiment에서 개발한 기능을 실제 Career-HY 서비스에 적용하는 방법을 설명합니다.

### 6.2 단계별 통합 방법

#### 1단계: 의존성 추가
```bash
pip install unstructured[pdf]>=0.10.0
```

#### 2단계: 파일 추가
- `implementations/loaders/structured_loader.py` → 서비스의 `loaders/structured_loader.py`
- `implementations/parsers/job_post_parser.py` → 서비스의 `parsers/job_post_parser.py`

#### 3단계: 데이터 로더 수정
- S3 JSON 메타데이터에서 `deadline`, `start_date`, `crawling_time` 포함
- 원본 메타데이터 전체 보존

#### 4단계: 청킹 로직 변경
```python
from loaders.structured_loader import StructuredDocumentLoader

loader = StructuredDocumentLoader(
    strategy="fast",
    target_sections=["preferred", "qualifications", "job_duties"],
    include_context=True
)
chunks = loader.load_from_documents(documents)
```

#### 5단계: 벡터 DB 저장 수정
- 메타데이터 전체 저장 (primitive 타입만)
- ChromaDB/FAISS 호환성 확인

#### 6단계: Response Generator 수정
- `RecommendedJob` 모델에 `deadline`, `start_date`, `crawling_time` 필드 추가
- 검색 결과에서 메타데이터 추출

#### 7단계: 프롬프트 수정
- 프롬프트에 시간 정보 포함

### 6.3 마이그레이션 체크리스트

#### 준비 단계
- [ ] 현재 서비스 코드베이스 구조 파악
- [ ] S3 JSON 메타데이터 구조 확인
- [ ] 기존 벡터 DB 백업
- [ ] 테스트 환경 준비

#### 코드 통합
- [ ] StructuredDocumentLoader 파일 추가
- [ ] JobPostParser 파일 추가
- [ ] 데이터 로더 수정
- [ ] 청킹 로직 변경
- [ ] 벡터 DB 저장 로직 확인
- [ ] Response Generator 수정
- [ ] 프롬프트 빌더 수정

#### 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 테스트
- [ ] A/B 테스트 준비

#### 배포
- [ ] 스테이징 환경 배포
- [ ] 모니터링 설정
- [ ] 프로덕션 배포
- [ ] 롤백 계획 수립

### 6.4 주의사항

#### 성능 고려사항
- **파싱 속도**: `fast` 전략이 `hi_res`보다 빠름 (약 10배)
- **정확도**: `hi_res`가 더 정확하지만 느림
- **권장**: 프로덕션에서는 `fast` 사용

#### 호환성
- 기존 벡터 DB는 새로운 구조와 호환되지 않을 수 있음
- 마이그레이션 또는 새 컬렉션 생성 필요

#### 에러 처리
- PDF 파싱 실패 시 fallback 처리 필요
- 메타데이터 누락 시 `None` 처리

자세한 내용은 `SERVICE_INTEGRATION_GUIDE.md` 참고 (향후 통합 문서로 이동 예정)

---

## 7. 참고 자료

### 7.1 문서
- `EXPERIMENT_SUMMARY_20251203.md`: 실험 요약 및 변경사항
- `GT_GENERATION_STRATEGY.md`: GT 생성 전략 상세
- `CLUSTERING_README.md`: 클러스터링 가이드
- `STRUCTURED_CHUNKS_REPORT.md`: 구조화 청크 리포트

### 7.2 외부 자료
- [Docker 공식 문서](https://docs.docker.com/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [RAG 평가 지표 가이드](https://docs.ragas.io/en/stable/concepts/metrics/)

### 7.3 데이터 소스
- **S3 버킷**: `career-hi`
- **PDF 경로**: `initial-dataset/pdf/` (1,473개 파일)
- **JSON 경로**: `initial-dataset/json/` (1,473개 파일)
- **Ground Truth**: `data/gt_eval_fullquery_cluster_ids.jsonl` (79개 쿼리)

---

## 📝 라이센스

이 프로젝트는 Career-HY 팀의 내부 실험용으로 개발되었습니다.

---

**작성일**: 2025-12-03  
**버전**: 2.0 (통합 문서)
