# Career-HY RAG 실험 가이드

체계적이고 재현 가능한 RAG 실험을 위한 종합 가이드입니다.


## 📊 실험 카테고리

### A. 임베딩 모델 실험
서로 다른 임베딩 모델의 성능을 비교합니다.

**실험 대상:**
- `text-embedding-ada-002` (베이스라인)
- `text-embedding-3-small`
- `text-embedding-3-large`
- 등등

**실험 설정:**
```yaml
embedder:
  type: "openai"
  model_name: "text-embedding-3-large"  # 변경 포인트
  batch_size: 5
```

### B. 청킹 전략 실험
문서 분할 방법에 따른 성능 차이를 측정합니다.

**실험 대상:**
- `no_chunk` (베이스라인) - 전체 문서 사용
- `recursive` - RecursiveCharacterTextSplitter
- `token` - 토큰 기반 분할

**실험 설정:**
```yaml
chunker:
  type: "recursive"
  chunk_size: 1000      # 실험 변수
  chunk_overlap: 200    # 실험 변수
```

**실험 변수:**
- `chunk_size`: 500, 1000, 1500, 2000 등
- `chunk_overlap`: 0, 100, 200, 300 등

### C. 검색 파라미터 실험
검색 시스템의 설정을 최적화합니다.

**실험 대상:**
- `top_k`: 검색할 문서 수
- `similarity_threshold`: 유사도 임계값

**실험 설정:**
```yaml
retriever:
  type: "chroma"
  top_k: 20           # 5, 10, 15, 20, 25
  similarity_threshold: 0.7  # 0.5, 0.6, 0.7, 0.8
```

### D. 조합 실험
성능이 좋은 개별 설정들의 조합을 테스트합니다.


## 🔬 실험 설정 파일 작성법

### 기본 템플릿
```yaml
# 실험 메타데이터
experiment_name: "descriptive_name"
description: "실험 목적과 변경사항 설명"
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

# 검색 설정
retriever:
  type: "chroma"
  collection_name: "unique_collection_name"  # 실험별 고유값
  persist_directory: "/tmp/chroma_unique"    # 실험별 고유값
  top_k: 10

# LLM 설정
llm:
  type: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000

# 데이터 설정 (고정)
data:
  pdf_prefix: "initial-dataset/pdf/"
  json_prefix: "initial-dataset/json/"
  test_queries_path: "data/test_queries.jsonl"

# 평가 설정 (고정)
evaluation:
  metrics: ["recall@k", "precision@k", "mrr", "map", "ndcg@k"]
  k_values: [1, 3, 5, 10]
```

### 실험별 고유값 설정
충돌 방지를 위해 각 실험마다 고유한 식별자를 사용합니다.

```yaml
# 베이스라인
collection_name: "job-postings-baseline"
persist_directory: "/tmp/chroma_baseline"

# 청킹 실험
collection_name: "job-postings-chunk-1000-200"
persist_directory: "/tmp/chroma_chunk_1000_200"

# 임베딩 실험
collection_name: "job-postings-3large"
persist_directory: "/tmp/chroma_3large"
```


## 🎯 실험 시나리오 예시

**상황** : baseline에서 chunk를 recursive chunk로 바꾼 후 결과를 확인하겠다.
1. core/interfaces/cunker.py에서 BaseChunker의 스펙 확인
2. implementations/chunkers 디렉토리에 recursive_chunker.py 파일을 만든 뒤 BaseChunker를 상속받는 구현체 코드 작성 (chunk 함수 오버라이딩 / __init__.py에 새로운 전략 등록)
3. configs 디렉토리에 baseline.yaml과 같은 형식으로 새로운 설정파일 작성
   - file name: recursive-test-1.yaml
   - experiment_name: "recursive-test-1"
   - description: "기존 베이스라인 실험 환경에서 recursive 청킹 적용 \n chunk_size: 1000, chunk_overlap: 200"
   - chunker
     - type: "recursive"
     - chunk_size: 1000
     - chunk_overlap: 200
4. utils/factor.py에 새로운 전략 등록
   - ```
       _chunkers: Dict[str, Type[BaseChunker]] = {
        "no_chunk": NoChunker,
        "recursive": RecursiveChunker,
    }
    ```
  - "recursive" - recursive-test-1.yaml 파일에 정의된 chunker의 type
  - RecursiveChunker - BaseChunker를 상속받아 구현한 구현체 클래스 이름
5. 실험 진행
   - docker compose build --no-cache
   - ./run_experiment.sh configs/recursive-test-1.yaml
6. result 디렉토리에서 결과 확인
7. https://smith.langchain.com/ 에서 상세 결과 확인
8. 노션에 실험 결과 정리
