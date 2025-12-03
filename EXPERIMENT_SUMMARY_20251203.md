# 12/3 최종 실험 결과 정리 및 리팩토링

## 📋 목차
1. [중간 이후의 문제점 및 해결](#1-중간-이후의-문제점-및-해결)
2. [GT Agent 설계](#2-gt-agent-설계)
3. [GT Dataset 완성](#3-gt-dataset-완성)
4. [실험 설정 및 결과](#4-실험-설정-및-결과)
5. [실제 서비스 적용 방법](#5-실제-서비스-적용-방법)
6. [변경된 코드 및 파일](#6-변경된-코드-및-파일)

---

## 1. 중간 이후의 문제점 및 해결

### 1.1 StructuredDocumentLoader 구현

#### 문제점
- 기존: 단순 텍스트 청킹으로 섹션 정보 손실
- 채용공고의 구조화된 정보(우대사항, 자격요건, 주요업무)를 활용하지 못함

#### 해결 방법
**StructuredDocumentLoader** 구현:
- `JobPostParser`를 활용한 섹션별 청킹
- 섹션 타입별로 청크 분리: `preferred`, `qualifications`, `job_duties`
- 각 청크에 섹션 타입 메타데이터 포함

#### 주요 특징
```python
# StructuredDocumentLoader
- 섹션별 청크 생성 (preferred, qualifications, job_duties)
- ChromaDB 호환 메타데이터 자동 생성
- 태그는 문서 레벨에서 추출 (모든 청크에 공유)
- benefits, hiring_process, notes는 제외
```

#### 통계
- 총 청크: 2,039개 (1,473개 문서)
- 섹션별 분포:
  - `preferred`: 1,036개 (50.8%)
  - `qualifications`: 398개 (19.5%)
  - `job_duties`: 279개 (13.7%)
  - `full_text` (fallback): 326개 (16.0%)

### 1.2 JobPostParser로 section_type 분리

#### 구현 내용
**JobPostParser** (`implementations/parsers/job_post_parser.py`):
- `unstructured` 라이브러리 기반 PDF 파싱
- 섹션 감지: "우대사항", "자격요건", "주요업무" 등
- 태그 추출: `#태그` 형식 자동 감지
- Context 주입: 각 청크에 `[회사: ...] [직무: ...]` 추가

**TextJobPostParser** (텍스트 기반):
- S3DataLoader에서 추출된 텍스트 직접 파싱
- PDF 파싱 없이 빠른 처리
- 섹션 감지 로직 동일

#### 섹션 감지 로직
```python
# 섹션 키워드 매핑
SECTION_KEYWORDS = {
    "preferred": ["우대사항", "우대", "이런 분이면 더 좋아요"],
    "qualifications": ["자격요건", "지원자격", "자격"],
    "job_duties": ["주요업무", "담당업무", "업무내용"],
    # ...
}
```

### 1.3 메타데이터 보존 및 확장

#### 문제점
- S3 JSON 메타데이터(`deadline`, `start_date`, `crawling_time`)가 청크에 포함되지 않음
- 검색 결과에서 시간 정보 누락

#### 해결 방법
1. **S3DataLoader** (`utils/data_loader.py`):
   - `**raw_metadata`로 원본 JSON 전체 보존
   - `deadline`, `start_date`, `crawling_time` 포함

2. **StructuredDocumentLoader** (`implementations/loaders/structured_loader.py`):
   - `base_metadata = {**doc_metadata, ...}`로 원본 메타데이터 전체 보존
   - 명시적으로 `deadline`, `start_date`, `crawling_time` 추가

3. **ChromaRetriever** (`implementations/retrievers/chroma_retriever.py`):
   - primitive 타입(`None`, `str`, `int`, `float`, `bool`) 저장 지원
   - 메타데이터 전체 저장

4. **Response Generator** (`implementations/response_generators/careerhy_generator.py`):
   - 검색된 문서의 메타데이터에서 `deadline`, `start_date`, `crawling_time` 추출
   - 생성된 응답에 포함

---

## 2. GT Agent 설계

### 2.1 전략: 중분류 기반 클러스터링

#### 핵심 아이디어
- **중분류 = 직무 그룹**으로 간주
- 유사한 중분류를 병합하여 의미 있는 클러스터 생성
- 클러스터별로 사용자 프로필 기반 쿼리 생성

#### 단계별 프로세스
1. **중분류별 문서 수집**: 최소 5개 이상 문서를 가진 중분류만 사용
2. **유사 중분류 병합**: 규칙 기반 + 임베딩 기반 병합
3. **대표 문서 선택**: 다양성, 범용성, 정보성 점수 기반
4. **쿼리 생성**: LLM 기반 사용자 프로필 생성
5. **GT 매칭**: 전체 클러스터 문서를 Ground Truth로 설정

### 2.2 구현 파일
- `GT_GENERATION_STRATEGY.md`: 상세 전략 문서
- `gt_generation_pipeline.py`: GT 생성 파이프라인
- `similarity_rules.json`: 유사 중분류 병합 규칙

---

## 3. GT Dataset 완성

### 3.1 데이터셋 구조

#### 입력 데이터
- **파일**: `data/gt_eval_fullquery_cluster_ids.jsonl`
- **형식**: JSONL (각 라인이 하나의 쿼리)
- **필드**:
  - `query_id`: 쿼리 고유 ID
  - `query_text`: 완전한 검색 쿼리 (질문 + 사용자 프로필)
  - `ground_truth`: Ground Truth 문서 리스트 (`rec_idx` 배열)

#### 쿼리 텍스트 구조
```
질문: [사용자 질문]
전공: [전공 정보]
관심 직무: [관심 직무]
자격증: [자격증 목록]
동아리/대외활동: [활동 내역]
수강 이력:
[강의명] | [핵심 역량] | [강의 개요] | [학습 목표]
...
```

#### Ground Truth 특징
- **클러스터 기반**: 같은 클러스터의 모든 문서를 GT로 사용
- **가변 크기**: 쿼리별 GT 문서 수가 다름 (평균 19.8개, 최대 57개)
- **중복 허용**: 같은 문서가 여러 쿼리의 GT에 포함될 수 있음

### 3.2 통계
- **총 쿼리 수**: 79개
- **평균 GT 문서 수**: 19.8개
- **최대 GT 문서 수**: 57개
- **최소 GT 문서 수**: 1개

---

## 4. 실험 설정 및 결과

### 4.1 실험 설정 파일

#### 1) Baseline: Recursive Chunking
**파일**: `configs/baseline_search.yaml`
- **Chunker**: `recursive` (chunk_size: 700, chunk_overlap: 100)
- **StructuredDocumentLoader**: 사용 안 함
- **목적**: 기존 방식과 비교

#### 2) StructuredDocumentLoader (섹션별 청킹)
**파일**: `configs/new_eval_baseline.yaml`
- **Chunker**: `no_chunk` (StructuredDocumentLoader가 청킹 수행)
- **StructuredDocumentLoader**: 사용 (`use_structured_loader: true`)
- **Target Sections**: `preferred`, `qualifications`, `job_duties`
- **목적**: 섹션별 청킹 효과 측정

#### 3) StructuredDocumentLoader + Recursive Chunking
**파일**: `configs/new_eval_baseline_recursive.yaml`
- **Chunker**: `recursive` (chunk_size: 500, chunk_overlap: 75)
- **StructuredDocumentLoader**: 사용
- **목적**: 섹션별 청킹 + 추가 세분화 효과 측정

### 4.2 평가 지표

#### Retrieval Metrics
1. `ndcg@10`: 순위 품질 (상위 10개)
2. `mrr@10`: 첫 정답 순위 (상위 10개)
3. `precision@3/5/10/20`: 정확도 (상위 K개)
4. `hit@10_count`: 상위 10개 중 맞은 개수
5. `r_recall`: Recall@(정답개수) - 가변 GT 크기 대응
6. `recall@10`: Recall@10 (추가)

#### Generation Metrics (LangSmith)
1. `recommendation_quality`: 추천 품질 전반
2. `personalization_score`: 개인화 수준
3. `response_helpfulness`: 도움이 되는 정도
4. `profile_alignment`: 프로필 일치도

### 4.3 동적 top_k 설정

#### 문제점
- GT 문서 수가 `top_k`보다 큰 경우 R-recall 계산 불가
- 예: GT 32개, `top_k=20` → 최대 recall 20/32 = 0.625

#### 해결 방법
```python
# pipeline.py의 _evaluate_generation_for_samples
gt_count = len(ground_truth)
evaluation_top_k = min(max(base_top_k, gt_count), 60)
# 최소 top_k와 GT 개수 중 큰 값 사용, 최대 60개로 제한
```

---

## 5. 실제 서비스 적용 방법

### 5.1 현재 서비스 구조 (추정)

#### 데이터 로딩
- S3에서 PDF + JSON 메타데이터 로드
- PDF 텍스트 추출
- 단순 텍스트 청킹 또는 전체 문서 사용

#### 청킹
- Recursive chunking 또는 전체 문서
- 섹션 정보 미활용

#### 검색
- ChromaDB 벡터 검색
- 메타데이터: `rec_idx`, `title`, `company`, `url` 등

### 5.2 적용 방법

#### 1단계: StructuredDocumentLoader 통합

**변경 파일**: 데이터 로딩 및 청킹 로직

```python
# 기존
documents = load_documents_from_s3()
chunks = chunker.chunk(documents)

# 변경 후
from implementations.loaders.structured_loader import StructuredDocumentLoader

loader = StructuredDocumentLoader(
    strategy="fast",  # 또는 "hi_res"
    target_sections=["preferred", "qualifications", "job_duties"],
    include_context=True
)
chunks = loader.load_from_documents(documents)
```

**필요한 파일**:
- `implementations/loaders/structured_loader.py`
- `implementations/parsers/job_post_parser.py`
- `utils/data_loader.py` (메타데이터 보존 확인)

#### 2단계: 메타데이터 확장

**변경 사항**:
- S3 JSON 메타데이터에서 `deadline`, `start_date`, `crawling_time` 포함
- ChromaDB 저장 시 메타데이터 전체 보존
- 검색 결과에 시간 정보 포함

**확인 사항**:
- S3 JSON에 `deadline`, `start_date`, `crawling_time` 필드 존재 여부
- ChromaDB 스키마 변경 필요 여부 (없음 - 동적 스키마)

#### 3단계: Response Generator 수정

**변경 파일**: 응답 생성 로직

```python
# 기존
recommended_job = RecommendedJob(
    rec_idx=metadata.get("rec_idx"),
    title=metadata.get("title"),
    url=metadata.get("url"),
    # deadline, start_date, crawling_time 없음
)

# 변경 후
recommended_job = RecommendedJob(
    rec_idx=metadata.get("rec_idx"),
    title=metadata.get("title"),
    url=metadata.get("url"),
    deadline=metadata.get("deadline"),  # 추가
    start_date=metadata.get("start_date"),  # 추가
    crawling_time=metadata.get("crawling_time"),  # 추가
)
```

**필요한 파일**:
- `implementations/response_generators/careerhy_generator.py`
- `core/interfaces/response_generator.py` (RecommendedJob 모델)

#### 4단계: 프롬프트 수정

**변경 파일**: `services/prompt_builder.py`

**변경 사항**:
- 프롬프트에 `deadline`, `start_date`, `crawling_time` 포함
- 섹션 타입 정보 활용 (선택적)

```python
# _format_retrieved_docs 메서드
doc_text = f"""[채용공고 {i}]
- 공고 ID: {metadata.get('rec_idx', 'N/A')}
- 제목: {metadata.get('title', 'N/A')}
- 회사: {metadata.get('company', 'N/A')}
- 마감일: {metadata.get('deadline', 'N/A')}  # 추가
- 시작일: {metadata.get('start_date', 'N/A')}  # 추가
- 크롤링 시간: {metadata.get('crawling_time', 'N/A')}  # 추가
- URL: {metadata.get('url', 'N/A')}

채용 내용:
{doc.get('text', '')[:1000]}..."""
```

### 5.3 마이그레이션 체크리스트

#### 데이터 준비
- [ ] S3 JSON 메타데이터에 `deadline`, `start_date`, `crawling_time` 확인
- [ ] 기존 ChromaDB 컬렉션 백업
- [ ] 새로운 컬렉션 생성 (섹션별 청킹용)

#### 코드 통합
- [ ] `StructuredDocumentLoader` 통합
- [ ] `JobPostParser` 의존성 추가 (`unstructured` 라이브러리)
- [ ] 메타데이터 보존 로직 확인
- [ ] Response Generator 수정
- [ ] 프롬프트 수정

#### 테스트
- [ ] 단위 테스트: StructuredDocumentLoader
- [ ] 통합 테스트: 전체 파이프라인
- [ ] 성능 테스트: 청킹 속도, 검색 품질
- [ ] A/B 테스트: 기존 방식 vs 새로운 방식

#### 배포
- [ ] 스테이징 환경 테스트
- [ ] 프로덕션 배포
- [ ] 모니터링 설정

---

## 6. 변경된 코드 및 파일

### 6.1 핵심 파일 목록

#### 데이터 로딩 및 청킹
- `implementations/loaders/structured_loader.py`: StructuredDocumentLoader 구현
- `implementations/parsers/job_post_parser.py`: JobPostParser (PDF 파싱)
- `utils/data_loader.py`: S3DataLoader (메타데이터 보존)

#### 검색 및 저장
- `implementations/retrievers/chroma_retriever.py`: ChromaRetriever (메타데이터 저장)
- `core/pipeline.py`: ExperimentPipeline (동적 top_k, 메타데이터 전달)

#### 응답 생성
- `implementations/response_generators/careerhy_generator.py`: Response Generator (메타데이터 추출)
- `services/prompt_builder.py`: 프롬프트 빌더 (메타데이터 포함)

#### 평가
- `implementations/evaluators/retrieval_evaluator.py`: RetrievalEvaluator (recall@10 추가, R-recall 수정)
- `implementations/evaluators/evaluators_back/langsmith_evaluator.py`: LangSmith 평가 (프로필 정보 확장)

#### 설정 파일
- `configs/baseline_search.yaml`: Baseline 설정
- `configs/new_eval_baseline.yaml`: StructuredDocumentLoader 설정
- `configs/new_eval_baseline_recursive.yaml`: StructuredDocumentLoader + Recursive 설정

### 6.2 주요 변경 사항

#### 1. 메타데이터 보존
- **파일**: `utils/data_loader.py`, `implementations/loaders/structured_loader.py`
- **변경**: `**raw_metadata` 또는 `**doc_metadata`로 원본 메타데이터 전체 보존
- **추가 필드**: `deadline`, `start_date`, `crawling_time`

#### 2. 동적 top_k
- **파일**: `core/pipeline.py`
- **변경**: `evaluation_top_k = min(max(base_top_k, gt_count), 60)`
- **목적**: GT 문서 수에 맞춰 검색 개수 조정

#### 3. 쿼리 파싱
- **파일**: `core/pipeline.py`
- **변경**: `_parse_query_text` 메서드 추가
- **기능**: `query_text`에서 질문과 사용자 프로필 분리

#### 4. 평가 지표
- **파일**: `implementations/evaluators/retrieval_evaluator.py`
- **변경**: `recall@10` 추가, R-recall 로직 수정
- **목적**: 가변 GT 크기 대응

#### 5. LangSmith 평가
- **파일**: `implementations/evaluators/evaluators_back/langsmith_evaluator.py`
- **변경**: 프로필 정보 확장 (certification, club_activities, catalogs)
- **변경**: 추천 공고 전체 텍스트 포함

---

## 7. 다음 단계

### 7.1 실험 결과 분석
- [ ] 세 가지 설정 비교 분석
- [ ] 섹션별 청킹 효과 검증
- [ ] Recursive Chunking 추가 효과 검증

### 7.2 코드 리팩토링
- [ ] 실제 서비스 코드베이스 확인
- [ ] 통합 계획 수립
- [ ] 단계별 마이그레이션

### 7.3 문서화
- [ ] API 문서 업데이트
- [ ] 사용 가이드 작성
- [ ] 트러블슈팅 가이드 작성

---

## 8. 참고 자료

- `GT_GENERATION_STRATEGY.md`: GT 생성 전략 상세 문서
- `STRUCTURED_CHUNKS_REPORT.md`: StructuredDocumentLoader 통계 리포트
- `EXPERIMENT_GUIDE.md`: 실험 가이드
- `CLUSTERING_README.md`: 클러스터링 가이드

---

**작성일**: 2025-12-03  
**작성자**: [작성자명]  
**버전**: 1.0

