# 실제 서비스 통합 가이드

## 📋 개요

이 문서는 ForkExperiment에서 개발한 StructuredDocumentLoader와 메타데이터 확장 기능을 실제 Career-HY 서비스에 적용하는 방법을 설명합니다.

---

## 1. 현재 서비스 구조 분석

### 1.1 예상 구조 (확인 필요)

```
Career-HY Service
├── data_loader.py          # S3 데이터 로딩
├── chunker.py              # 텍스트 청킹
├── retriever.py            # 벡터 검색
├── response_generator.py   # 응답 생성
└── prompt_builder.py       # 프롬프트 구성
```

### 1.2 확인 사항

#### 데이터 로딩
- [ ] S3에서 PDF와 JSON 메타데이터를 어떻게 로드하는가?
- [ ] JSON 메타데이터에 `deadline`, `start_date`, `crawling_time` 필드가 있는가?
- [ ] 현재 메타데이터 구조는?

#### 청킹
- [ ] 현재 어떤 청킹 방식을 사용하는가? (Recursive, Fixed, None)
- [ ] 섹션 정보를 활용하는가?
- [ ] 청크 메타데이터에 어떤 정보를 포함하는가?

#### 검색
- [ ] 어떤 벡터 DB를 사용하는가? (ChromaDB, FAISS, 등)
- [ ] 검색 시 메타데이터 필터링을 사용하는가?
- [ ] 검색 결과에 어떤 메타데이터를 포함하는가?

#### 응답 생성
- [ ] RecommendedJob 모델에 어떤 필드가 있는가?
- [ ] 프롬프트에 어떤 정보를 포함하는가?

---

## 2. 단계별 통합 방법

### 2.1 1단계: 의존성 추가

#### 필요한 라이브러리
```python
# requirements.txt 또는 pyproject.toml에 추가
unstructured>=0.10.0  # JobPostParser용
unstructured[pdf]     # PDF 파싱용
```

#### 설치 명령
```bash
pip install unstructured[pdf]
# 또는
pip install "unstructured[pdf]>=0.10.0"
```

### 2.2 2단계: 파일 추가

#### 복사할 파일
```
Career-HY Service/
├── loaders/
│   ├── __init__.py
│   └── structured_loader.py          # 새로 추가
├── parsers/
│   ├── __init__.py
│   └── job_post_parser.py           # 새로 추가
└── ...
```

#### 파일 위치
- `implementations/loaders/structured_loader.py` → `loaders/structured_loader.py`
- `implementations/parsers/job_post_parser.py` → `parsers/job_post_parser.py`

### 2.3 3단계: 데이터 로더 수정

#### 현재 코드 (예상)
```python
# data_loader.py
def load_documents(pdf_prefix, json_prefix):
    metadata_map = load_json_metadata(json_prefix)
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        metadata = metadata_map[rec_idx]
        documents.append({
            "text": text,
            "metadata": {
                "rec_idx": rec_idx,
                "title": metadata.get("title"),
                "company": metadata.get("company"),
                "url": metadata.get("url"),
            }
        })
    return documents
```

#### 수정 후
```python
# data_loader.py
def load_documents(pdf_prefix, json_prefix):
    metadata_map = load_json_metadata(json_prefix)
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        raw_metadata = metadata_map[rec_idx]
        
        # 원본 메타데이터 전체 보존
        metadata = {
            **raw_metadata,  # deadline, start_date, crawling_time 포함
            "rec_idx": rec_idx,
            "title": raw_metadata.get("title") or raw_metadata.get("post_title") or "",
            "company": raw_metadata.get("company") or raw_metadata.get("company_name") or "",
            "url": raw_metadata.get("url") or raw_metadata.get("detail_url") or "",
        }
        
        documents.append({
            "text": text,
            "metadata": metadata
        })
    return documents
```

### 2.4 4단계: 청킹 로직 변경

#### 옵션 A: StructuredDocumentLoader 사용 (권장)

```python
# chunker.py 또는 document_processor.py
from loaders.structured_loader import StructuredDocumentLoader

class DocumentProcessor:
    def __init__(self, use_structured_loader=True):
        self.use_structured_loader = use_structured_loader
        if use_structured_loader:
            self.structured_loader = StructuredDocumentLoader(
                strategy="fast",  # 또는 "hi_res"
                target_sections=["preferred", "qualifications", "job_duties"],
                include_context=True
            )
    
    def process_documents(self, documents):
        if self.use_structured_loader:
            # StructuredDocumentLoader 사용
            chunks = self.structured_loader.load_from_documents(documents)
            # Chunk 객체를 딕셔너리로 변환
            return [chunk.to_dict() for chunk in chunks]
        else:
            # 기존 청킹 방식
            return self._chunk_with_recursive_splitter(documents)
```

#### 옵션 B: 기존 청커에 섹션 정보 추가 (점진적 적용)

```python
# 기존 청커를 유지하면서 섹션 정보만 추가
def chunk_with_section_info(text, metadata):
    # JobPostParser로 섹션 감지
    from parsers.job_post_parser import TextJobPostParser
    
    parser = TextJobPostParser(include_context=True)
    parsed_chunks = parser.process_text(text, metadata)
    
    # 기존 청커로 추가 세분화 (선택적)
    if need_further_chunking:
        chunks = recursive_splitter.split_text(parsed_chunks)
    else:
        chunks = parsed_chunks
    
    return chunks
```

### 2.5 5단계: 벡터 DB 저장 수정

#### ChromaDB 사용 시

```python
# retriever.py 또는 vector_store.py
def add_documents(self, documents, embeddings):
    # ChromaDB에 저장
    metadatas = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        
        # primitive 타입만 저장 (ChromaDB 제약)
        safe_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_metadata[key] = value
            elif isinstance(value, list):
                safe_metadata[key] = ", ".join(map(str, value))
            else:
                safe_metadata[key] = str(value)
        
        metadatas.append(safe_metadata)
    
    self.collection.add(
        ids=[doc["metadata"]["rec_idx"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        embeddings=embeddings,
        metadatas=metadatas
    )
```

#### FAISS 사용 시

```python
# FAISS는 메타데이터를 별도로 저장해야 함
def add_documents(self, documents, embeddings):
    # FAISS 인덱스에 벡터 저장
    self.index.add(embeddings)
    
    # 메타데이터는 별도 저장 (딕셔너리 또는 DB)
    for i, doc in enumerate(documents):
        doc_id = doc["metadata"]["rec_idx"]
        self.metadata_store[doc_id] = doc["metadata"]
```

### 2.6 6단계: 검색 결과 메타데이터 확인

#### ChromaDB 사용 시

```python
# retriever.py
def search(self, query_embedding, top_k=20):
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    for i in range(len(results["documents"][0])):
        doc = {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]  # 모든 메타데이터 포함
        }
        search_results.append(doc)
    
    return search_results
```

### 2.7 7단계: Response Generator 수정

#### RecommendedJob 모델 확장

```python
# models.py 또는 response_generator.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class RecommendedJob:
    rec_idx: str
    title: str
    url: str
    deadline: Optional[str] = None      # 추가
    start_date: Optional[str] = None     # 추가
    crawling_time: Optional[str] = None # 추가
    recommendation_reason: str = ""
```

#### 응답 생성 로직 수정

```python
# response_generator.py
def _convert_to_response(self, structured_result, retrieved_docs):
    recommended_jobs = []
    
    for job_index in structured_result.recommended_job_indices:
        doc_index = job_index - 1
        if 0 <= doc_index < len(retrieved_docs):
            doc = retrieved_docs[doc_index]
            metadata = doc.get("metadata", {})
            
            recommended_job = RecommendedJob(
                rec_idx=metadata.get("rec_idx"),
                title=metadata.get("title") or metadata.get("post_title", "제목 없음"),
                url=metadata.get("url") or metadata.get("detail_url", ""),
                deadline=metadata.get("deadline"),        # 추가
                start_date=metadata.get("start_date"),    # 추가
                crawling_time=metadata.get("crawling_time"), # 추가
                recommendation_reason=reason
            )
            recommended_jobs.append(recommended_job)
    
    return recommended_jobs
```

### 2.8 8단계: 프롬프트 수정

#### 프롬프트 빌더 수정

```python
# prompt_builder.py
def _format_retrieved_docs(self, retrieved_docs):
    formatted_docs = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get("metadata", {})
        
        doc_text = f"""[채용공고 {i}]
- 공고 ID: {metadata.get('rec_idx', 'N/A')}
- 제목: {metadata.get('title', 'N/A')}
- 회사: {metadata.get('company', 'N/A')}
- 마감일: {metadata.get('deadline', 'N/A')}      # 추가
- 시작일: {metadata.get('start_date', 'N/A')}     # 추가
- 크롤링 시간: {metadata.get('crawling_time', 'N/A')} # 추가
- URL: {metadata.get('url', 'N/A')}

채용 내용:
{doc.get('text', '')[:1000]}..."""
        
        formatted_docs.append(doc_text)
    
    return "\n\n".join(formatted_docs)
```

---

## 3. 설정 파일 예시

### 3.1 환경 변수

```bash
# .env 또는 환경 변수
USE_STRUCTURED_LOADER=true
STRUCTURED_PARSER_STRATEGY=fast  # fast 또는 hi_res
STRUCTURED_TARGET_SECTIONS=preferred,qualifications,job_duties
```

### 3.2 설정 파일 (YAML 또는 JSON)

```yaml
# config.yaml
data:
  use_structured_loader: true
  structured_parser_strategy: "fast"
  structured_target_sections:
    - "preferred"
    - "qualifications"
    - "job_duties"

chunker:
  type: "recursive"  # StructuredDocumentLoader 사용 시 "no_chunk" 또는 "recursive"
  chunk_size: 500
  chunk_overlap: 75
```

---

## 4. 마이그레이션 체크리스트

### 4.1 준비 단계
- [ ] 현재 서비스 코드베이스 구조 파악
- [ ] S3 JSON 메타데이터 구조 확인 (`deadline`, `start_date`, `crawling_time` 존재 여부)
- [ ] 기존 벡터 DB 백업
- [ ] 테스트 환경 준비

### 4.2 코드 통합
- [ ] `StructuredDocumentLoader` 파일 추가
- [ ] `JobPostParser` 파일 추가
- [ ] 데이터 로더 수정 (메타데이터 보존)
- [ ] 청킹 로직 변경
- [ ] 벡터 DB 저장 로직 확인
- [ ] Response Generator 수정
- [ ] 프롬프트 빌더 수정

### 4.3 테스트
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성
- [ ] 성능 테스트 (청킹 속도, 검색 품질)
- [ ] A/B 테스트 준비

### 4.4 배포
- [ ] 스테이징 환경 배포
- [ ] 모니터링 설정
- [ ] 프로덕션 배포
- [ ] 롤백 계획 수립

---

## 5. 주의사항

### 5.1 성능 고려사항

#### StructuredDocumentLoader
- **파싱 속도**: `fast` 전략이 `hi_res`보다 빠름 (약 10배)
- **정확도**: `hi_res`가 더 정확하지만 느림
- **권장**: 프로덕션에서는 `fast` 사용, 필요시 `hi_res` 선택적 사용

#### 메모리 사용
- 섹션별 청킹으로 청크 수 증가 (약 1.4배)
- 벡터 DB 저장 공간 증가 고려

### 5.2 호환성

#### 기존 데이터
- 기존 벡터 DB는 새로운 구조와 호환되지 않을 수 있음
- 마이그레이션 또는 새 컬렉션 생성 필요

#### API 변경
- `RecommendedJob` 모델에 필드 추가 → API 응답 형식 변경
- 클라이언트 호환성 확인 필요

### 5.3 에러 처리

#### 파싱 실패
- PDF 파싱 실패 시 fallback 처리 필요
- 텍스트 기반 파서로 대체

#### 메타데이터 누락
- `deadline`, `start_date`, `crawling_time`이 없는 경우 `None` 처리
- 프롬프트에서 "정보 없음" 표시

---

## 6. 롤백 계획

### 6.1 단계별 롤백

#### 1단계 롤백: StructuredDocumentLoader 비활성화
```python
# 설정 변경만으로 롤백 가능
USE_STRUCTURED_LOADER=false
```

#### 2단계 롤백: 메타데이터 필드 제거
```python
# Response Generator에서 필드 제거
deadline=None  # 항상 None으로 설정
```

#### 3단계 롤백: 코드 롤백
- Git으로 이전 버전으로 복귀
- 벡터 DB는 기존 백업 사용

---

## 7. 모니터링

### 7.1 주요 지표

#### 성능 지표
- 청킹 속도 (문서당 평균 시간)
- 검색 속도 (쿼리당 평균 시간)
- 메모리 사용량

#### 품질 지표
- 검색 정확도 (Recall@K, Precision@K)
- 응답 품질 (LangSmith 평가)
- 사용자 만족도

### 7.2 로깅

```python
# 주요 단계별 로깅
logger.info("StructuredDocumentLoader 초기화 완료")
logger.info(f"청크 생성 완료: {len(chunks)}개")
logger.info(f"검색 결과: {len(results)}개")
logger.warning(f"메타데이터 누락: {missing_fields}")
```

---

## 8. FAQ

### Q1: 기존 벡터 DB를 그대로 사용할 수 있나요?
**A**: 아니요. 섹션별 청킹으로 청크 구조가 변경되므로 새 컬렉션 생성이 필요합니다.

### Q2: `unstructured` 라이브러리가 없으면?
**A**: `TextJobPostParser`를 사용하여 텍스트 기반 파싱 가능 (PDF 파싱 불가).

### Q3: 성능 저하가 있나요?
**A**: 파싱 단계에서 약간의 오버헤드가 있지만, 검색 품질 향상으로 상쇄됩니다.

### Q4: 메타데이터가 없는 문서는?
**A**: `None`으로 처리하며, 프롬프트에서 "정보 없음"으로 표시됩니다.

---

**작성일**: 2025-12-03  
**버전**: 1.0

