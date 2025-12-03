# 대표 문서 선택 로직 개발 이력

## 📋 개요

`gt_generation_results/gt_clusters.json`의 각 클러스터에서 대표 문서 3개를 선택하는 로직을 개발하고 개선한 전체 과정을 정리합니다.

---

## 🔄 개발 과정 및 시행착오

### 1단계: 기존 로직 분석 (초기 상태)

#### 기존 대표 문서 선택 기준
- **파일**: `gt_generation_pipeline.py`의 `phase3_select_representative_docs` 함수
- **방법**: 점수 기반 선택
- **점수 계산식**:
  ```
  점수 = (대분류 다양성 × 2) + (중분류 다양성 × 1) + (태그 풍부도 × 0.1)
  ```
- **선택 방식**: 점수가 높은 상위 3개 문서 선택

#### 문제점
- 점수 기반 방식은 다양성을 중시하지만, 실제 클러스터의 특성과 유사한 문서를 선택하지 못함
- 태그가 많거나 여러 카테고리에 속한 문서가 우선 선택됨
- 클러스터의 핵심 특성(group_name, mid_categories)과의 유사도를 반영하지 못함

---

### 2단계: 첫 번째 개선 시도 (제목 + 태그 임베딩 기반)

#### 목표
- group_name별로 rec_ids 중에서 유사한 3개의 문서 선택
- 제목 + 태그 임베딩을 사용하여 문서 간 유사도 계산

#### 구현 내용
- **파일**: `gt_generation_pipeline.py`의 `phase3_select_representative_docs` 함수 수정
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (SentenceTransformer)
- **임베딩 텍스트**: `[제목] {태그1} {태그2}...` 형식
- **선택 알고리즘**:
  1. 첫 번째 문서: 클러스터의 첫 번째 rec_id
  2. 두 번째 문서: 첫 번째 문서와 가장 유사한 문서
  3. 세 번째 문서: 이미 선택된 문서들과의 평균 유사도가 가장 높은 문서

#### 코드 예시
```python
# 제목 + 태그 텍스트 생성
tags_str = " ".join(tags) if tags else ""
text = f"[{title}] {tags_str}".strip()

# 임베딩 생성
embeddings = model.encode(valid_docs, ...)

# 유사도 기반으로 3개 선택
selected_indices = select_similar_docs(embeddings, n)
```

#### 문제점
- 문서 간 유사도는 계산하지만, 클러스터의 특성(group_name, mid_categories)과의 유사도를 직접 반영하지 못함
- 클러스터의 의미와 가장 일치하는 문서를 선택하는 것이 아니라, 서로 유사한 문서들을 선택함

---

### 3단계: 두 번째 개선 시도 (쿼리-문서 유사도 기반)

#### 목표
- 클러스터의 `group_name + mid_categories` 텍스트를 쿼리로 사용
- 각 문서의 `cleaned_title`과 쿼리 간의 유사도를 계산
- 가장 유사한 상위 3개 문서 선택

#### 구현 내용
- **새 파일**: `update_representative_docs.py` 작성
- **쿼리 생성**: `group_name + mid_categories` → 예: "품질관리 QA QC"
- **문서 텍스트**: 각 rec_id에 해당하는 `cleaned_title` 사용
- **유사도 계산**: 코사인 유사도 (임베딩 벡터 간 내적)

#### 코드 예시
```python
# 쿼리 텍스트 생성
query_text = f"{group_name} {mid_cats_str}".strip()
# 예: "품질관리 QA QC"

# 쿼리 임베딩
query_embedding = model.encode([query_text], ...)[0]

# 문서 제목들 임베딩
doc_embeddings = model.encode(valid_cleaned_titles, ...)

# 유사도 계산
similarities = np.dot(doc_embeddings, query_embedding)

# 상위 3개 선택
top_indices = np.argsort(similarities)[::-1][:3]
```

#### 장점
- 클러스터의 의미적 특성과 직접적으로 일치하는 문서 선택
- `group_name`과 `mid_categories`가 클러스터의 핵심을 나타내므로, 이를 쿼리로 사용하면 더 정확한 선택 가능

---

### 4단계: 최종 통합 (파이프라인 통합)

#### 목표
- `update_representative_docs.py`의 로직을 `gt_generation_pipeline.py`에 통합
- 한 번의 실행으로 전체 파이프라인 완료
- 명확한 문서화 추가

#### 구현 내용
- **파일**: `gt_generation_pipeline.py`의 `phase3_select_representative_docs` 함수 재작성
- **로직**: 3단계의 쿼리-문서 유사도 방식 적용
- **문서화**: 입력/출력 데이터, 처리 과정, 예시 추가

#### 최종 로직
```python
def phase3_select_representative_docs(clusters, doc_map, n=3):
    """
    각 클러스터별로:
    1. group_name + mid_categories → 쿼리 텍스트 생성
    2. 해당 클러스터의 rec_ids에 속한 문서들의 cleaned_title 수집
    3. 쿼리 임베딩과 문서 제목 임베딩 간 코사인 유사도 계산
    4. 가장 유사한 상위 n개 문서 선택
    """
    for cluster_name, cluster_data in clusters.items():
        # 쿼리 생성
        query_text = f"{group_name} {' '.join(mid_categories)}"
        
        # 문서 제목 수집
        cleaned_titles = [doc_map[rec_id]['cleaned_title'] 
                         for rec_id in rec_ids 
                         if rec_id in doc_map]
        
        # 임베딩 및 유사도 계산
        query_emb = model.encode([query_text])[0]
        doc_embs = model.encode(cleaned_titles)
        similarities = np.dot(doc_embs, query_emb)
        
        # 상위 3개 선택
        top_indices = np.argsort(similarities)[::-1][:3]
        representative_docs = [rec_ids[i] for i in top_indices]
```

---

## 📊 최종 결과

### 입력 데이터
1. **clustering_results_tag_based/*_classification.json**
   - 각 대분류별 분류 결과 파일
   - 각 문서는 `rec_id`, `title`, `cleaned_title`, `tags`, `company` 포함

2. **similarity_rules_template.json**
   - 유사 중분류를 그룹으로 묶는 규칙
   - 형식: `{대분류: {group_name: [중분류1, 중분류2, ...]}}`

### 출력 데이터
**gt_generation_results/gt_clusters.json**
```json
{
  "metadata": {
    "created_at": "2025-11-25T10:37:33.385176",
    "total_clusters": 88,
    "total_docs": 2844,
    "unique_docs": 1078
  },
  "clusters": {
    "#건설·건축_품질관리": {
      "major_category": "#건설·건축",
      "group_name": "품질관리",
      "mid_categories": ["QA", "QC"],
      "num_docs": 32,
      "rec_ids": ["50285218", "50463673", ...],
      "representative_docs": ["50314015", "50440884", "50463673"]
    }
  }
}
```

### 처리 과정
1. **Phase 1**: 중분류 기반 초기 클러스터 생성
2. **Phase 2**: 유사 중분류 병합 (규칙 기반)
3. **Phase 3**: 대표 문서 선택 (쿼리-문서 유사도 기반) ⭐
4. **Phase 4**: 통계 및 결과 저장

---

## 🎯 핵심 개선 사항

### Before (기존 방식)
- 점수 기반: 다양성과 태그 풍부도만 고려
- 클러스터 특성과 무관한 문서 선택 가능

### After (최종 방식)
- 쿼리-문서 유사도: 클러스터의 의미적 특성과 직접 매칭
- `group_name + mid_categories`를 쿼리로 사용하여 정확도 향상
- `cleaned_title`만 사용하여 간결하고 정확한 비교

---

## 📝 사용법

### 실행
```bash
python gt_generation_pipeline.py
```

### 결과 확인
- `gt_generation_results/gt_clusters.json`: 각 클러스터의 `representative_docs` 필드 확인
- `gt_generation_results/gt_clusters_summary.csv`: 요약 정보 확인
- `gt_generation_results/gt_generation_statistics.txt`: 통계 정보 확인

---

## 🔍 예시: 대표 문서 선택 과정

### 클러스터 정보
- **클러스터명**: `#건설·건축_품질관리`
- **group_name**: `품질관리`
- **mid_categories**: `["QA", "QC"]`
- **rec_ids**: `["50285218", "50463673", "50314015", ...]`

### 처리 단계
1. **쿼리 생성**: `"품질관리 QA QC"`
2. **문서 제목 수집**:
   - `"50285218"`: `"태광 QA팀"`
   - `"50463673"`: `"품질관리 QA"`
   - `"50314015"`: `"QA 품질보증"`
   - ...
3. **임베딩 생성**:
   - 쿼리 임베딩: `[0.12, -0.05, 0.33, ...]` (768차원)
   - 문서 임베딩들: 각각 768차원 벡터
4. **유사도 계산**:
   - `"50314015"`: 0.85 (가장 높음)
   - `"50440884"`: 0.82
   - `"50463673"`: 0.80
   - ...
5. **최종 선택**: `["50314015", "50440884", "50463673"]`

---

## 🛠️ 기술 스택

- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (SentenceTransformer)
- **유사도 계산**: 코사인 유사도 (정규화된 임베딩 벡터 간 내적)
- **Python 라이브러리**:
  - `sentence-transformers`: 임베딩 생성
  - `numpy`: 벡터 연산
  - `tqdm`: 진행 상황 표시

---

## 📌 참고 사항

1. **`update_representative_docs.py`**: 이제 사용되지 않음 (통합 완료)
2. **Fallback 메커니즘**: 모델 로드 실패 시 기존 점수 기반 방법으로 대체
3. **문서 부족 처리**: 클러스터 내 문서가 3개 미만인 경우 모든 문서 선택

---

## ✅ 완료된 작업

- [x] 기존 로직 분석
- [x] 제목+태그 임베딩 기반 방식 구현 (1차 시도)
- [x] 쿼리-문서 유사도 기반 방식 구현 (2차 시도)
- [x] 별도 스크립트 작성 (`update_representative_docs.py`)
- [x] 파이프라인 통합 (`gt_generation_pipeline.py`)
- [x] 문서화 완료
- [x] 최종 테스트 및 검증

---

## 🎓 교훈

1. **의미적 유사도가 중요**: 단순히 문서 간 유사도보다는 클러스터 특성과의 유사도가 더 중요
2. **쿼리 기반 접근**: 클러스터의 핵심 특성을 쿼리로 사용하면 더 정확한 결과
3. **단계적 개선**: 점수 기반 → 문서 간 유사도 → 쿼리-문서 유사도로 단계적 개선
4. **통합의 중요성**: 별도 스크립트보다는 파이프라인에 통합하는 것이 유지보수에 유리

---

**작성일**: 2025-11-25  
**최종 수정일**: 2025-11-25  
**버전**: 1.0


