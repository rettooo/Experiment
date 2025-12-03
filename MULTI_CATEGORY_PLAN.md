# 다중 대분류 할당 계획

## 목표
- 하나의 문서(rec_id)가 여러 대분류에 속할 수 있도록 수정
- 각 대분류 내에서 중분류로 분류
- 나중에 군집화를 한번 더 수행할 수 있도록 구조 유지

## 예시: rec_id "50496628"
**태그:**
- `#디자인`
- `#기획·전략`
- `#마케팅·홍보·조사`
- `#미디어·문화·스포츠`
- `#PD/AD/FD` (중분류 키워드)

**할당 결과:**
1. **#디자인** → 중분류: ["콘텐츠디자인"]
2. **#기획·전략** → 중분류: ["기획", "콘텐츠기획"] (둘 다 매칭)
3. **#마케팅·홍보·조사** → 중분류: ["콘텐츠마케팅", "시장조사"] (둘 다 매칭)
4. **#미디어·문화·스포츠** → 중분류: ["PD/AD/FD"]

---

## 수정 계획

### 1. `assign_major_category` → `assign_major_categories` 변경
**현재:**
```python
def assign_major_category(...) -> str:
    # 하나의 대분류만 반환
    return major_cat
```

**변경 후:**
```python
def assign_major_categories(...) -> List[str]:
    """
    여러 대분류 할당
    
    우선순위:
    1. tags에서 대분류 태그 직접 매칭 (예: "#영업·판매·무역")
    2. title + tags에서 중분류 키워드 매칭 → 해당 대분류 추가
    3. 제목에서 직접 대분류 키워드 매칭
    4. 최소 1개는 할당 (없으면 "기타")
    
    Returns:
        대분류명 리스트 (예: ["#디자인", "#기획·전략", "#마케팅·홍보·조사"])
    """
    major_categories = set()
    
    # 1. 태그에서 직접 매칭
    for tag in tags:
        for major_cat in job_classification.keys():
            if tag.lower() == major_cat.lower():
                major_categories.add(major_cat)
    
    # 2. 중분류 키워드 매칭 (제목 우선 가중치)
    # 각 대분류별로 점수 계산하고, 점수가 0보다 크면 추가
    scores = {}
    for major_cat, mid_keywords_list in job_classification.items():
        score = 0
        for keyword in mid_keywords_list:
            if keyword.lower() in title_lower:
                score += 2
            if keyword.lower() in tags_text:
                score += 1
        if score > 0:
            scores[major_cat] = score
            major_categories.add(major_cat)
    
    # 3. 제목에서 직접 대분류 키워드 매칭
    # (기존 major_keywords 사용)
    
    # 최소 1개는 할당
    if not major_categories:
        major_categories.add("기타")
    
    return sorted(list(major_categories))
```

### 2. `aggregate_by_rec_id` 수정
**현재:**
```python
doc_map[rec_id] = {
    ...
    "major_category": major_category,  # 단일 값
    "mid_category": mid_category,       # 단일 값
}
```

**변경 후:**
```python
doc_map[rec_id] = {
    ...
    "major_categories": major_categories,  # List[str]
    "category_assignments": {               # Dict[str, List[str]]
        major_cat: assign_mid_categories(tags, title, major_cat)
        for major_cat in major_categories
    }
}
```

### 3. `save_tag_based_results` 수정
**현재:**
- 각 문서를 하나의 대분류에만 할당
- 대분류별로 문서를 그룹화

**변경 후:**
- 각 문서를 여러 대분류에 할당
- 대분류별로 문서 목록을 저장하되, 같은 문서가 여러 대분류에 나타날 수 있음
- 각 대분류 내에서 중분류별로 그룹화

**저장 구조:**
```json
{
  "major_category": "#디자인",
  "statistics": {
    "total_docs": 150,
    "unique_docs": 145,  // 중복 제거
    "num_mid_categories": 12,
    "overlap_info": {
      "with_#기획·전략": 45,
      "with_#마케팅·홍보·조사": 38,
      ...
    }
  },
  "mid_categories": {
    "콘텐츠디자인": {
      "size": 1,
      "documents": [
        {
          "rec_id": "50496628",
          "title": "...",
          "tags": [...],
          "other_major_categories": ["#기획·전략", "#마케팅·홍보·조사", "#미디어·문화·스포츠"],
          "mid_categories_in_this_major": ["콘텐츠디자인"],
          "mid_categories_in_others": {
            "#기획·전략": ["기획", "콘텐츠기획"],
            "#마케팅·홍보·조사": ["콘텐츠마케팅", "시장조사"],
            "#미디어·문화·스포츠": ["PD/AD/FD"]
          }
        }
      ]
    }
  }
}
```

### 4. `assign_mid_category` → `assign_mid_categories` 변경
**현재:**
```python
def assign_mid_category(...) -> str:
    # 하나의 중분류만 반환
    return mid_cat
```

**변경 후:**
```python
def assign_mid_categories(...) -> List[str]:
    """
    여러 중분류 할당
    
    우선순위:
    1. 제목에서 중분류 키워드 매칭 (가중치 3배)
    2. 태그에서 중분류 키워드 매칭 (가중치 1배)
    3. 부분 매칭도 고려
    
    Returns:
        중분류명 리스트 (예: ["기획", "콘텐츠기획"])
    """
    mid_categories = set()
    
    # 해당 대분류의 중분류 키워드 가져오기
    mid_keywords = job_classification.get(major_category, [])
    
    if not mid_keywords:
        return ["기타"]
    
    title_lower = title.lower()
    tags_text = " ".join(tags).lower()
    
    # 각 중분류 키워드에 대해 점수 계산
    scores = {}
    for mid_kw in mid_keywords:
        mid_kw_lower = mid_kw.lower()
        score = 0
        
        # 제목에서 매칭 (가중치 3배)
        if mid_kw_lower in title_lower:
            score += title_lower.count(mid_kw_lower) * 3
        
        # 태그에서 매칭 (가중치 1배)
        if mid_kw_lower in tags_text:
            score += tags_text.count(mid_kw_lower)
        
        # 부분 매칭 (제목에서만, 2글자 이상)
        if len(mid_kw) >= 2:
            mid_kw_parts = mid_kw.replace("/", " ").replace("·", " ").split()
            for part in mid_kw_parts:
                if len(part) >= 2 and part.lower() in title_lower:
                    score += 1
        
        if score > 0:
            scores[mid_kw] = score
            mid_categories.add(mid_kw)
    
    # 점수가 0인 경우 "기타" 추가
    if not mid_categories:
        mid_categories.add("기타")
    
    # 점수 순으로 정렬하여 반환 (선택사항: 상위 N개만)
    return sorted(list(mid_categories), key=lambda x: scores.get(x, 0), reverse=True)
```

### 5. 통계 및 요약 수정
**중복 통계 포함:**
- 전체 고유 문서 수 (중복 제거)
- 전체 할당 수 (대분류 + 중분류 할당 총합)
- 대분류별 문서 수 (중복 허용)
- 대분류별 고유 문서 수 (중복 제거)
- 중분류별 문서 수 (중복 허용)
- 문서별 대분류 할당 수 분포 (예: 1개 대분류: 1000개, 2개 대분류: 200개, 3개 이상: 50개)
- 문서별 중분류 할당 수 분포 (대분류별)
- 대분류 간 중복 통계 (예: #디자인과 #기획·전략이 함께 나타나는 문서 수)

**통계 출력 예시:**
```
📊 전체 통계:
   - 고유 문서 수: 1,473개
   - 전체 대분류 할당 수: 2,150개 (평균 1.46개/문서)
   - 전체 중분류 할당 수: 3,200개 (평균 2.17개/문서)

📊 대분류 할당 분포:
   - 1개 대분류: 1,200개 문서 (81.5%)
   - 2개 대분류: 200개 문서 (13.6%)
   - 3개 대분류: 60개 문서 (4.1%)
   - 4개 이상: 13개 문서 (0.9%)

📊 대분류 간 중복 (상위 10개):
   - #디자인 + #기획·전략: 45개 문서
   - #마케팅·홍보·조사 + #기획·전략: 38개 문서
   - #it개발·데이터 + #연구·r&d: 32개 문서
   ...

📊 중분류 할당 분포 (대분류별):
   [#디자인]
   - 1개 중분류: 120개 문서
   - 2개 중분류: 25개 문서
   - 3개 이상: 5개 문서
   ...
```

**CSV 요약:**
- `rec_id`, `major_categories` (쉼표로 구분), `mid_categories_by_major` (JSON 형태)
- `other_major_categories` 컬럼 추가

---

## 구현 순서
1. `assign_major_categories` 함수 구현
2. `assign_mid_categories` 함수 구현
3. `aggregate_by_rec_id` 수정
4. `save_tag_based_results` 수정 (중복 통계 포함)
5. 테스트 (rec_id "50496628" 확인)
6. 전체 실행 및 결과 검증

---

## 주의사항
- 중분류 할당은 각 대분류별로 독립적으로 수행
- 같은 문서가 여러 대분류에 나타나므로, 통계 계산 시 주의
- 중분류도 여러 개 할당 가능하므로, 각 대분류 내에서도 중복 가능
- 나중에 군집화를 할 때는 각 대분류별로 독립적으로 수행 가능
- 통계 계산 시 고유 문서 수와 할당 수를 구분하여 표시

