# Ground Truth 생성 전략

## 현재 상황 분석

### 데이터 현황
- **고유 문서 수**: 1,473개
- **전체 대분류 할당 수**: 3,993개 (평균 2.71개/문서)
- **전체 중분류 할당 수**: 7,399개 (평균 5.02개/문서)
- **총 중분류 수**: 450개
- **평균 문서 수/중분류**: 16.4개
- **5개 미만 문서 중분류**: 236개 (52.4%) ← **문제점**

### 주요 특징
1. **다중 할당**: 같은 문서가 여러 대분류/중분류에 할당됨
2. **불균형 분포**: 일부 중분류는 783개 문서, 일부는 1개만
3. **세분화**: 중분류가 매우 세분화되어 있어 직무 단위로 묶을 필요 있음

---

## 전략 1: 중분류 기반 클러스터링 (권장)

### 핵심 아이디어
**중분류 = 직무 그룹**으로 간주하고, 유사한 중분류를 병합하여 의미 있는 클러스터 생성

### 단계별 프로세스

#### Step 1: 중분류 필터링 및 그룹화
```python
# 1. 최소 문서 수 기준 필터링
MIN_DOCS_PER_MID = 5  # 5개 미만 중분류는 제외 또는 병합

# 2. 중분류별 문서 수집
mid_to_docs = defaultdict(list)
for major_cat, mid_cats in doc["category_assignments"].items():
    for mid_cat in mid_cats:
        mid_to_docs[mid_cat].append(doc["rec_id"])

# 3. 문서 수 기준 분류
large_mid_categories = {mid: docs for mid, docs in mid_to_docs.items() 
                       if len(docs) >= MIN_DOCS_PER_MID}
small_mid_categories = {mid: docs for mid, docs in mid_to_docs.items() 
                        if len(docs) < MIN_DOCS_PER_MID}
```

#### Step 2: 유사 중분류 병합 (선택적)
```python
# 예: 건설_건축의 경우
similarity_groups = {
    "품질관리_그룹": ["QA", "QC", "품질관리", "품질보증"],
    "설계_그룹": ["설계엔지니어", "3D설계", "전기설계", "토목설계", "건축설계"],
    "안전_그룹": ["안전관리자", "안전보건관리자", "보건관리자"],
    "환경_그룹": ["대기환경기사", "수질환경기사", "토양환경기사", "환경관리자"],
    "기사_그룹": ["전기기사", "기계기사", "건축기사"],
}

# 병합된 클러스터 생성
merged_clusters = {}
for group_name, mid_list in similarity_groups.items():
    all_docs = set()
    for mid in mid_list:
        if mid in mid_to_docs:
            all_docs.update(mid_to_docs[mid])
    merged_clusters[group_name] = list(all_docs)
```

#### Step 3: 클러스터별 대표 문서 선택
```python
def select_representative_docs(cluster_docs, n=3):
    """
    클러스터에서 대표 문서 선택
    
    전략:
    1. 가장 많은 중분류에 할당된 문서 우선 (다양성)
    2. 여러 대분류에 속한 문서 우선 (범용성)
    3. 태그가 풍부한 문서 우선 (정보성)
    """
    scored_docs = []
    for rec_id in cluster_docs:
        doc = doc_map[rec_id]
        score = (
            len(doc["major_categories"]) * 2 +  # 대분류 다양성
            sum(len(mids) for mids in doc["category_assignments"].values()) +  # 중분류 다양성
            len(doc["filtered_tags"]) * 0.1  # 태그 풍부도
        )
        scored_docs.append((rec_id, score))
    
    # 상위 N개 선택
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [rec_id for rec_id, _ in scored_docs[:n]]
```

#### Step 4: 쿼리 생성
```python
def generate_query_from_cluster(cluster_docs, cluster_name, mid_keywords):
    """
    클러스터 기반 사용자 쿼리 생성
    
    입력:
    - cluster_docs: 클러스터 문서 리스트
    - cluster_name: 클러스터명 (예: "품질관리_그룹")
    - mid_keywords: 중분류 키워드 리스트 (예: ["QA", "QC", "품질관리"])
    
    출력:
    - 사용자 프로필 기반 쿼리
    """
    # LLM 프롬프트 예시
    prompt = f"""
    다음 직무 그룹에 맞는 사용자 프로필을 생성하세요:
    
    직무 그룹: {cluster_name}
    관련 키워드: {', '.join(mid_keywords)}
    관련 문서 수: {len(cluster_docs)}개
    
    생성할 정보:
    1. 경력 (신입/경력/무관)
    2. 관심 직무/키워드
    3. 수강 이력 (관련 강의명)
    4. 기타 프로필 정보
    
    JSON 형식으로 반환:
    {{
        "career_level": "...",
        "job_interests": [...],
        "courses": [...],
        "profile_text": "..."
    }}
    """
    
    # LLM 호출하여 프로필 생성
    # ...
    
    return query_text
```

#### Step 5: GT 매칭
```python
# Ground Truth 생성
gt_dataset = []

for cluster_name, cluster_docs in clusters.items():
    # 대표 문서 선택
    representative_docs = select_representative_docs(cluster_docs, n=3)
    
    # 쿼리 생성
    query = generate_query_from_cluster(
        cluster_docs, 
        cluster_name, 
        mid_keywords=get_mid_keywords(cluster_name)
    )
    
    # GT 매칭
    # 전체 클러스터 문서를 GT로 설정 (중복 허용)
    gt_dataset.append({
        "query": query,
        "query_id": f"gt_{cluster_name}_{len(gt_dataset)}",
        "cluster_name": cluster_name,
        "ground_truth_docs": cluster_docs,  # 전체 클러스터 문서
        "representative_docs": representative_docs,  # 대표 문서
        "num_gt_docs": len(cluster_docs),
    })
```

---

## 전략 2: 대분류 → 중분류 계층적 클러스터링

### 핵심 아이디어
**대분류 내에서 중분류를 기반으로 클러스터 생성**, 이후 유사 클러스터 병합

### 단계별 프로세스

#### Step 1: 대분류별 클러스터 생성
```python
# 각 대분류별로 처리
for major_cat in major_categories:
    # 해당 대분류의 모든 중분류 수집
    mid_clusters = {}
    for doc in major_docs:
        mid_cats = doc["category_assignments"].get(major_cat, [])
        for mid_cat in mid_cats:
            if mid_cat not in mid_clusters:
                mid_clusters[mid_cat] = []
            mid_clusters[mid_cat].append(doc["rec_id"])
    
    # 최소 크기 필터링
    valid_clusters = {
        mid: docs for mid, docs in mid_clusters.items() 
        if len(docs) >= MIN_DOCS_PER_CLUSTER
    }
```

#### Step 2: 유사 중분류 클러스터 병합
```python
# 임베딩 기반 유사도 계산
def merge_similar_clusters(mid_clusters, similarity_threshold=0.7):
    """
    유사한 중분류 클러스터 병합
    
    방법:
    1. 중분류명 유사도 (문자열 매칭)
    2. 문서 겹침 비율 (Jaccard similarity)
    3. 태그 유사도 (임베딩 기반)
    """
    merged = {}
    processed = set()
    
    for mid1, docs1 in mid_clusters.items():
        if mid1 in processed:
            continue
        
        cluster_group = [mid1]
        cluster_docs = set(docs1)
        
        for mid2, docs2 in mid_clusters.items():
            if mid2 in processed or mid1 == mid2:
                continue
            
            # 문서 겹침 비율 계산
            overlap = len(set(docs1) & set(docs2))
            jaccard = overlap / len(set(docs1) | set(docs2))
            
            if jaccard > similarity_threshold:
                cluster_group.append(mid2)
                cluster_docs.update(docs2)
                processed.add(mid2)
        
        merged[f"{'+'.join(cluster_group)}"] = list(cluster_docs)
        processed.add(mid1)
    
    return merged
```

---

## 전략 3: 하이브리드 접근 (최종 권장)

### 핵심 아이디어
**중분류 기반 + 임베딩 기반 + 규칙 기반**을 결합한 다단계 클러스터링

### 전체 파이프라인

```python
def generate_gt_dataset(doc_map, output_path):
    """
    Ground Truth 데이터셋 생성 파이프라인
    """
    
    # ============================================================
    # Phase 1: 중분류 기반 초기 클러스터 생성
    # ============================================================
    print("Phase 1: 중분류 기반 클러스터 생성...")
    
    # 1-1. 중분류별 문서 수집 (대분류 무시)
    mid_to_docs = defaultdict(set)
    mid_to_major = defaultdict(set)  # 중분류가 속한 대분류들
    
    for doc in doc_map.values():
        for major_cat, mid_cats in doc["category_assignments"].items():
            for mid_cat in mid_cats:
                mid_to_docs[mid_cat].add(doc["rec_id"])
                mid_to_major[mid_cat].add(major_cat)
    
    # 1-2. 최소 문서 수 필터링
    MIN_DOCS = 5
    valid_mid_clusters = {
        mid: list(docs) for mid, docs in mid_to_docs.items()
        if len(docs) >= MIN_DOCS
    }
    
    print(f"   ✅ {len(valid_mid_clusters)}개 유효 중분류 클러스터 생성")
    
    # ============================================================
    # Phase 2: 유사 중분류 병합 (규칙 기반)
    # ============================================================
    print("\nPhase 2: 유사 중분류 병합...")
    
    # 2-1. 키워드 기반 유사도 그룹 정의
    similarity_rules = load_similarity_rules()  # JSON 파일에서 로드
    
    # 예시:
    # {
    #   "품질관리": ["QA", "QC", "품질관리", "품질보증", "품질분석"],
    #   "설계": ["설계엔지니어", "3D설계", "전기설계", "토목설계", "건축설계"],
    #   ...
    # }
    
    merged_clusters = merge_by_rules(valid_mid_clusters, similarity_rules)
    print(f"   ✅ {len(merged_clusters)}개 병합 클러스터 생성")
    
    # ============================================================
    # Phase 3: 임베딩 기반 세밀한 클러스터링 (선택적)
    # ============================================================
    print("\nPhase 3: 임베딩 기반 세밀한 클러스터링...")
    
    # 3-1. 큰 클러스터(20개 이상)는 임베딩으로 세분화
    refined_clusters = {}
    for cluster_name, cluster_docs in merged_clusters.items():
        if len(cluster_docs) >= 20:
            # 임베딩 생성 및 HDBSCAN 클러스터링
            sub_clusters = refine_with_embedding(cluster_docs, doc_map)
            for i, sub_cluster in enumerate(sub_clusters):
                refined_clusters[f"{cluster_name}_sub{i+1}"] = sub_cluster
        else:
            refined_clusters[cluster_name] = cluster_docs
    
    print(f"   ✅ {len(refined_clusters)}개 최종 클러스터 생성")
    
    # ============================================================
    # Phase 4: 대표 문서 선택 및 쿼리 생성
    # ============================================================
    print("\nPhase 4: 대표 문서 선택 및 쿼리 생성...")
    
    gt_dataset = []
    
    for cluster_name, cluster_docs in refined_clusters.items():
        # 4-1. 대표 문서 선택
        representative_docs = select_representative_docs(
            cluster_docs, 
            doc_map, 
            n=3
        )
        
        # 4-2. 클러스터 키워드 추출
        cluster_keywords = extract_cluster_keywords(
            cluster_docs, 
            doc_map
        )
        
        # 4-3. LLM 기반 쿼리 생성
        query = generate_query_with_llm(
            cluster_name=cluster_name,
            keywords=cluster_keywords,
            sample_docs=representative_docs,
            doc_map=doc_map
        )
        
        # 4-4. GT 데이터 생성
        gt_dataset.append({
            "query_id": f"gt_{len(gt_dataset)+1:04d}",
            "query": query["profile_text"],
            "query_metadata": {
                "career_level": query.get("career_level", "경력 무관"),
                "job_interests": query.get("job_interests", []),
                "courses": query.get("courses", []),
            },
            "cluster_info": {
                "cluster_name": cluster_name,
                "cluster_keywords": cluster_keywords,
                "num_docs": len(cluster_docs),
            },
            "ground_truth": {
                "rec_ids": cluster_docs,  # 전체 클러스터 문서
                "representative_rec_ids": representative_docs,  # 대표 문서
            },
        })
    
    print(f"   ✅ {len(gt_dataset)}개 GT 쿼리 생성")
    
    # ============================================================
    # Phase 5: 결과 저장 및 검증
    # ============================================================
    print("\nPhase 5: 결과 저장 및 검증...")
    
    # 5-1. JSON 저장
    with open(output_path / "gt_dataset_v4.json", "w", encoding="utf-8") as f:
        json.dump(gt_dataset, f, ensure_ascii=False, indent=2)
    
    # 5-2. CSV 저장 (검색 파이프라인용)
    df = pd.DataFrame([
        {
            "query_id": item["query_id"],
            "query": item["query"],
            "ground_truth_rec_ids": ",".join(item["ground_truth"]["rec_ids"]),
            "num_gt_docs": len(item["ground_truth"]["rec_ids"]),
            "cluster_name": item["cluster_info"]["cluster_name"],
        }
        for item in gt_dataset
    ])
    df.to_csv(output_path / "gt_dataset_v4.csv", index=False, encoding="utf-8-sig")
    
    # 5-3. 통계 생성
    generate_gt_statistics(gt_dataset, output_path)
    
    print(f"   ✅ 결과 저장 완료: {output_path}")
    
    return gt_dataset
```

---

## 구체적 구현 제안

### 1. 유사도 규칙 파일 생성
```json
// similarity_rules.json
{
  "#건설·건축": {
    "품질관리": ["QA", "QC", "품질관리", "품질보증", "품질분석"],
    "설계": ["설계엔지니어", "3D설계", "전기설계", "토목설계", "건축설계"],
    "안전": ["안전관리자", "안전보건관리자", "보건관리자"],
    "환경": ["대기환경기사", "수질환경기사", "토양환경기사", "환경관리자"],
    "기사": ["전기기사", "기계기사", "건축기사"]
  },
  "#it개발·데이터": {
    "백엔드": ["백엔드/서버개발", "서버개발", "백엔드개발"],
    "프론트엔드": ["프론트엔드", "웹개발", "앱개발"],
    "데이터": ["데이터엔지니어", "데이터 사이언티스트", "BI 엔지니어"],
    "QA": ["QA/테스터", "QA", "테스터"]
  },
  ...
}
```

### 2. 대표 문서 선택 전략
```python
def select_representative_docs(cluster_docs, doc_map, n=3):
    """
    클러스터에서 대표 문서 선택
    
    점수 계산:
    - 대분류 다양성: +2점/대분류
    - 중분류 다양성: +1점/중분류
    - 태그 풍부도: +0.1점/태그
    - 다른 클러스터와의 겹침: +1점 (범용성)
    """
    scored = []
    
    for rec_id in cluster_docs:
        doc = doc_map[rec_id]
        
        score = (
            len(doc["major_categories"]) * 2 +
            sum(len(mids) for mids in doc["category_assignments"].values()) +
            len(doc["filtered_tags"]) * 0.1
        )
        
        scored.append((rec_id, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [rec_id for rec_id, _ in scored[:n]]
```

### 3. LLM 쿼리 생성 프롬프트
```python
def generate_query_with_llm(cluster_name, keywords, sample_docs, doc_map):
    """
    LLM을 사용한 사용자 프로필 쿼리 생성
    """
    # 샘플 문서 제목 추출
    sample_titles = [doc_map[rec_id]["title"] for rec_id in sample_docs[:3]]
    
    prompt = f"""
    다음 직무 그룹에 맞는 구직자 프로필을 생성하세요:
    
    직무 그룹: {cluster_name}
    관련 키워드: {', '.join(keywords[:10])}
    관련 채용공고 예시:
    {chr(10).join(f'- {title}' for title in sample_titles)}
    
    생성할 프로필:
    1. 경력 수준 (신입/1-3년/3-5년/5년 이상/무관)
    2. 관심 직무 및 키워드 (3-5개)
    3. 수강 이력 (관련 강의명 3-5개)
    4. 기타 프로필 정보
    
    JSON 형식:
    {{
        "career_level": "...",
        "job_interests": ["...", "..."],
        "courses": ["...", "..."],
        "profile_text": "전체 프로필 텍스트 (수강 이력 포함)"
    }}
    """
    
    # LLM 호출 (OpenAI, Claude 등)
    response = llm_client.generate(prompt)
    return json.loads(response)
```

---

## 예상 결과

### 클러스터 예시
```
품질관리_그룹 (건설_건축):
  - 문서 수: 45개 (QA: 28개 + QC: 27개 - 중복 제거)
  - 대표 문서: 3개
  - 생성 쿼리: "3년차 품질관리 경력, QA/QC 업무, 품질보증 관련 강의 수강"
  - GT 문서: 45개 rec_id
```

### GT 데이터셋 구조
```json
{
  "query_id": "gt_0001",
  "query": "3년차 품질관리 경력자...",
  "ground_truth": {
    "rec_ids": ["50055702", "50210446", ...],
    "num_docs": 45
  },
  "cluster_info": {
    "cluster_name": "품질관리_그룹",
    "major_category": "#건설·건축",
    "keywords": ["QA", "QC", "품질관리"]
  }
}
```

---

## 다음 단계

1. **유사도 규칙 파일 생성**: 각 대분류별 유사 중분류 그룹 정의
2. **대표 문서 선택 로직 구현**: 점수 기반 선택 알고리즘
3. **LLM 쿼리 생성 파이프라인**: 벡엔드 API 연동 또는 직접 호출
4. **GT 데이터셋 검증**: 클러스터 품질 확인 및 수동 검토
5. **검색 파이프라인 테스트**: 생성된 GT로 검색 성능 평가

---

## 주의사항

1. **중복 허용**: 같은 문서가 여러 클러스터에 나타날 수 있음 (의도된 동작)
2. **최소 클러스터 크기**: 5개 이상 문서를 가진 클러스터만 사용
3. **쿼리 다양성**: 같은 클러스터에서도 다양한 쿼리 생성 (경력 수준, 관심사 등)
4. **검증 필요**: 생성된 GT는 수동 검토를 통해 품질 확인



