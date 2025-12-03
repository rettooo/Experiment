"""
계층적 채용공고 클러스터링 파이프라인 (태그 기반)
21개 대분류 → 중분류 태그 기반 분류

파이프라인:
1. 전처리: 회사명 + 노이즈 제거
2. 대분류 할당: 태그 매칭
3. 중분류 할당: 중분류 키워드 매칭 (태그 기반)
4. 결과 저장 및 통계

[향후 옵션]
- 임베딩 기반 클러스터링 (옵션)
- DBSCAN 클러스터링 (옵션)
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Tuple, Set
import warnings

warnings.filterwarnings("ignore")

# 필요한 라이브러리
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 1. JOB CLASSIFICATION 로드
# ============================================================================


def load_job_classification(
    json_path: str = "job_classfication.json",
) -> Dict[str, List[str]]:
    """21개 대분류 및 중분류 키워드 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


JOB_CLASSIFICATION = load_job_classification()
MAJOR_CATEGORIES = list(JOB_CLASSIFICATION.keys())

print(f"✅ 대분류 {len(MAJOR_CATEGORIES)}개 로드 완료")


# ============================================================================
# 2. 노이즈 제거 설정
# ============================================================================

NOISE_WORDS = {
    # 채용 관련
    "채용",
    "공채",
    "수시",
    "정규직",
    "신입",
    "경력",
    "모집",
    "인턴",
    "사원",
    "직원",
    "담당자",
    "팀원",
    "인재",
    # 시기 관련
    "상반기",
    "하반기",
    "1분기",
    "2분기",
    "3분기",
    "4분기",
    # 년도
    "2024",
    "2025",
    "24년",
    "25년",
    "'24",
    "'25",
    # 기타
    "대졸",
    "공고",
    "안내",
    "및",
}

DATE_PATTERNS = [
    r"\d{4}년?",  # 2024년, 2024
    r"'?\d{2}년?",  # '24년, 24년
    r"\d{1,2}분기",  # 1분기, 2분기
    r"(상|하)반기",  # 상반기, 하반기
]

# 지역 태그 정의
LOCATION_TAGS = {
    # 광역시/도
    "#서울",
    "#경기",
    "#인천",
    "#부산",
    "#대구",
    "#광주",
    "#대전",
    "#울산",
    "#세종",
    "#세종특별자치시",
    "#강원",
    "#충북",
    "#충남",
    "#전북",
    "#전남",
    "#경북",
    "#경남",
    "#제주",
    # 서울 구
    "#강남구",
    "#강동구",
    "#강북구",
    "#강서구",
    "#관악구",
    "#광진구",
    "#구로구",
    "#금천구",
    "#노원구",
    "#도봉구",
    "#동대문구",
    "#동작구",
    "#마포구",
    "#서대문구",
    "#서초구",
    "#성동구",
    "#성북구",
    "#송파구",
    "#양천구",
    "#영등포구",
    "#용산구",
    "#은평구",
    "#종로구",
    "#중구",
    "#중랑구",
    # 경기 시/군
    "#화성시",
    "#성남시",
    "#안양시",
    "#평택시",
    "#용인시",
    "#과천시",
    "#안산시",
    "#시흥시",
    "#안성시",
    "#광명시",
    "#이천시",
    "#수원시",
    "#고양시",
    "#남양주시",
    "#부천시",
    "#의정부시",
    "#파주시",
    "#김포시",
    "#광주시",
    "#하남시",
    "#오산시",
    "#양주시",
    "#포천시",
    "#여주시",
    "#연천군",
    "#가평군",
    "#양평군",
    "#동두천시",
    # 부산 구/군
    "#해운대구",
    "#사하구",
    "#동래구",
    "#부산진구",
    "#남구",
    "#북구",
    "#수영구",
    "#사상구",
    "#금정구",
    "#연제구",
    "#동구",
    "#영도구",
    "#기장군",
    # 대구 구/군
    "#달서구",
    "#수성구",
    "#달성군",
    # 인천 구/군
    "#연수구",
    "#남동구",
    "#부평구",
    "#계양구",
    "#미추홀구",
    "#강화군",
    "#옹진군",
    # 광주 구
    "#광산구",
    # 대전 구
    "#유성구",
    "#대덕구",
    # 울산 구/군
    "#울주군",
    # 충남 시/군
    "#천안시",
    "#아산시",
    "#당진시",
    "#서산시",
    "#논산시",
    "#계룡시",
    "#공주시",
    "#보령시",
    "#금산군",
    "#부여군",
    "#서천군",
    "#청양군",
    "#홍성군",
    "#예산군",
    "#태안군",
    # 충북 시/군
    "#청주시",
    "#충주시",
    "#제천시",
    "#음성군",
    "#진천군",
    "#괴산군",
    "#증평군",
    "#영동군",
    "#옥천군",
    "#보은군",
    "#단양군",
    # 전북 시/군
    "#전주시",
    "#익산시",
    "#군산시",
    "#정읍시",
    "#남원시",
    "#김제시",
    "#완주군",
    "#고창군",
    "#부안군",
    "#임실군",
    "#순창군",
    "#진안군",
    "#장수군",
    "#무주군",
    # 전남 시/군
    "#여수시",
    "#순천시",
    "#목포시",
    "#광양시",
    "#나주시",
    "#무안군",
    "#영암군",
    "#담양군",
    "#곡성군",
    "#구례군",
    "#고흥군",
    "#보성군",
    "#화순군",
    "#장흥군",
    "#강진군",
    "#해남군",
    "#영광군",
    "#함평군",
    "#진도군",
    "#완도군",
    "#장성군",
    "#신안군",
    # 경북 시/군
    "#포항시",
    "#경산시",
    "#구미시",
    "#김천시",
    "#안동시",
    "#영주시",
    "#영천시",
    "#상주시",
    "#문경시",
    "#경주시",
    "#칠곡군",
    "#예천군",
    "#봉화군",
    "#울진군",
    "#울릉군",
    "#의성군",
    "#청송군",
    "#영양군",
    "#영덕군",
    "#청도군",
    "#고령군",
    "#성주군",
    "#군위군",
    # 경남 시/군
    "#창원시",
    "#김해시",
    "#진주시",
    "#양산시",
    "#거제시",
    "#통영시",
    "#사천시",
    "#밀양시",
    "#함안군",
    "#창녕군",
    "#고성군",
    "#하동군",
    "#산청군",
    "#함양군",
    "#거창군",
    "#합천군",
    "#의령군",
    "#남해군",
    # 강원 시/군
    "#원주시",
    "#춘천시",
    "#강릉시",
    "#동해시",
    "#속초시",
    "#삼척시",
    "#태백시",
    "#홍천군",
    "#횡성군",
    "#영월군",
    "#평창군",
    "#정선군",
    "#철원군",
    "#화천군",
    "#양구군",
    "#인제군",
    "#고성군",
    "#양양군",
    # 제주
    "#제주시",
    "#서귀포시",
}


# ============================================================================
# 3. 전처리 함수
# ============================================================================


def clean_text_advanced(text: str, company: str, noise_words: set = NOISE_WORDS) -> str:
    """
    회사명 + 노이즈 단어 + 날짜 제거 (강화 버전)

    Args:
        text: 정제할 텍스트 (title 또는 tag)
        company: 회사명 (metadata에서 가져옴)
        noise_words: 제거할 노이즈 단어 set

    Returns:
        정제된 텍스트
    """

    cleaned = text

    # 1. 회사명 제거 (정확히 매칭 - 대소문자 무시)
    if company and company != "미상" and len(company) > 2:
        # 정확한 회사명 제거
        cleaned = re.sub(re.escape(company), "", cleaned, flags=re.IGNORECASE)

        # 회사명 단어들 개별 제거 (띄어쓰기로 분리)
        company_words = company.split()
        for word in company_words:
            if len(word) > 2:  # 2글자 이상만
                cleaned = re.sub(
                    r"\b" + re.escape(word) + r"\b", "", cleaned, flags=re.IGNORECASE
                )

    # 2. [회사명] 또는 (회사명) 패턴 제거
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = re.sub(r"\([^\)]*회사[^\)]*\)", "", cleaned)
    cleaned = re.sub(r"\([^\)]*주식회사[^\)]*\)", "", cleaned)
    cleaned = re.sub(r"\([^\)]*그룹[^\)]*\)", "", cleaned)

    # 3. 회사명_년도 패턴 제거
    cleaned = re.sub(r"^[가-힣a-zA-Z\s&\(\)]+_?\d{2,4}년?\s*", "", cleaned)

    # 4. 날짜 정규식 제거
    for pattern in DATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # 5. 노이즈 단어 제거
    for noise in noise_words:
        cleaned = cleaned.replace(noise, "")

    # 6. 회사 관련 단어 제거
    company_patterns = [
        r"주식회사",
        r"\(주\)",
        r"㈜",
        r"그룹사",
        r"계열사",
        r"본사",
    ]
    for pattern in company_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # 7. 특수문자 및 공백 정리
    cleaned = re.sub(r"[_\-/\\|]+", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip()

    return cleaned


def clean_tags(
    tags: List[str], company: str, noise_words: set = NOISE_WORDS
) -> List[str]:
    """
    태그 리스트 정제 (강화 버전)

    - 회사명 포함 태그는 제거
    - 노이즈 단어 포함 태그는 제거
    - 남은 태그에서도 회사명/노이즈 제거
    """
    cleaned_tags = []

    for tag in tags:
        tag_lower = tag.lower()

        # 회사명 포함 태그는 완전히 제거
        if company and company != "미상" and len(company) > 2:
            if company.lower() in tag_lower:
                continue

        # 노이즈 단어 포함 태그도 제거
        if any(noise in tag_lower for noise in noise_words):
            continue

        # 태그 내용에서 회사명 제거 (혹시 남아있는 경우)
        cleaned_tag = tag
        if company and company != "미상" and len(company) > 2:
            cleaned_tag = re.sub(
                re.escape(company), "", cleaned_tag, flags=re.IGNORECASE
            )
            cleaned_tag = cleaned_tag.strip()

        # 빈 태그가 아니면 추가
        if cleaned_tag and len(cleaned_tag) > 1:
            cleaned_tags.append(cleaned_tag)

    return cleaned_tags


def filter_location_tags(tags: List[str]) -> List[str]:
    """지역 태그 제거"""
    return [tag for tag in tags if tag not in LOCATION_TAGS]


# ============================================================================
# 4. 대분류 할당
# ============================================================================


def assign_major_categories(
    tags: List[str],
    title: str,
    job_classification: Dict[str, List[str]] = JOB_CLASSIFICATION,
) -> List[str]:
    """
    여러 대분류 카테고리 할당

    우선순위:
    1. tags에서 대분류 태그 직접 매칭 (예: "#영업·판매·무역")
    2. title + tags에서 중분류 키워드 매칭 (제목 우선 가중치) → 점수가 0보다 크면 추가
    3. 제목에서 직접 대분류 키워드 매칭
    4. 최소 1개는 할당 (없으면 "기타")

    Returns:
        대분류명 리스트 (예: ["#디자인", "#기획·전략", "#마케팅·홍보·조사"])
    """
    major_categories = set()

    # 1. tags에서 대분류 태그 직접 매칭
    for tag in tags:
        tag_lower = tag.lower()
        for major_cat in job_classification.keys():
            if tag_lower == major_cat.lower():
                major_categories.add(major_cat)

    # 2. 제목 + 태그에서 중분류 키워드 매칭 (제목에 가중치 부여)
    title_lower = title.lower()
    tags_text = " ".join(tags).lower()

    # 각 대분류별 매칭 점수 계산
    scores = {}
    for major_cat, mid_keywords_list in job_classification.items():
        score = 0
        for keyword in mid_keywords_list:
            keyword_lower = keyword.lower()

            # 제목에서 매칭 (가중치 2배)
            if keyword_lower in title_lower:
                score += 2

            # 태그에서 매칭 (가중치 1배)
            if keyword_lower in tags_text:
                score += 1

        if score > 0:
            scores[major_cat] = score
            major_categories.add(major_cat)

    # 3. 제목에서 직접 대분류 키워드 매칭
    # 대분류별 대표 키워드 매핑
    major_keywords = {
        "#it개발·데이터": [
            "소프트웨어",
            "sw",
            "s/w",
            "개발자",
            "프로그래밍",
            "코딩",
            "엔지니어",
            "개발",
            "데이터",
            "ai",
            "머신러닝",
        ],
        "#생산": ["품질", "qa", "qc", "생산", "제조", "공정", "설계", "기계"],
        "#연구·r&d": ["연구", "r&d", "연구개발", "연구원", "개발"],
        "#영업·판매·무역": ["영업", "판매", "무역", "mr", "영업직"],
        "#디자인": ["디자인", "디자이너", "ui", "ux", "그래픽"],
        "#마케팅·홍보·조사": ["마케팅", "홍보", "광고", "브랜드"],
        "#회계·세무·재무": ["회계", "재무", "세무", "경리"],
        "#인사·노무·hrd": ["인사", "hr", "채용", "노무"],
        "#총무·법무·사무": ["총무", "법무", "사무", "행정"],
        "#구매·자재·물류": ["구매", "물류", "자재", "scm"],
        "#기획·전략": ["기획", "전략", "기획자", "pm", "po"],
        "#교육": ["교육", "강사", "교사", "학원"],
        "#의료": ["의료", "간호", "의사", "병원"],
        "#건설·건축": ["건설", "건축", "토목", "공사"],
    }

    # 제목에서 직접 대분류 키워드 찾기
    for major_cat, keywords in major_keywords.items():
        for keyword in keywords:
            if keyword.lower() in title_lower:
                # 기존 점수에 추가 (제목 직접 매칭은 높은 가중치)
                scores[major_cat] = scores.get(major_cat, 0) + 3
                major_categories.add(major_cat)

    # 최소 1개는 할당
    if not major_categories:
        major_categories.add("기타")

    return sorted(list(major_categories))


# ============================================================================
# 5. 데이터 로드 및 집계
# ============================================================================


def load_chunks(chunk_file: str) -> List[Dict]:
    """청크 파일 로드"""
    print(f"📂 청크 파일 로드: {chunk_file}")
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   ✅ {len(chunks):,}개 청크 로드 완료")
    return chunks


def aggregate_by_rec_id(chunks: List[Dict]) -> Dict[str, Dict]:
    """
    rec_id별로 집계 + 전처리 + 다중 대분류/중분류 할당

    Returns:
        {rec_id: {
            'title': str,
            'cleaned_title': str,
            'tags': List[str],
            'filtered_tags': List[str],
            'cleaned_tags': List[str],
            'company': str,
            'url': str,
            'major_categories': List[str],
            'category_assignments': Dict[str, List[str]]  # {major_cat: [mid_cats]}
        }}
    """
    print("\n📊 rec_id별 데이터 집계 및 전처리 중...")

    doc_map = {}
    for chunk in tqdm(chunks, desc="집계 진행"):
        metadata = chunk["metadata"]
        rec_id = metadata["rec_id"]

        if rec_id not in doc_map:
            tags = metadata.get("tags", [])
            company = metadata.get("company", "미상")

            # 지역 태그 필터링
            filtered_tags = filter_location_tags(tags)

            # 전처리
            title = metadata.get("title", "")
            cleaned_title = clean_text_advanced(title, company)
            cleaned_tags = clean_tags(filtered_tags, company)

            # 여러 대분류 할당
            major_categories = assign_major_categories(filtered_tags, title)

            # 각 대분류별로 중분류 할당
            category_assignments = {}
            for major_cat in major_categories:
                mid_categories = assign_mid_categories(filtered_tags, title, major_cat)
                category_assignments[major_cat] = mid_categories

            doc_map[rec_id] = {
                "rec_id": rec_id,
                "title": title,
                "cleaned_title": cleaned_title,
                "company": company,
                "url": metadata.get("url", ""),
                "tags": tags,
                "filtered_tags": filtered_tags,
                "cleaned_tags": cleaned_tags,
                "major_categories": major_categories,
                "category_assignments": category_assignments,
            }

    print(f"   ✅ {len(doc_map):,}개 문서 집계 완료")

    # 통계
    total_tags = sum(len(doc["tags"]) for doc in doc_map.values())
    filtered_tags_count = sum(len(doc["filtered_tags"]) for doc in doc_map.values())
    cleaned_tags_count = sum(len(doc["cleaned_tags"]) for doc in doc_map.values())

    print(f"   📊 태그 통계:")
    print(f"      - 전체 태그: {total_tags:,}개")
    print(f"      - 지역 필터링 후: {filtered_tags_count:,}개")
    print(f"      - 노이즈 제거 후: {cleaned_tags_count:,}개")

    # 대분류 통계 (중복 포함)
    all_major_categories = []
    for doc in doc_map.values():
        all_major_categories.extend(doc["major_categories"])

    category_counts = Counter(all_major_categories)
    unique_docs = len(doc_map)
    total_assignments = len(all_major_categories)

    print(f"\n   📊 대분류 통계:")
    print(f"      - 고유 문서 수: {unique_docs:,}개")
    print(
        f"      - 전체 대분류 할당 수: {total_assignments:,}개 (평균 {total_assignments/unique_docs:.2f}개/문서)"
    )
    print(f"      - 대분류 분포 (상위 10개):")
    for cat, count in category_counts.most_common(10):
        print(f"         - {cat}: {count}개")

    # 대분류 할당 수 분포
    major_count_dist = Counter(len(doc["major_categories"]) for doc in doc_map.values())
    print(f"\n   📊 대분류 할당 분포:")
    for count, num_docs in sorted(major_count_dist.items()):
        print(
            f"      - {count}개 대분류: {num_docs}개 문서 ({num_docs/unique_docs*100:.1f}%)"
        )

    return doc_map


# ============================================================================
# 6. 중분류 할당 (태그 기반)
# ============================================================================


def assign_mid_categories(
    tags: List[str],
    title: str,
    major_category: str,
    job_classification: Dict[str, List[str]] = JOB_CLASSIFICATION,
) -> List[str]:
    """
    여러 중분류 카테고리 할당

    Args:
        tags: 문서의 태그 리스트
        title: 문서 제목
        major_category: 이미 할당된 대분류
        job_classification: 대분류별 중분류 키워드 딕셔너리

    Returns:
        중분류명 리스트 (예: ["기획", "콘텐츠기획"])
    """

    # 해당 대분류의 중분류 키워드 가져오기
    mid_keywords = job_classification.get(major_category, [])

    if not mid_keywords:
        return ["기타"]

    # 제목과 태그 분리 (제목에 더 높은 가중치)
    title_lower = title.lower()
    tags_text = " ".join(tags).lower()

    # 중분류 키워드 매칭 점수 계산 (제목 우선 가중치)
    scores = {}
    mid_categories = set()

    for mid_kw in mid_keywords:
        mid_kw_lower = mid_kw.lower()
        score = 0

        # 제목에서 매칭 (가중치 3배)
        if mid_kw_lower in title_lower:
            score += title_lower.count(mid_kw_lower) * 3

        # 태그에서 매칭 (가중치 1배)
        if mid_kw_lower in tags_text:
            score += tags_text.count(mid_kw_lower)

        # 부분 매칭도 고려 (제목에서만, 2글자 이상 키워드)
        if len(mid_kw) >= 2:
            # 키워드의 주요 부분이 제목에 포함되는지
            # 예: "백엔드/서버개발" → "백엔드", "서버", "개발" 각각 체크
            mid_kw_parts = mid_kw.replace("/", " ").replace("·", " ").split()
            for part in mid_kw_parts:
                if len(part) >= 2 and part.lower() in title_lower:
                    score += 1  # 부분 매칭은 낮은 가중치

        if score > 0:
            scores[mid_kw] = score
            mid_categories.add(mid_kw)

    # 점수가 0인 경우 "기타" 추가
    if not mid_categories:
        mid_categories.add("기타")

    # 점수 순으로 정렬하여 반환
    return sorted(list(mid_categories), key=lambda x: scores.get(x, 0), reverse=True)


# ============================================================================
# 7. 임베딩 생성 (옵션 - 향후 사용)
# ============================================================================


def embed_documents(
    doc_map: Dict[str, Dict], model_name: str = "jhgan/ko-sroberta-multitask"
) -> Tuple[np.ndarray, List[str]]:
    """
    정제된 태그 + 제목 임베딩 생성

    형식: "[정제된 제목] {정제된 태그1} {정제된 태그2}"

    Returns:
        embeddings: (N, 768) numpy array
        rec_ids: List of rec_id in same order
    """
    print(f"\n🔤 임베딩 생성 중... (모델: {model_name})")

    # 모델 로드
    print("   📥 모델 로드 중...")
    model = SentenceTransformer(model_name)
    print("   ✅ 모델 로드 완료")

    # 텍스트 생성
    texts = []
    rec_ids = []

    for rec_id, doc in tqdm(doc_map.items(), desc="텍스트 생성"):
        job_title = doc["cleaned_title"]
        tags_str = " ".join(doc["cleaned_tags"])
        text = f"[{job_title}] {tags_str}".strip()

        texts.append(text)
        rec_ids.append(rec_id)

    # 임베딩
    print(f"   🔄 {len(texts):,}개 텍스트 임베딩 중...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    print(f"   ✅ 임베딩 완료: {embeddings.shape}")
    return embeddings, rec_ids


# ============================================================================
# 7. UMAP 차원 축소
# ============================================================================


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 15,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, umap.UMAP]:
    """UMAP 차원 축소"""
    print(f"\n🗜️  UMAP 차원 축소 중...")
    print(
        f"   파라미터: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}"
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"   ✅ 축소 완료: {reduced_embeddings.shape}")

    return reduced_embeddings, reducer


# ============================================================================
# 8. HDBSCAN 클러스터링
# ============================================================================


def cluster_documents(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.01,
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """HDBSCAN 클러스터링"""
    print(f"\n🔍 HDBSCAN 클러스터링 중...")
    print(
        f"   파라미터: min_cluster_size={min_cluster_size}, min_samples={min_samples}"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(reduced_embeddings)

    # 통계
    unique_labels = set(labels)
    num_clusters = len([l for l in unique_labels if l != -1])
    num_noise = list(labels).count(-1)

    cluster_sizes = [list(labels).count(i) for i in unique_labels if i != -1]
    min_size = min(cluster_sizes) if cluster_sizes else 0
    max_size = max(cluster_sizes) if cluster_sizes else 0
    avg_size = np.mean(cluster_sizes) if cluster_sizes else 0

    print(f"   ✅ 클러스터링 완료")
    print(f"   📊 결과:")
    print(f"      - 클러스터 수: {num_clusters}개")
    print(f"      - 노이즈: {num_noise}개 ({num_noise/len(labels)*100:.1f}%)")
    print(f"      - 최소 크기: {min_size}개")
    print(f"      - 최대 크기: {max_size}개")
    print(f"      - 평균 크기: {avg_size:.1f}개")

    return labels, clusterer


# ============================================================================
# 9. TF-IDF 기반 클러스터 정제
# ============================================================================


def refine_clusters_with_tfidf(
    doc_map: Dict[str, Dict],
    rec_ids: List[str],
    labels: np.ndarray,
    top_n_keywords: int = 10,
    min_docs_per_cluster: int = 5,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """TF-IDF 기반 클러스터 정제"""
    print(f"\n🔧 TF-IDF 기반 클러스터 정제 중...")

    # 클러스터별 문서 그룹화
    cluster_docs = defaultdict(list)
    for rec_id, label in zip(rec_ids, labels):
        if label != -1:
            cluster_docs[label].append(rec_id)

    print(f"   📊 초기 클러스터: {len(cluster_docs)}개")

    # TF-IDF로 클러스터별 키워드 추출
    cluster_keywords = {}
    refined_clusters = {}

    for cluster_id, cluster_rec_ids in tqdm(cluster_docs.items(), desc="클러스터 정제"):
        # 클러스터 텍스트
        cluster_texts = []
        for rec_id in cluster_rec_ids:
            doc = doc_map[rec_id]
            text = f"{doc['cleaned_title']} {' '.join(doc['cleaned_tags'])}"
            cluster_texts.append(text)

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()

            # 상위 키워드
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-top_n_keywords:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]

            # 키워드 포함 여부 체크
            valid_rec_ids = []
            for rec_id, text in zip(cluster_rec_ids, cluster_texts):
                if any(kw in text for kw in top_keywords):
                    valid_rec_ids.append(rec_id)

            # 정제된 클러스터 저장
            if len(valid_rec_ids) >= min_docs_per_cluster:
                refined_clusters[cluster_id] = valid_rec_ids
                cluster_keywords[cluster_id] = top_keywords

        except Exception as e:
            # 에러 발생 시 원본 유지
            if len(cluster_rec_ids) >= min_docs_per_cluster:
                refined_clusters[cluster_id] = cluster_rec_ids
                cluster_keywords[cluster_id] = []

    print(f"   ✅ 정제 완료")
    print(f"   📊 최종 클러스터: {len(refined_clusters)}개")

    total_before = sum(len(docs) for docs in cluster_docs.values())
    total_after = sum(len(docs) for docs in refined_clusters.values())
    removed = total_before - total_after

    print(f"   📊 문서 통계:")
    print(f"      - 정제 전: {total_before:,}개")
    print(f"      - 정제 후: {total_after:,}개")
    print(f"      - 제거됨: {removed:,}개 ({removed/total_before*100:.1f}%)")

    return refined_clusters, cluster_keywords


# ============================================================================
# 10. 중분류 키워드 매칭 추가
# ============================================================================


def enhance_keywords_with_mid_classification(
    refined_clusters: Dict[int, List[str]],
    cluster_keywords: Dict[int, List[str]],
    doc_map: Dict[str, Dict],
    mid_keywords: List[str],
) -> Dict[int, List[str]]:
    """
    TF-IDF 키워드에 중분류 키워드 추가

    Args:
        refined_clusters: 정제된 클러스터
        cluster_keywords: TF-IDF 키워드
        doc_map: 문서 맵
        mid_keywords: 해당 대분류의 중분류 키워드 리스트

    Returns:
        enhanced_keywords: 강화된 키워드
    """

    enhanced_keywords = {}

    for cluster_id, rec_ids in refined_clusters.items():
        # 기존 TF-IDF 키워드
        keywords_set = set(cluster_keywords.get(cluster_id, []))

        # 클러스터 텍스트 결합
        cluster_texts = []
        for rec_id in rec_ids:
            doc = doc_map[rec_id]
            cluster_texts.append(
                f"{doc['cleaned_title']} {' '.join(doc['cleaned_tags'])}"
            )

        combined_text = " ".join(cluster_texts).lower()

        # 중분류 키워드 매칭
        mid_keywords_in_cluster = []
        for mid_kw in mid_keywords:
            count = combined_text.count(mid_kw.lower())
            if count > 0:
                mid_keywords_in_cluster.append((mid_kw, count))

        # 빈도순 정렬
        mid_keywords_in_cluster.sort(key=lambda x: x[1], reverse=True)

        # 상위 3개 중분류 키워드 추가
        for mid_kw, _ in mid_keywords_in_cluster[:3]:
            keywords_set.add(mid_kw)

        enhanced_keywords[cluster_id] = list(keywords_set)[:15]

    return enhanced_keywords


# ============================================================================
# 11. 대분류별 클러스터링
# ============================================================================


def cluster_by_major_category(
    major_category: str,
    category_doc_map: Dict[str, Dict],
    category_rec_ids: List[str],
    category_embeddings: np.ndarray,
    mid_keywords: List[str],
    min_cluster_size: int = 8,
) -> Dict:
    """
    대분류 내에서 중분류 기반 클러스터링

    Returns:
        {
            "labels": np.ndarray,
            "clusters": Dict[int, List[str]],
            "keywords": Dict[int, List[str]],
            "statistics": Dict
        }
    """

    print(f"\n{'='*70}")
    print(f"📊 [{major_category}] 클러스터링")
    print(f"{'='*70}")
    print(f"   📝 문서 수: {len(category_rec_ids)}개")
    print(f"   📝 중분류 키워드 수: {len(mid_keywords)}개")

    # 문서 수가 너무 적으면 단일 클러스터로 처리
    if len(category_rec_ids) < min_cluster_size:
        print(f"   ⚠️  문서 수 부족, 단일 클러스터로 처리")
        return {
            "labels": np.zeros(len(category_rec_ids), dtype=int),
            "clusters": {0: category_rec_ids},
            "keywords": {0: mid_keywords[:10]},
            "statistics": {
                "total_docs": len(category_rec_ids),
                "num_clusters": 1,
                "noise_docs": 0,
                "avg_cluster_size": len(category_rec_ids),
                "min_cluster_size": len(category_rec_ids),
                "max_cluster_size": len(category_rec_ids),
            },
        }

    # UMAP 차원 축소
    reduced, reducer = reduce_dimensions(
        category_embeddings,
        n_components=min(15, len(category_rec_ids) - 2),
        n_neighbors=min(30, len(category_rec_ids) - 1),
        min_dist=0.0,
        random_state=42,
    )

    # HDBSCAN 클러스터링
    labels, clusterer = cluster_documents(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=3,
        cluster_selection_epsilon=0.01,
    )

    # TF-IDF 정제
    refined_clusters, cluster_keywords = refine_clusters_with_tfidf(
        category_doc_map,
        category_rec_ids,
        labels,
        top_n_keywords=10,
        min_docs_per_cluster=5,
    )

    # 중분류 키워드 매칭 추가
    enhanced_keywords = enhance_keywords_with_mid_classification(
        refined_clusters, cluster_keywords, category_doc_map, mid_keywords
    )

    # 통계
    num_noise = list(labels).count(-1)

    result = {
        "labels": labels,
        "clusters": refined_clusters,
        "keywords": enhanced_keywords,
        "statistics": {
            "total_docs": len(category_rec_ids),
            "num_clusters": len(refined_clusters),
            "noise_docs": num_noise,
            "avg_cluster_size": (
                np.mean([len(docs) for docs in refined_clusters.values()])
                if refined_clusters
                else 0
            ),
            "min_cluster_size": (
                min([len(docs) for docs in refined_clusters.values()])
                if refined_clusters
                else 0
            ),
            "max_cluster_size": (
                max([len(docs) for docs in refined_clusters.values()])
                if refined_clusters
                else 0
            ),
        },
    }

    return result


# ============================================================================
# 12. 시각화
# ============================================================================


def visualize_single_category(
    major_category: str, embeddings: np.ndarray, labels: np.ndarray, output_dir: Path
):
    """단일 대분류 시각화"""

    # 2D UMAP 축소
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, len(embeddings) - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    reduced_2d = reducer_2d.fit_transform(embeddings)

    # 시각화
    plt.figure(figsize=(12, 8))

    unique_labels = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:
            plt.scatter(
                reduced_2d[mask, 0],
                reduced_2d[mask, 1],
                c=[color],
                label="Noise",
                alpha=0.3,
                s=30,
            )
        else:
            plt.scatter(
                reduced_2d[mask, 0],
                reduced_2d[mask, 1],
                c=[color],
                label=f"Cluster {label}",
                alpha=0.6,
                s=50,
            )

    plt.title(
        f"{major_category} - Cluster Visualization", fontsize=14, fontweight="bold"
    )
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    # 파일명
    safe_name = major_category.replace("/", "_").replace("#", "").replace("·", "_")
    viz_file = output_dir / f"{safe_name}_visualization.png"
    plt.savefig(viz_file, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# 13. 결과 저장
# ============================================================================


def save_hierarchical_results(all_results: Dict, doc_map: Dict, output_dir: Path):
    """계층적 클러스터링 결과 저장"""

    print(f"\n💾 결과 저장 중...")

    # 1. 전체 요약
    summary_file = output_dir / "hierarchical_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("계층적 클러스터링 결과 요약\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 전체 통계
        total_major_cats = len(all_results)
        total_clusters = sum(
            r["statistics"]["num_clusters"] for r in all_results.values()
        )
        total_docs = sum(r["statistics"]["total_docs"] for r in all_results.values())

        f.write("📊 전체 통계:\n")
        f.write(f"   - 대분류 수: {total_major_cats}개\n")
        f.write(f"   - 총 세부 클러스터 수: {total_clusters}개\n")
        f.write(f"   - 총 문서 수: {total_docs}개\n\n")

        # 대분류별 요약
        f.write("📋 대분류별 요약:\n\n")
        for major_cat, result in sorted(all_results.items()):
            stats = result["statistics"]
            f.write(f"{major_cat}:\n")
            f.write(f"   - 문서 수: {stats['total_docs']}개\n")
            f.write(f"   - 세부 클러스터: {stats['num_clusters']}개\n")
            f.write(f"   - 노이즈: {stats['noise_docs']}개\n")
            f.write(f"   - 평균 클러스터 크기: {stats['avg_cluster_size']:.1f}개\n\n")

    print(f"   ✅ 전체 요약: {summary_file}")

    # 2. 대분류별 상세 결과
    for major_cat, result in all_results.items():
        safe_name = major_cat.replace("/", "_").replace("#", "").replace("·", "_")

        # JSON 저장
        detail_file = output_dir / f"{safe_name}_clusters.json"
        detail_data = {
            "major_category": major_cat,
            "statistics": result["statistics"],
            "clusters": {},
        }

        for cluster_id, rec_ids in result["clusters"].items():
            detail_data["clusters"][f"cluster_{cluster_id}"] = {
                "size": len(rec_ids),
                "keywords": result["keywords"].get(cluster_id, []),
                "documents": [
                    {
                        "rec_id": rec_id,
                        "title": doc_map[rec_id]["title"],
                        "cleaned_title": doc_map[rec_id]["cleaned_title"],
                        "company": doc_map[rec_id]["company"],
                        "tags": doc_map[rec_id]["cleaned_tags"][:10],
                    }
                    for rec_id in rec_ids[:20]  # 샘플 20개만
                ],
            }

        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(detail_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ [{major_cat}] 상세 결과: {detail_file}")

    # 3. CSV 요약
    csv_file = output_dir / "cluster_summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("major_category,cluster_id,size,keywords\n")

        for major_cat, result in sorted(all_results.items()):
            for cluster_id, rec_ids in result["clusters"].items():
                keywords_str = ", ".join(result["keywords"].get(cluster_id, [])[:5])
                f.write(f'"{major_cat}",{cluster_id},{len(rec_ids)},"{keywords_str}"\n')

    print(f"   ✅ CSV 요약: {csv_file}")

    # 4. 통합 JSON
    all_data_file = output_dir / "all_clusters_hierarchical.json"
    all_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_major_categories": len(all_results),
            "total_clusters": sum(
                r["statistics"]["num_clusters"] for r in all_results.values()
            ),
            "total_documents": sum(
                r["statistics"]["total_docs"] for r in all_results.values()
            ),
        },
        "results": {},
    }

    for major_cat, result in all_results.items():
        all_data["results"][major_cat] = {
            "statistics": result["statistics"],
            "cluster_ids": list(result["clusters"].keys()),
            "cluster_sizes": {k: len(v) for k, v in result["clusters"].items()},
            "keywords": result["keywords"],
        }

    with open(all_data_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 통합 결과: {all_data_file}")


# ============================================================================
# 14. 메인 파이프라인
# ============================================================================


def save_tag_based_results(doc_map: Dict, output_dir: Path):
    """태그 기반 분류 결과 저장 (다중 대분류/중분류 지원 + 중복 통계)"""

    print(f"\n💾 결과 저장 중...")

    # 전체 통계 계산
    unique_docs = len(doc_map)

    # 모든 대분류 할당 수집
    all_major_assignments = []
    all_mid_assignments = []
    for doc in doc_map.values():
        all_major_assignments.extend(doc["major_categories"])
        for major_cat, mid_cats in doc["category_assignments"].items():
            all_mid_assignments.extend([(major_cat, mid_cat) for mid_cat in mid_cats])

    total_major_assignments = len(all_major_assignments)
    total_mid_assignments = len(all_mid_assignments)

    # 대분류별 통계
    major_counts = Counter(all_major_assignments)

    # 대분류 할당 수 분포
    major_count_dist = Counter(len(doc["major_categories"]) for doc in doc_map.values())

    # 대분류 간 중복 통계
    major_overlap = defaultdict(int)
    for doc in doc_map.values():
        major_cats = doc["major_categories"]
        if len(major_cats) > 1:
            # 모든 쌍 조합
            for i, cat1 in enumerate(major_cats):
                for cat2 in major_cats[i + 1 :]:
                    pair = tuple(sorted([cat1, cat2]))
                    major_overlap[pair] += 1

    # 1. 전체 요약
    summary_file = output_dir / "tag_based_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("태그 기반 계층적 분류 결과 요약 (다중 대분류/중분류 지원)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 전체 통계
        f.write("📊 전체 통계:\n")
        f.write(f"   - 고유 문서 수: {unique_docs:,}개\n")
        f.write(
            f"   - 전체 대분류 할당 수: {total_major_assignments:,}개 (평균 {total_major_assignments/unique_docs:.2f}개/문서)\n"
        )
        f.write(
            f"   - 전체 중분류 할당 수: {total_mid_assignments:,}개 (평균 {total_mid_assignments/unique_docs:.2f}개/문서)\n"
        )
        f.write(f"   - 대분류 수: {len(major_counts)}개\n\n")

        # 대분류 할당 분포
        f.write("📊 대분류 할당 분포:\n")
        for count, num_docs in sorted(major_count_dist.items()):
            f.write(
                f"   - {count}개 대분류: {num_docs}개 문서 ({num_docs/unique_docs*100:.1f}%)\n"
            )
        f.write("\n")

        # 대분류 간 중복 통계
        f.write("📊 대분류 간 중복 (상위 15개):\n")
        for (cat1, cat2), count in sorted(
            major_overlap.items(), key=lambda x: x[1], reverse=True
        )[:15]:
            f.write(f"   - {cat1} + {cat2}: {count}개 문서\n")
        f.write("\n")

        # 대분류별 상세
        f.write("📋 대분류별 상세:\n\n")
        for major_cat, count in sorted(major_counts.items()):
            # 해당 대분류에 속한 고유 문서 수
            unique_docs_in_cat = sum(
                1 for doc in doc_map.values() if major_cat in doc["major_categories"]
            )

            f.write(f"{major_cat}:\n")
            f.write(f"   - 할당 문서 수: {count}개 (중복 포함)\n")
            f.write(f"   - 고유 문서 수: {unique_docs_in_cat}개\n")

            # 중분류 통계
            mid_counts = Counter(
                mid_cat for major, mid_cat in all_mid_assignments if major == major_cat
            )

            f.write(f"   - 중분류 수: {len(mid_counts)}개\n")
            f.write(f"   - 중분류 분포 (상위 10개):\n")
            for mid_cat, mid_count in mid_counts.most_common(10):
                f.write(f"      • {mid_cat}: {mid_count}개\n")
            f.write("\n")

    print(f"   ✅ 전체 요약: {summary_file}")

    # 2. 대분류별 상세 JSON
    for major_cat in major_counts.keys():
        safe_name = major_cat.replace("/", "_").replace("#", "").replace("·", "_")

        # 해당 대분류에 속한 문서들
        major_docs = [
            doc for doc in doc_map.values() if major_cat in doc["major_categories"]
        ]

        unique_docs_in_cat = len(major_docs)

        # 중분류별 그룹화
        mid_groups = defaultdict(list)
        for doc in major_docs:
            mid_cats = doc["category_assignments"].get(major_cat, ["기타"])
            for mid_cat in mid_cats:
                mid_groups[mid_cat].append(doc)

        # 중복 통계 (이 대분류와 함께 나타나는 다른 대분류)
        overlap_info = {}
        for other_cat, count in major_overlap.items():
            if major_cat in other_cat:
                other = other_cat[0] if other_cat[1] == major_cat else other_cat[1]
                overlap_info[f"with_{other.replace('#', '').replace('·', '_')}"] = count

        detail_file = output_dir / f"{safe_name}_classification.json"
        detail_data = {
            "major_category": major_cat,
            "statistics": {
                "total_docs": len(major_docs),  # 중복 포함
                "unique_docs": unique_docs_in_cat,  # 중복 제거
                "num_mid_categories": len(mid_groups),
                "overlap_info": overlap_info,
            },
            "mid_categories": {},
        }

        for mid_cat, docs in sorted(
            mid_groups.items(), key=lambda x: len(x[1]), reverse=True
        ):
            # 중복 제거 (같은 문서가 여러 중분류에 나타날 수 있음)
            unique_docs_in_mid = len(set(doc["rec_id"] for doc in docs))

            detail_data["mid_categories"][mid_cat] = {
                "size": unique_docs_in_mid,  # 고유 문서 수
                "total_assignments": len(docs),  # 중복 포함
                "documents": [],
            }

            # 각 문서에 대해 다른 대분류 정보 추가
            seen_rec_ids = set()
            for doc in docs:
                rec_id = doc["rec_id"]
                if rec_id in seen_rec_ids:
                    continue
                seen_rec_ids.add(rec_id)

                other_major_cats = [
                    c for c in doc["major_categories"] if c != major_cat
                ]
                mid_cats_in_others = {
                    other_cat: doc["category_assignments"].get(other_cat, [])
                    for other_cat in other_major_cats
                }

                detail_data["mid_categories"][mid_cat]["documents"].append(
                    {
                        "rec_id": doc["rec_id"],
                        "title": doc["title"],
                        "cleaned_title": doc["cleaned_title"],
                        "company": doc["company"],
                        "url": doc.get("url", ""),  # URL 추가
                        "tags": doc["filtered_tags"],
                        "other_major_categories": other_major_cats,
                        "mid_categories_in_this_major": doc["category_assignments"].get(
                            major_cat, []
                        ),
                        "mid_categories_in_others": mid_cats_in_others,
                    }
                )

                if len(detail_data["mid_categories"][mid_cat]["documents"]) >= 20:
                    break

        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(detail_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ [{major_cat}] 상세 결과: {detail_file}")

    # 3. CSV 요약
    csv_file = output_dir / "classification_summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("major_category,mid_category,unique_docs,total_assignments\n")

        for major_cat in sorted(major_counts.keys()):
            mid_counts = Counter(
                mid_cat for major, mid_cat in all_mid_assignments if major == major_cat
            )

            # 중분류별 고유 문서 수 계산
            for mid_cat in mid_counts.keys():
                unique_count = sum(
                    1
                    for doc in doc_map.values()
                    if major_cat in doc["major_categories"]
                    and mid_cat in doc["category_assignments"].get(major_cat, [])
                )
                f.write(
                    f'"{major_cat}","{mid_cat}",{unique_count},{mid_counts[mid_cat]}\n'
                )

    print(f"   ✅ CSV 요약: {csv_file}")

    # 4. 통합 JSON
    all_data_file = output_dir / "all_classifications.json"
    all_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_unique_documents": unique_docs,
            "total_major_assignments": total_major_assignments,
            "total_mid_assignments": total_mid_assignments,
            "avg_major_per_doc": total_major_assignments / unique_docs,
            "avg_mid_per_doc": total_mid_assignments / unique_docs,
        },
        "classifications": {},
    }

    for major_cat in sorted(major_counts.keys()):
        unique_docs_in_cat = sum(
            1 for doc in doc_map.values() if major_cat in doc["major_categories"]
        )

        mid_counts = Counter(
            mid_cat for major, mid_cat in all_mid_assignments if major == major_cat
        )

        all_data["classifications"][major_cat] = {
            "total_assignments": major_counts[major_cat],
            "unique_docs": unique_docs_in_cat,
            "num_mid_categories": len(mid_counts),
            "mid_category_distribution": dict(mid_counts.most_common()),
        }

    with open(all_data_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 통합 결과: {all_data_file}")


def main():
    # 기본 설정
    chunk_file = "structured_chunks/all_chunks_20251120_141253.json"
    output_dir = Path("clustering_results_tag_based")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("🚀 태그 기반 계층적 분류 파이프라인 시작")
    print("=" * 80)
    print("📌 대분류 → 중분류 태그 매칭 기반 분류")
    print("📌 임베딩/DBSCAN은 결과 확인 후 결정")
    print("=" * 80)

    # Step 1: 청크 로드
    chunks = load_chunks(chunk_file)

    # Step 2: rec_id별 집계 + 전처리 + 대분류/중분류 할당
    doc_map = aggregate_by_rec_id(chunks)

    # Step 3: 결과 저장
    save_tag_based_results(doc_map, output_dir)

    print("\n" + "=" * 80)
    print("✅ 태그 기반 분류 파이프라인 완료!")
    print("=" * 80)
    print(f"\n📁 결과 디렉토리: {output_dir.absolute()}")
    print("\n💡 다음 단계:")
    print("   1. 결과 확인 (classification_summary.csv)")
    print("   2. 품질 평가 후 임베딩/DBSCAN 적용 여부 결정")


if __name__ == "__main__":
    main()
