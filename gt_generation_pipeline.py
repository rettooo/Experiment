#!/usr/bin/env python3
"""
Ground Truth 생성 파이프라인

Phase 1: 중분류 기반 초기 클러스터 생성
Phase 2: 유사 중분류 병합 (규칙 기반)
Phase 3: 대표 문서 선택 (group_name + mid_categories와 cleaned_title 간 유사도 기반)
Phase 4: 통계 및 결과 저장

================================================================================
입력 데이터:
================================================================================
1. clustering_results_tag_based/*_classification.json
   - 각 대분류별 분류 결과 파일
   - 각 문서는 rec_id, title, cleaned_title, tags, company 등을 포함
   - 예시: 건설_건축_classification.json, it개발_데이터_classification.json 등

2. similarity_rules_template.json
   - 유사 중분류를 그룹으로 묶는 규칙 파일
   - 형식: {대분류: {group_name: [중분류1, 중분류2, ...]}}
   - 예시: {"#건설·건축": {"품질관리": ["QA", "QC"], "설계": ["설계엔지니어", "3D설계", ...]}}

================================================================================
출력 데이터:
================================================================================
1. gt_generation_results/gt_clusters.json
   - 메타데이터와 클러스터 정보 포함
   - 각 클러스터는 다음 필드를 포함:
     * major_category: 대분류
     * group_name: 그룹명
     * mid_categories: 중분류 리스트
     * num_docs: 문서 수
     * rec_ids: 전체 문서 ID 리스트
     * representative_docs: 대표 문서 ID 리스트 (3개)

   예시 구조:
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
       },
       ...
     }
   }

2. gt_generation_results/gt_clusters_summary.csv
   - 클러스터 요약 정보 (CSV 형식)
   - 컬럼: cluster_name, major_category, group_name, mid_categories,
           num_mid_categories, num_docs, representative_docs

3. gt_generation_results/gt_generation_statistics.txt
   - 전체 통계 정보 (텍스트 형식)
   - 전체/대분류별 클러스터 수, 문서 수, 크기 분포 등

================================================================================
처리 과정:
================================================================================
1. Phase 1: 중분류 기반 초기 클러스터 생성
   - 입력: clustering_results_tag_based/*_classification.json
   - 처리: 각 중분류별로 문서 수집 (최소 5개 문서 이상)
   - 출력: {중분류: {rec_id 집합}}

2. Phase 2: 유사 중분류 병합 (규칙 기반)
   - 입력: Phase 1 결과 + similarity_rules_template.json
   - 처리: 유사한 중분류들을 group_name으로 묶어서 클러스터 생성
   - 출력: {cluster_name: {rec_ids, mid_categories, major_category, group_name}}

3. Phase 3: 대표 문서 선택
   - 입력: Phase 2 결과 + doc_map (cleaned_title 포함)
   - 처리:
     a. 각 클러스터의 group_name + mid_categories를 쿼리 텍스트로 생성
        예: "품질관리 QA QC"
     b. 해당 클러스터의 rec_ids에 속한 문서들의 cleaned_title 수집
     c. SentenceTransformer로 쿼리와 문서 제목 임베딩 생성
     d. 코사인 유사도 계산하여 상위 3개 선택
   - 출력: 각 클러스터에 representative_docs 필드 추가

4. Phase 4: 통계 및 결과 저장
   - 입력: Phase 3 결과
   - 처리: 통계 계산 및 JSON/CSV/TXT 파일 저장
   - 출력: gt_generation_results/ 폴더의 모든 결과 파일

================================================================================
대표 문서 선택 예시:
================================================================================
클러스터: "#건설·건축_품질관리"
- group_name: "품질관리"
- mid_categories: ["QA", "QC"]
- 쿼리 텍스트: "품질관리 QA QC"
- rec_ids: ["50285218", "50463673", "50314015", ...]

각 문서의 cleaned_title:
- "50285218": "태광 QA팀"
- "50463673": "품질관리 QA"
- "50314015": "QA 품질보증"
- ...

유사도 계산 결과:
- "50314015": 0.85 (가장 높음)
- "50440884": 0.82
- "50463673": 0.80
- ...

최종 representative_docs: ["50314015", "50440884", "50463673"]
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_doc_map():
    """문서 맵 로드 (분류 결과에서)"""
    # 실제로는 structured_chunks에서 로드하거나, 분류 결과를 사용
    # 여기서는 분류 결과 JSON 파일들을 로드
    doc_map = {}

    # 모든 대분류 분류 파일 로드
    classification_dir = Path("clustering_results_tag_based")

    for json_file in classification_dir.glob("*_classification.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        major_cat = data["major_category"]

        # 각 문서 추출
        for mid_cat, mid_data in data["mid_categories"].items():
            for doc in mid_data["documents"]:
                rec_id = doc["rec_id"]

                if rec_id not in doc_map:
                    doc_map[rec_id] = {
                        "rec_id": rec_id,
                        "title": doc["title"],
                        "cleaned_title": doc.get("cleaned_title", doc["title"]),
                        "company": doc["company"],
                        "tags": doc["tags"],
                        "major_categories": [],
                        "category_assignments": {},
                    }

                # 대분류 추가
                if major_cat not in doc_map[rec_id]["major_categories"]:
                    doc_map[rec_id]["major_categories"].append(major_cat)

                # 중분류 추가
                if major_cat not in doc_map[rec_id]["category_assignments"]:
                    doc_map[rec_id]["category_assignments"][major_cat] = []

                mid_cats_in_major = doc.get("mid_categories_in_this_major", [mid_cat])
                for mid in mid_cats_in_major:
                    if mid not in doc_map[rec_id]["category_assignments"][major_cat]:
                        doc_map[rec_id]["category_assignments"][major_cat].append(mid)

    print(f"✅ 문서 맵 로드 완료: {len(doc_map):,}개 문서")
    return doc_map


def load_similarity_rules(rules_file: str = "similarity_rules_template.json") -> Dict:
    """유사도 규칙 파일 로드"""
    rules_path = Path(rules_file)
    if not rules_path.exists():
        print(f"❌ 규칙 파일을 찾을 수 없습니다: {rules_file}")
        return {}

    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    print(f"✅ 유사도 규칙 로드 완료: {len(rules)}개 대분류")
    return rules


def phase1_collect_mid_clusters(doc_map: Dict) -> Dict[str, Set[str]]:
    """
    Phase 1: 중분류 기반 초기 클러스터 생성

    Returns:
        {mid_category: set of rec_ids}
    """
    print("\n" + "=" * 80)
    print("Phase 1: 중분류 기반 초기 클러스터 생성")
    print("=" * 80)

    mid_to_docs = defaultdict(set)
    mid_to_major = defaultdict(set)

    for doc in doc_map.values():
        for major_cat, mid_cats in doc["category_assignments"].items():
            for mid_cat in mid_cats:
                mid_to_docs[mid_cat].add(doc["rec_id"])
                mid_to_major[mid_cat].add(major_cat)

    # 최소 문서 수 필터링
    MIN_DOCS = 5
    valid_mid_clusters = {
        mid: docs for mid, docs in mid_to_docs.items() if len(docs) >= MIN_DOCS
    }

    print(f"   📊 전체 중분류 수: {len(mid_to_docs)}개")
    print(f"   ✅ 유효 중분류 (≥{MIN_DOCS}개 문서): {len(valid_mid_clusters)}개")
    print(f"   ❌ 제외된 중분류: {len(mid_to_docs) - len(valid_mid_clusters)}개")

    # 통계
    size_dist = Counter(len(docs) for docs in valid_mid_clusters.values())
    print(f"\n   📊 중분류별 문서 수 분포:")
    for size, count in sorted(size_dist.items(), reverse=True)[:10]:
        print(f"      - {size}개 문서: {count}개 중분류")

    return valid_mid_clusters, mid_to_major


def phase2_merge_similar_clusters(
    mid_clusters: Dict[str, Set[str]],
    similarity_rules: Dict,
    mid_to_major: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    """
    Phase 2: 유사 중분류 병합 (규칙 기반)

    Returns:
        {cluster_name: [rec_id, ...]}
    """
    print("\n" + "=" * 80)
    print("Phase 2: 유사 중분류 병합 (규칙 기반)")
    print("=" * 80)

    merged_clusters = {}
    processed_mids = set()

    # 각 대분류별로 처리
    for major_cat, groups in similarity_rules.items():
        print(f"\n   📋 [{major_cat}] 처리 중...")

        for group_name, mid_list in groups.items():
            # 해당 대분류에 속한 중분류만 필터링
            valid_mids = [
                mid
                for mid in mid_list
                if mid in mid_clusters and major_cat in mid_to_major.get(mid, set())
            ]

            if not valid_mids:
                continue

            # 클러스터명 생성
            cluster_name = f"{major_cat}_{group_name}"

            # 모든 문서 수집
            cluster_docs = set()
            for mid in valid_mids:
                cluster_docs.update(mid_clusters[mid])
                processed_mids.add(mid)

            if len(cluster_docs) >= 5:  # 최소 5개 문서
                merged_clusters[cluster_name] = {
                    "rec_ids": list(cluster_docs),
                    "mid_categories": valid_mids,
                    "major_category": major_cat,
                    "group_name": group_name,
                }
                print(
                    f"      ✅ {cluster_name}: {len(valid_mids)}개 중분류 → {len(cluster_docs)}개 문서"
                )

    # 규칙에 포함되지 않은 중분류 처리
    unprocessed_mids = set(mid_clusters.keys()) - processed_mids
    if unprocessed_mids:
        print(f"\n   ⚠️  규칙에 포함되지 않은 중분류: {len(unprocessed_mids)}개")
        for mid in sorted(unprocessed_mids)[:10]:
            docs_count = len(mid_clusters[mid])
            majors = ", ".join(sorted(mid_to_major.get(mid, set())))
            print(f"      - {mid}: {docs_count}개 문서 ({majors})")

        # 개별 중분류를 클러스터로 유지
        for mid in unprocessed_mids:
            if len(mid_clusters[mid]) >= 5:
                major = list(mid_to_major.get(mid, {"기타"}))[0]
                cluster_name = f"{major}_{mid}"
                merged_clusters[cluster_name] = {
                    "rec_ids": list(mid_clusters[mid]),
                    "mid_categories": [mid],
                    "major_category": major,
                    "group_name": mid,
                }

    print(f"\n   ✅ 최종 클러스터 수: {len(merged_clusters)}개")

    # 통계
    cluster_sizes = [len(c["rec_ids"]) for c in merged_clusters.values()]
    print(f"\n   📊 클러스터 크기 분포:")
    size_dist = Counter(cluster_sizes)
    for size, count in sorted(size_dist.items(), reverse=True)[:10]:
        print(f"      - {size}개 문서: {count}개 클러스터")

    return merged_clusters


def phase3_select_representative_docs(
    clusters: Dict,
    doc_map: Dict,
    n: int = 3,
    model_name: str = "jhgan/ko-sroberta-multitask",
) -> Dict:
    """
    Phase 3: 대표 문서 선택 (group_name + mid_categories와 cleaned_title 간 유사도 기반)

    각 클러스터별로:
    1. group_name + mid_categories 텍스트를 쿼리로 생성
    2. 해당 클러스터의 rec_ids에 속한 문서들의 cleaned_title 수집
    3. 쿼리 임베딩과 문서 제목 임베딩 간 코사인 유사도 계산
    4. 가장 유사한 상위 n개 문서 선택

    입력:
        - clusters: 클러스터 딕셔너리 (group_name, mid_categories, rec_ids 포함)
        - doc_map: 문서 맵 (rec_id -> cleaned_title 포함)
        - n: 선택할 대표 문서 수 (기본값: 3)
        - model_name: 임베딩 모델명

    출력:
        - clusters: representative_docs 필드가 추가된 클러스터 딕셔너리
    """
    print("\n" + "=" * 80)
    print("Phase 3: 대표 문서 선택 (group_name + mid_categories 기반 유사도)")
    print("=" * 80)

    # 임베딩 모델 로드
    print(f"   📥 임베딩 모델 로드 중: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"   ✅ 모델 로드 완료")
    except Exception as e:
        print(f"   ❌ 모델 로드 실패: {e}")
        print(f"   ⚠️  기존 점수 기반 방법으로 대체합니다.")
        return phase3_select_representative_docs_fallback(clusters, doc_map, n)

    # 각 클러스터별로 처리
    total_clusters = len(clusters)
    processed = 0
    skipped = 0

    for cluster_name, cluster_data in tqdm(clusters.items(), desc="   클러스터 처리"):
        group_name = cluster_data["group_name"]
        mid_categories = cluster_data["mid_categories"]
        rec_ids = cluster_data["rec_ids"]

        # 클러스터 쿼리 텍스트 생성 (group_name + mid_categories)
        mid_cats_str = " ".join(mid_categories)
        query_text = f"{group_name} {mid_cats_str}".strip()

        # 유효한 rec_ids 필터링 (cleaned_title이 있는 것만)
        valid_rec_ids = []
        valid_cleaned_titles = []

        for rec_id in rec_ids:
            if rec_id not in doc_map:
                continue

            doc = doc_map[rec_id]
            cleaned_title = doc.get("cleaned_title", doc.get("title", ""))

            if cleaned_title and len(cleaned_title.strip()) > 0:
                valid_rec_ids.append(rec_id)
                valid_cleaned_titles.append(cleaned_title)

        if len(valid_rec_ids) < n:
            # 문서가 부족한 경우 모든 문서 선택
            cluster_data["representative_docs"] = valid_rec_ids[:n]
            skipped += 1
            processed += 1
            continue

        try:
            # 쿼리 텍스트 임베딩
            query_embedding = model.encode(
                [query_text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

            # 문서 제목들 임베딩
            doc_embeddings = model.encode(
                valid_cleaned_titles,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # 유사도 계산 (코사인 유사도)
            similarities = np.dot(doc_embeddings, query_embedding)

            # 상위 n개 선택
            top_indices = np.argsort(similarities)[::-1][:n]
            selected_rec_ids = [valid_rec_ids[i] for i in top_indices]

            cluster_data["representative_docs"] = selected_rec_ids
            processed += 1

        except Exception as e:
            print(f"   ⚠️  {cluster_name} 처리 실패: {e}")
            # 실패 시 첫 n개 선택
            cluster_data["representative_docs"] = valid_rec_ids[:n]
            skipped += 1
            processed += 1

    print(f"\n   ✅ {processed}/{total_clusters}개 클러스터 처리 완료")
    if skipped > 0:
        print(f"   ⚠️  {skipped}개 클러스터는 문서 부족 또는 오류로 인해 첫 n개 선택")

    return clusters


def phase3_select_representative_docs_fallback(
    clusters: Dict, doc_map: Dict, n: int = 3
) -> Dict:
    """
    Fallback: 기존 점수 기반 방법
    """
    print("   ⚠️  Fallback: 점수 기반 방법 사용")

    for cluster_name, cluster_data in clusters.items():
        cluster_docs = cluster_data["rec_ids"]

        scored_docs = []
        for rec_id in cluster_docs:
            if rec_id not in doc_map:
                continue

            doc = doc_map[rec_id]

            # 점수 계산
            score = (
                len(doc["major_categories"]) * 2  # 대분류 다양성
                + sum(
                    len(mids) for mids in doc["category_assignments"].values()
                )  # 중분류 다양성
                + len(doc.get("tags", [])) * 0.1  # 태그 풍부도
            )

            scored_docs.append((rec_id, score))

        # 상위 N개 선택
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        cluster_data["representative_docs"] = [rec_id for rec_id, _ in scored_docs[:n]]

    return clusters


def phase4_generate_statistics(clusters: Dict, doc_map: Dict, output_dir: Path):
    """
    Phase 4: 통계 및 결과 저장
    """
    print("\n" + "=" * 80)
    print("Phase 4: 통계 및 결과 저장")
    print("=" * 80)

    output_dir.mkdir(exist_ok=True, parents=True)

    # 전체 통계
    total_clusters = len(clusters)
    total_docs = sum(len(c["rec_ids"]) for c in clusters.values())
    unique_docs = len(set(rec_id for c in clusters.values() for rec_id in c["rec_ids"]))

    # 대분류별 통계
    major_stats = defaultdict(lambda: {"clusters": 0, "docs": 0, "unique_docs": set()})
    for cluster_data in clusters.values():
        major = cluster_data["major_category"]
        major_stats[major]["clusters"] += 1
        major_stats[major]["docs"] += len(cluster_data["rec_ids"])
        major_stats[major]["unique_docs"].update(cluster_data["rec_ids"])

    # 통계 파일 저장
    stats_file = output_dir / "gt_generation_statistics.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Ground Truth 생성 통계\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("📊 전체 통계:\n")
        f.write(f"   - 총 클러스터 수: {total_clusters}개\n")
        f.write(f"   - 총 문서 할당 수: {total_docs:,}개 (중복 포함)\n")
        f.write(f"   - 고유 문서 수: {unique_docs:,}개\n")
        f.write(f"   - 평균 문서 수/클러스터: {total_docs/total_clusters:.1f}개\n")
        f.write(f"   - 평균 클러스터 수/문서: {total_docs/unique_docs:.2f}개\n\n")

        f.write("📋 대분류별 통계:\n\n")
        for major, stats in sorted(major_stats.items()):
            f.write(f"{major}:\n")
            f.write(f"   - 클러스터 수: {stats['clusters']}개\n")
            f.write(f"   - 문서 할당 수: {stats['docs']}개 (중복 포함)\n")
            f.write(f"   - 고유 문서 수: {len(stats['unique_docs'])}개\n")
            f.write(
                f"   - 평균 문서 수/클러스터: {stats['docs']/stats['clusters']:.1f}개\n\n"
            )

        f.write("📊 클러스터 크기 분포:\n")
        cluster_sizes = [len(c["rec_ids"]) for c in clusters.values()]
        size_dist = Counter(cluster_sizes)
        for size, count in sorted(size_dist.items(), reverse=True):
            f.write(f"   - {size}개 문서: {count}개 클러스터\n")

    print(f"   ✅ 통계 파일 저장: {stats_file}")

    # 클러스터 상세 정보 저장 (JSON)
    clusters_file = output_dir / "gt_clusters.json"
    clusters_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_clusters": total_clusters,
            "total_docs": total_docs,
            "unique_docs": unique_docs,
        },
        "clusters": {},
    }

    for cluster_name, cluster_data in clusters.items():
        clusters_data["clusters"][cluster_name] = {
            "major_category": cluster_data["major_category"],
            "group_name": cluster_data["group_name"],
            "mid_categories": cluster_data["mid_categories"],
            "num_docs": len(cluster_data["rec_ids"]),
            "rec_ids": cluster_data["rec_ids"],  # 전체 문서
            "representative_docs": cluster_data["representative_docs"],
        }

    with open(clusters_file, "w", encoding="utf-8") as f:
        json.dump(clusters_data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 클러스터 정보 저장: {clusters_file}")

    # CSV 요약
    csv_file = output_dir / "gt_clusters_summary.csv"
    rows = []
    for cluster_name, cluster_data in clusters.items():
        rows.append(
            {
                "cluster_name": cluster_name,
                "major_category": cluster_data["major_category"],
                "group_name": cluster_data["group_name"],
                "mid_categories": ", ".join(cluster_data["mid_categories"]),
                "num_mid_categories": len(cluster_data["mid_categories"]),
                "num_docs": len(cluster_data["rec_ids"]),
                "representative_docs": ", ".join(cluster_data["representative_docs"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"   ✅ CSV 요약 저장: {csv_file}")

    # 대분류별 샘플 출력
    print("\n📋 대분류별 클러스터 샘플:")
    for major in sorted(set(c["major_category"] for c in clusters.values()))[:5]:
        major_clusters = [
            (name, data)
            for name, data in clusters.items()
            if data["major_category"] == major
        ]
        major_clusters.sort(key=lambda x: len(x[1]["rec_ids"]), reverse=True)

        print(f"\n   {major}:")
        for cluster_name, cluster_data in major_clusters[:3]:
            print(f"      - {cluster_name}: {len(cluster_data['rec_ids'])}개 문서")
            print(f"        중분류: {', '.join(cluster_data['mid_categories'][:5])}")
            print(
                f"        대표 문서: {', '.join(cluster_data['representative_docs'])}"
            )


def validate_similarity_rules(
    rules: Dict, classification_data: Optional[Dict] = None
) -> bool:
    """
    유사도 규칙 검증
    
    Args:
        rules: 유사도 규칙 딕셔너리
        classification_data: 분류 데이터 (None이면 자동 로드)
    
    Returns:
        검증 통과 여부
    """
    if classification_data is None:
        classification_file = Path("clustering_results_tag_based/all_classifications.json")
        if not classification_file.exists():
            print("⚠️  분류 데이터 파일을 찾을 수 없어 검증을 건너뜁니다.")
            return True
        
        with open(classification_file, "r", encoding="utf-8") as f:
            classification_data = json.load(f)
    
    print("=" * 80)
    print("유사도 규칙 검증")
    print("=" * 80)
    
    all_rule_mids = set()
    errors = []
    warnings = []
    
    for major_cat, groups in rules.items():
        if major_cat not in classification_data.get("classifications", {}):
            errors.append(f"❌ 대분류 '{major_cat}'가 데이터에 없습니다")
            continue
        
        actual_mids = set(
            classification_data["classifications"][major_cat].get(
                "mid_category_distribution", {}
            ).keys()
        )
        
        for group_name, mid_list in groups.items():
            for mid in mid_list:
                if mid in all_rule_mids:
                    warnings.append(f"⚠️  중분류 '{mid}'가 여러 그룹에 포함됨")
                all_rule_mids.add(mid)
                
                if mid not in actual_mids:
                    errors.append(
                        f"❌ '{major_cat}'의 중분류 '{mid}'가 데이터에 없습니다"
                    )
        
        missing_mids = actual_mids - all_rule_mids
        if missing_mids:
            missing_mids = {m for m in missing_mids if m != "기타"}
            if missing_mids:
                warnings.append(
                    f"⚠️  '{major_cat}'의 다음 중분류가 규칙에 없습니다: {sorted(missing_mids)[:10]}"
                )
    
    if errors:
        print("\n❌ 오류:")
        for error in errors[:20]:
            print(f"   {error}")
        if len(errors) > 20:
            print(f"   ... 외 {len(errors)-20}개 오류")
    
    if warnings:
        print("\n⚠️  경고:")
        for warning in warnings[:20]:
            print(f"   {warning}")
        if len(warnings) > 20:
            print(f"   ... 외 {len(warnings)-20}개 경고")
    
    if not errors and not warnings:
        print("\n✅ 모든 규칙이 유효합니다!")
    
    return len(errors) == 0


def convert_gt_csv_to_jsonl(
    csv_path: Path, jsonl_path: Path
) -> None:
    """
    GT CSV를 파이프라인에서 기대하는 JSONL 형식으로 변환
    
    Args:
        csv_path: 입력 CSV 파일 경로
        jsonl_path: 출력 JSONL 파일 경로
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f_in, jsonl_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)

        required_fields = ["GT_ID", "완전한_검색_쿼리", "같은_군집_id들"]
        for field in required_fields:
            if field not in reader.fieldnames:
                raise ValueError(
                    f"CSV에 필요한 컬럼이 없습니다: {field} (필드 목록: {reader.fieldnames})"
                )

        count = 0
        for row in reader:
            gt_id_raw = row.get("GT_ID")
            if gt_id_raw is None or gt_id_raw == "":
                continue

            try:
                query_id = int(gt_id_raw)
            except ValueError:
                query_id = gt_id_raw

            query_text = row.get("완전한_검색_쿼리", "").strip()

            cluster_ids_raw = row.get("같은_군집_id들", "")
            if cluster_ids_raw is None:
                cluster_ids_raw = ""

            rec_ids = [
                part.strip()
                for part in cluster_ids_raw.split(",")
                if part.strip()
            ]

            ground_truth = [{"rec_idx": rec_id} for rec_id in rec_ids]

            record = {
                "query_id": query_id,
                "query_text": query_text,
                "ground_truth": ground_truth,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ 변환 완료: {count}개 레코드 → {jsonl_path}")


def create_evaluation_data(
    csv_path: Path, output_jsonl_path: Path
) -> None:
    """
    GT Analysis CSV를 평가용 JSONL 데이터로 변환
    
    Args:
        csv_path: 입력 CSV 파일 경로
        output_jsonl_path: 출력 JSONL 파일 경로
    """
    def extract_rec_idx_from_url(url: str) -> str:
        """URL에서 rec_idx 추출"""
        if not url or not url.startswith("http"):
            return ""
        if "rec_idx=" in url:
            return url.split("rec_idx=")[-1].split("&")[0]
        return ""

    print(f"{'='*60}")
    print(f"GT Analysis CSV → 평가용 데이터 변환")
    print(f"{'='*60}")
    print(f"입력: {csv_path.name}")
    print(f"출력: {output_jsonl_path.name}")

    grouped_data = defaultdict(lambda: {"query_text": "", "ground_truth": []})
    total_rows, skipped_rows = 0, 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            query_id = row.get("GT_ID", "").strip()
            if not query_id:
                skipped_rows += 1
                continue

            if not grouped_data[query_id]["query_text"]:
                query_text = row.get("완전한_검색_쿼리", "").strip()
                if not query_text:
                    skipped_rows += 1
                    continue
                grouped_data[query_id]["query_text"] = query_text

            url = row.get("URL", "").strip()
            if url:
                rec_idx = extract_rec_idx_from_url(url)
                if rec_idx:
                    gt_doc = {
                        "rec_idx": rec_idx,
                        "job_title": row.get("공고_제목", "").strip(),
                        "url": url,
                    }
                    if not any(
                        doc["rec_idx"] == rec_idx
                        for doc in grouped_data[query_id]["ground_truth"]
                    ):
                        grouped_data[query_id]["ground_truth"].append(gt_doc)

    print(f"✅ 총 {total_rows}개 행 처리 완료, {skipped_rows}개 행 건너뜀")
    print(f" - 고유 쿼리: {len(grouped_data)}개")

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    total_queries = len(grouped_data)
    total_gt = 0

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for query_id, query_data in sorted(
            grouped_data.items(),
            key=lambda x: int(x[0]) if x[0].isdigit() else 0,
        ):
            output_entry = {
                "query_id": query_id,
                "query_text": query_data["query_text"],
                "ground_truth": query_data["ground_truth"],
            }
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            total_gt += len(query_data["ground_truth"])

    print(f"✅ 총 {total_queries}개 쿼리, {total_gt}개 정답 문서 저장 완료")
    print(f" - 평균 정답 문서/쿼리: {total_gt/total_queries:.2f}")


def run_full_pipeline():
    """전체 GT 생성 파이프라인 실행"""
    print("=" * 80)
    print("🚀 Ground Truth 생성 파이프라인")
    print("=" * 80)
    print("Phase 1: 중분류 기반 초기 클러스터 생성")
    print("Phase 2: 유사 중분류 병합 (규칙 기반)")
    print("Phase 3: 대표 문서 선택")
    print("Phase 4: 통계 및 결과 저장")
    print("=" * 80)

    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    doc_map = load_doc_map()
    similarity_rules = load_similarity_rules()

    if not similarity_rules:
        print("❌ 유사도 규칙을 로드할 수 없습니다. 종료합니다.")
        return

    # 규칙 검증
    print("\n🔍 유사도 규칙 검증 중...")
    is_valid = validate_similarity_rules(similarity_rules)
    if not is_valid:
        print("⚠️  규칙에 오류가 있습니다. 수정 후 다시 실행하세요.")
        return

    # Phase 1: 중분류 기반 초기 클러스터 생성
    mid_clusters, mid_to_major = phase1_collect_mid_clusters(doc_map)

    # Phase 2: 유사 중분류 병합
    merged_clusters = phase2_merge_similar_clusters(
        mid_clusters, similarity_rules, mid_to_major
    )

    # Phase 3: 대표 문서 선택
    final_clusters = phase3_select_representative_docs(merged_clusters, doc_map)

    # Phase 4: 통계 및 결과 저장
    output_dir = Path("gt_generation_results")
    phase4_generate_statistics(final_clusters, doc_map, output_dir)

    print("\n" + "=" * 80)
    print("✅ Ground Truth 생성 파이프라인 완료!")
    print("=" * 80)
    print(f"\n📁 결과 디렉토리: {output_dir.absolute()}")
    print("\n💡 다음 단계:")
    print("   1. 통계 확인 (gt_generation_statistics.txt)")
    print("   2. 클러스터 품질 검토 (gt_clusters.json)")
    print("   3. 필요시 규칙 수정 및 재실행")


def main():
    """메인 함수 (명령줄 인자 지원)"""
    parser = argparse.ArgumentParser(
        description="Ground Truth 생성 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행
  python gt_generation_pipeline.py

  # CSV → JSONL 변환
  python gt_generation_pipeline.py --convert-csv data/gt.csv data/output.jsonl

  # 평가용 데이터 생성
  python gt_generation_pipeline.py --create-eval data/gt_analysis.csv data/eval.jsonl

  # 규칙 검증만 실행
  python gt_generation_pipeline.py --validate-rules
        """,
    )

    parser.add_argument(
        "--convert-csv",
        nargs=2,
        metavar=("CSV_PATH", "JSONL_PATH"),
        help="GT CSV를 JSONL로 변환",
    )
    parser.add_argument(
        "--create-eval",
        nargs=2,
        metavar=("CSV_PATH", "JSONL_PATH"),
        help="GT Analysis CSV를 평가용 JSONL로 변환",
    )
    parser.add_argument(
        "--validate-rules",
        action="store_true",
        help="유사도 규칙 검증만 실행",
    )

    args = parser.parse_args()

    # CSV → JSONL 변환
    if args.convert_csv:
        csv_path = Path(args.convert_csv[0])
        jsonl_path = Path(args.convert_csv[1])
        convert_gt_csv_to_jsonl(csv_path, jsonl_path)
        return

    # 평가용 데이터 생성
    if args.create_eval:
        csv_path = Path(args.create_eval[0])
        jsonl_path = Path(args.create_eval[1])
        create_evaluation_data(csv_path, jsonl_path)
        return

    # 규칙 검증만 실행
    if args.validate_rules:
        similarity_rules = load_similarity_rules()
        if similarity_rules:
            validate_similarity_rules(similarity_rules)
        return

    # 기본: 전체 파이프라인 실행
    run_full_pipeline()


if __name__ == "__main__":
    main()
