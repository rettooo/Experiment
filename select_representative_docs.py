"""
중분류별 대표 문서 3개 선정 스크립트

선정 기준:
1. 태그 다양성 점수 (태그 수가 적절한 문서 선호)
2. 단일 분류 점수 (다른 대분류에 속하지 않는 문서 선호)
3. 제목 명확성 점수 (제목이 중분류를 잘 나타내는 문서 선호)
4. 태그 관련성 점수 (중분류 키워드와 관련된 태그가 많은 문서 선호)
"""

import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import re


def calculate_tag_diversity_score(doc: Dict[str, Any]) -> float:
    """태그 다양성 점수 (3-10개 태그가 이상적)"""
    tag_count = len(doc.get("tags", []))
    if tag_count == 0:
        return 0.0
    elif 3 <= tag_count <= 10:
        return 1.0
    elif tag_count < 3:
        return tag_count / 3.0
    else:
        # 10개 초과는 감점
        return max(0.5, 1.0 - (tag_count - 10) * 0.05)


def calculate_single_category_score(doc: Dict[str, Any]) -> float:
    """단일 분류 점수 (다른 대분류에 속하지 않을수록 높음)"""
    other_major_count = len(doc.get("other_major_categories", []))
    if other_major_count == 0:
        return 1.0
    elif other_major_count == 1:
        return 0.7
    elif other_major_count == 2:
        return 0.4
    else:
        return 0.2


def calculate_title_clarity_score(doc: Dict[str, Any], mid_category: str) -> float:
    """제목 명확성 점수 (제목에 중분류 키워드가 포함되면 높음)"""
    title = doc.get("cleaned_title", "").lower()
    mid_lower = mid_category.lower()
    
    # 중분류 키워드가 제목에 직접 포함되면 높은 점수
    if mid_lower in title:
        return 1.0
    
    # 중분류 키워드의 일부가 포함되면 중간 점수
    mid_words = mid_lower.split()
    for word in mid_words:
        if len(word) >= 2 and word in title:
            return 0.6
    
    return 0.3


def calculate_tag_relevance_score(doc: Dict[str, Any], mid_category: str) -> float:
    """태그 관련성 점수 (중분류와 관련된 태그가 많을수록 높음)"""
    tags = [tag.lower() for tag in doc.get("tags", [])]
    mid_lower = mid_category.lower()
    
    # 중분류 키워드가 태그에 직접 포함되면 높은 점수
    if any(mid_lower in tag for tag in tags):
        return 1.0
    
    # 중분류 키워드의 일부가 태그에 포함되면 중간 점수
    mid_words = mid_lower.split()
    matches = 0
    for word in mid_words:
        if len(word) >= 2:
            matches += sum(1 for tag in tags if word in tag)
    
    if matches > 0:
        return min(0.8, 0.3 + matches * 0.1)
    
    return 0.2


def calculate_total_score(
    doc: Dict[str, Any],
    mid_category: str,
    weights: Dict[str, float] = None
) -> float:
    """총점 계산"""
    if weights is None:
        weights = {
            "tag_diversity": 0.2,
            "single_category": 0.3,
            "title_clarity": 0.3,
            "tag_relevance": 0.2,
        }
    
    scores = {
        "tag_diversity": calculate_tag_diversity_score(doc),
        "single_category": calculate_single_category_score(doc),
        "title_clarity": calculate_title_clarity_score(doc, mid_category),
        "tag_relevance": calculate_tag_relevance_score(doc, mid_category),
    }
    
    total = sum(scores[key] * weights[key] for key in scores)
    return total, scores


def select_representative_docs(
    mid_category_data: Dict[str, Any],
    mid_category_name: str,
    num_repr: int = 3
) -> List[Dict[str, Any]]:
    """중분류별 대표 문서 선정"""
    documents = mid_category_data.get("documents", [])
    
    if len(documents) == 0:
        return []
    
    # 각 문서에 점수 부여
    scored_docs = []
    for doc in documents:
        total_score, score_breakdown = calculate_total_score(doc, mid_category_name)
        scored_docs.append({
            "doc": doc,
            "total_score": total_score,
            "score_breakdown": score_breakdown,
        })
    
    # 점수 순으로 정렬
    scored_docs.sort(key=lambda x: x["total_score"], reverse=True)
    
    # 상위 N개 선택
    selected = scored_docs[:num_repr]
    
    return [item["doc"] for item in selected]


def process_all_classifications(
    input_dir: Path = Path("clustering_results_tag_based"),
    output_file: Path = Path("representative_documents.json")
):
    """모든 대분류 파일을 처리하여 중분류별 대표 문서 선정"""
    
    print("=" * 80)
    print("중분류별 대표 문서 선정")
    print("=" * 80)
    
    all_representatives = {}
    total_mid_categories = 0
    total_docs_selected = 0
    
    # 모든 대분류 JSON 파일 처리
    json_files = sorted(input_dir.glob("*_classification.json"))
    
    for json_file in json_files:
        print(f"\n📂 처리 중: {json_file.name}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        major_category = data["major_category"]
        mid_categories = data.get("mid_categories", {})
        
        major_repr = {}
        
        for mid_category, mid_data in mid_categories.items():
            total_mid_categories += 1
            size = mid_data.get("size", 0)
            
            # 대표 문서 선정
            representatives = select_representative_docs(
                mid_data, mid_category, num_repr=3
            )
            
            total_docs_selected += len(representatives)
            
            major_repr[mid_category] = {
                "size": size,
                "representative_count": len(representatives),
                "representatives": representatives,
            }
            
            print(f"   ✅ {mid_category}: {size}개 문서 중 {len(representatives)}개 선정")
        
        all_representatives[major_category] = major_repr
    
    # 결과 저장
    output_data = {
        "metadata": {
            "total_major_categories": len(all_representatives),
            "total_mid_categories": total_mid_categories,
            "total_representative_docs": total_docs_selected,
        },
        "representatives": all_representatives,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ 완료!")
    print("=" * 80)
    print(f"\n📊 통계:")
    print(f"   - 대분류 수: {len(all_representatives)}개")
    print(f"   - 중분류 수: {total_mid_categories}개")
    print(f"   - 선정된 대표 문서: {total_docs_selected}개")
    print(f"\n📁 결과 파일: {output_file.absolute()}")
    
    # CSV 요약도 생성
    csv_file = output_file.parent / "representative_documents_summary.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("major_category,mid_category,size,representative_count,rec_ids\n")
        for major_cat, mid_repr in all_representatives.items():
            for mid_cat, data in mid_repr.items():
                rec_ids = ",".join([doc["rec_id"] for doc in data["representatives"]])
                f.write(
                    f'"{major_cat}","{mid_cat}",{data["size"]},{data["representative_count"]},"{rec_ids}"\n'
                )
    
    print(f"📁 CSV 요약: {csv_file.absolute()}")


if __name__ == "__main__":
    process_all_classifications()


