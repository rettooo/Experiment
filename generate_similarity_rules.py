#!/usr/bin/env python3
"""
유사도 규칙 파일 생성 및 검증 도구

사용법:
1. 템플릿 파일을 기반으로 규칙 생성
2. 실제 데이터와 비교하여 검증
3. 수동으로 규칙 추가/수정
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

def load_classification_data():
    """분류 데이터 로드"""
    with open("clustering_results_tag_based/all_classifications.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_rules_template():
    """규칙 템플릿 로드"""
    template_file = Path("similarity_rules_template.json")
    if template_file.exists():
        with open(template_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def validate_rules(rules, classification_data):
    """
    규칙 검증:
    1. 규칙에 있는 중분류가 실제 데이터에 존재하는지
    2. 규칙에 없는 중분류가 있는지
    3. 중복된 중분류가 있는지
    """
    print("=" * 80)
    print("유사도 규칙 검증")
    print("=" * 80)
    
    all_rule_mids = set()
    errors = []
    warnings = []
    
    for major_cat, groups in rules.items():
        if major_cat not in classification_data["classifications"]:
            errors.append(f"❌ 대분류 '{major_cat}'가 데이터에 없습니다")
            continue
        
        actual_mids = set(classification_data["classifications"][major_cat]["mid_category_distribution"].keys())
        
        for group_name, mid_list in groups.items():
            for mid in mid_list:
                # 중복 체크
                if mid in all_rule_mids:
                    warnings.append(f"⚠️  중분류 '{mid}'가 여러 그룹에 포함됨")
                all_rule_mids.add(mid)
                
                # 존재 여부 체크
                if mid not in actual_mids:
                    errors.append(f"❌ '{major_cat}'의 중분류 '{mid}'가 데이터에 없습니다")
        
        # 규칙에 없는 중분류 체크
        missing_mids = actual_mids - all_rule_mids
        if missing_mids:
            # "기타"는 제외
            missing_mids = {m for m in missing_mids if m != "기타"}
            if missing_mids:
                warnings.append(f"⚠️  '{major_cat}'의 다음 중분류가 규칙에 없습니다: {sorted(missing_mids)[:10]}")
    
    # 결과 출력
    if errors:
        print("\n❌ 오류:")
        for error in errors[:20]:  # 최대 20개만
            print(f"   {error}")
        if len(errors) > 20:
            print(f"   ... 외 {len(errors)-20}개 오류")
    
    if warnings:
        print("\n⚠️  경고:")
        for warning in warnings[:20]:  # 최대 20개만
            print(f"   {warning}")
        if len(warnings) > 20:
            print(f"   ... 외 {len(warnings)-20}개 경고")
    
    if not errors and not warnings:
        print("\n✅ 모든 규칙이 유효합니다!")
    
    return len(errors) == 0

def analyze_coverage(rules, classification_data):
    """규칙 커버리지 분석"""
    print("\n" + "=" * 80)
    print("규칙 커버리지 분석")
    print("=" * 80)
    
    total_mids = 0
    covered_mids = 0
    
    for major_cat, data in classification_data["classifications"].items():
        actual_mids = set(data["mid_category_distribution"].keys())
        total_mids += len(actual_mids)
        
        if major_cat in rules:
            rule_mids = set()
            for mid_list in rules[major_cat].values():
                rule_mids.update(mid_list)
            
            covered = len(actual_mids & rule_mids)
            covered_mids += covered
            
            coverage = covered / len(actual_mids) * 100 if actual_mids else 0
            print(f"\n{major_cat}:")
            print(f"   전체 중분류: {len(actual_mids)}개")
            print(f"   규칙 포함: {covered}개 ({coverage:.1f}%)")
            if coverage < 100:
                missing = actual_mids - rule_mids
                missing = {m for m in missing if m != "기타"}
                if missing:
                    print(f"   미포함: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    
    overall_coverage = covered_mids / total_mids * 100 if total_mids > 0 else 0
    print(f"\n{'='*80}")
    print(f"전체 커버리지: {covered_mids}/{total_mids}개 ({overall_coverage:.1f}%)")
    print(f"{'='*80}")

def generate_cluster_summary(rules, classification_data):
    """규칙 기반 클러스터 요약 생성"""
    print("\n" + "=" * 80)
    print("규칙 기반 클러스터 요약")
    print("=" * 80)
    
    cluster_summary = {}
    
    for major_cat, groups in rules.items():
        if major_cat not in classification_data["classifications"]:
            continue
        
        mid_dist = classification_data["classifications"][major_cat]["mid_category_distribution"]
        
        cluster_summary[major_cat] = {}
        
        for group_name, mid_list in groups.items():
            # 그룹 내 문서 수 계산
            total_docs = 0
            for mid in mid_list:
                if mid in mid_dist:
                    total_docs += mid_dist[mid]
            
            cluster_summary[major_cat][group_name] = {
                "mid_categories": mid_list,
                "num_mid_categories": len(mid_list),
                "estimated_docs": total_docs  # 중복 포함 추정치
            }
    
    # 요약 출력
    total_clusters = 0
    for major_cat, clusters in cluster_summary.items():
        print(f"\n{major_cat}:")
        print(f"   클러스터 수: {len(clusters)}개")
        total_clusters += len(clusters)
        
        for cluster_name, info in sorted(clusters.items(), key=lambda x: x[1]["estimated_docs"], reverse=True)[:5]:
            print(f"      - {cluster_name}: {info['num_mid_categories']}개 중분류, 약 {info['estimated_docs']}개 문서")
    
    print(f"\n{'='*80}")
    print(f"전체 예상 클러스터 수: {total_clusters}개")
    print(f"{'='*80}")
    
    return cluster_summary

def main():
    """메인 함수"""
    print("=" * 80)
    print("유사도 규칙 파일 생성 및 검증")
    print("=" * 80)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    classification_data = load_classification_data()
    rules = load_rules_template()
    
    if not rules:
        print("❌ 규칙 템플릿 파일을 찾을 수 없습니다.")
        print("   similarity_rules_template.json 파일을 먼저 생성하세요.")
        return
    
    print(f"   ✅ {len(rules)}개 대분류 규칙 로드 완료")
    
    # 검증
    print("\n🔍 규칙 검증 중...")
    is_valid = validate_rules(rules, classification_data)
    
    # 커버리지 분석
    analyze_coverage(rules, classification_data)
    
    # 클러스터 요약
    cluster_summary = generate_cluster_summary(rules, classification_data)
    
    # 최종 규칙 파일 저장
    output_file = Path("similarity_rules.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 최종 규칙 파일 저장: {output_file}")
    
    if is_valid:
        print("\n✅ 규칙 파일이 준비되었습니다!")
        print("   다음 단계: GT 생성 파이프라인 실행")
    else:
        print("\n⚠️  규칙에 오류가 있습니다. 수정 후 다시 실행하세요.")

if __name__ == "__main__":
    main()



