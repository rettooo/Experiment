"""GT Analysis CSV -> 5개의 정답 데이터로 묶기"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


# GT csv 로드하고 gt_id별로 그룹화하기
def load_gt_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    grouped_data = defaultdict(
        lambda: {
            "query": None,
            "ground_truth_docs": [],
            "user_profile": {},
            "metadata": {},
        }
    )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt_id = row.get("GT_ID", "").strip()
            if not gt_id:
                continue

            # 쿼리 텍스트 (첫 번째 행에서만 가져오기)
            if grouped_data[gt_id]["query"] is None:
                query_text = row.get("완전한_검색_쿼리", "").strip()
                grouped_data[gt_id]["query"] = query_text

                # 사용자 프로필 파싱
                major = row.get("학생_전공", "").strip()
                interest_job = row.get("학생_관심분야", "").strip()
                courses = row.get("수강과목", "").strip()

                grouped_data[gt_id]["user_profile"] = {
                    "major": major if major else None,
                    "interest_job": interest_job.split(", ") if interest_job else [],
                    "courses": courses.split(", ") if courses else [],
                    "certification": [],
                    "club_activities": [],
                }

                # 메타데이터
                grouped_data[gt_id]["metadata"]["gt_id"] = gt_id
                grouped_data[gt_id]["metadata"]["alternative_query"] = row.get(
                    "학생_질문", ""
                ).strip()

            # Ground truth 문서 추가 (URL에서 rec_idx 추출)
            url = row.get("URL", "").strip()

            if url and url.startswith("http"):
                # URL에서 rec_idx 추출
                if "rec_idx=" in url:
                    rec_idx = url.split("rec_idx=")[-1].split("&")[0]
                    grouped_data[gt_id]["ground_truth_docs"].append(rec_idx)
                else:
                    print(
                        f"⚠️  GT_ID {gt_id}: rec_idx를 찾을 수 없는 URL: {url[:50]}..."
                    )

            # 메타데이터 업데이트 (첫 번째 공고 정보만, URL 제외)
            if "company_name" not in grouped_data[gt_id]["metadata"]:
                grouped_data[gt_id]["metadata"]["company_name"] = row.get(
                    "회사명", ""
                ).strip()
                grouped_data[gt_id]["metadata"]["job_title"] = row.get(
                    "공고_제목", ""
                ).strip()

    return dict(grouped_data)


def convert_to_jsonl(grouped_data: Dict[str, Dict[str, Any]], output_path: str):
    """그룹화된 데이터를 JSONL로 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        for gt_id, data in sorted(
            grouped_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0
        ):
            # 정답이 없는 쿼리는 제외
            if not data["ground_truth_docs"]:
                print(f"⚠️  GT_ID {gt_id}: 정답 문서 없음, 제외")
                continue

            # 쿼리가 없는 경우 제외
            if not data["query"]:
                print(f"⚠️  GT_ID {gt_id}: 쿼리 텍스트 없음, 제외")
                continue

            # JSONL 형식으로 저장
            entry = {
                "query": data["query"],
                "ground_truth_docs": data["ground_truth_docs"],
                "user_profile": data["user_profile"],
                "metadata": data["metadata"],
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def validate_jsonl(jsonl_path: str):
    """생성된 JSONL 파일 검증"""

    print(f"\n{'='*60}")
    print(f"📊 JSONL 파일 검증: {jsonl_path}")
    print(f"{'='*60}")

    total_queries = 0
    total_gt_docs = 0
    gt_doc_counts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            entry = json.loads(line)
            total_queries += 1
            gt_count = len(entry["ground_truth_docs"])
            total_gt_docs += gt_count
            gt_doc_counts.append(gt_count)

            # 첫 3개 샘플 출력
            if i <= 3:
                print(f"\n[쿼리 {i}]")
                print(f"  GT_ID: {entry['metadata']['gt_id']}")
                print(f"  쿼리 길이: {len(entry['query'])} 자")
                print(f"  정답 문서 수: {gt_count}")
                print(f"  정답 문서 IDs: {entry['ground_truth_docs'][:3]}...")
                print(f"  회사명: {entry['metadata'].get('company_name', 'N/A')}")
                print(
                    f"  공고 제목: {entry['metadata'].get('job_title', 'N/A')[:50]}..."
                )
                print(f"  전공: {entry['user_profile']['major']}")

    avg_gt_docs = total_gt_docs / total_queries if total_queries > 0 else 0
    min_gt = min(gt_doc_counts) if gt_doc_counts else 0
    max_gt = max(gt_doc_counts) if gt_doc_counts else 0

    print(f"\n{'='*60}")
    print(f"📈 통계")
    print(f"{'='*60}")
    print(f"총 쿼리 수: {total_queries}")
    print(f"총 정답 문서 수: {total_gt_docs}")
    print(f"평균 정답 문서/쿼리: {avg_gt_docs:.2f}")
    print(f"최소 정답 문서: {min_gt}")
    print(f"최대 정답 문서: {max_gt}")

    # 정답 개수별 분포
    from collections import Counter

    dist = Counter(gt_doc_counts)
    print(f"\n정답 문서 개수 분포:")
    for count, freq in sorted(dist.items()):
        print(f"  {count}개 정답: {freq}개 쿼리 ({freq/total_queries*100:.1f}%)")


def main():
    """메인 실행 함수"""

    csv_path = "Experiment/data/GT Analysis v3.0 2025-08-25.csv"
    output_path = "Experiment/data/test_queries_fixed_v2.jsonl"

    print(f"{'='*60}")
    print(f"GT CSV → JSONL 변환 (URL 제외 버전)")
    print(f"{'='*60}")
    print(f"입력: {csv_path}")
    print(f"출력: {output_path}")

    # 1. CSV 로드 및 그룹화
    print(f"\n1️⃣  CSV 로드 및 GT_ID별 그룹화...")
    grouped_data = load_gt_csv(csv_path)
    print(f"   ✅ {len(grouped_data)}개 고유 쿼리 로드 완료")

    # 2. JSONL로 변환
    print(f"\n2️⃣  JSONL 변환 중...")
    convert_to_jsonl(grouped_data, output_path)
    print(f"   ✅ {output_path} 생성 완료")

    # 3. 검증
    print(f"\n3️⃣  생성된 파일 검증...")
    validate_jsonl(output_path)

    print(f"\n{'='*60}")
    print(f"✅ 변환 완료!")
    print(f"{'='*60}")
    print(f"\n다음 명령어로 새 파일을 사용하세요:")
    print(f"  mv data/test_queries_fixed.jsonl data/test_queries_fixed_old.jsonl")
    print(f"  mv data/test_queries_fixed_v2.jsonl data/test_queries_fixed.jsonl")


if __name__ == "__main__":
    main()
