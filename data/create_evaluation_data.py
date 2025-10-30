"""
JSONL
평가용 데이터 생성

분석용
GT Analysis CSV
- queries.csv: 입력용 쿼리 데이터
- ground_truth.csv: 정답 비교용 데이터
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


# url에서 rec_idx 추출하기
def extract_rec_idx_from_url(url: str) -> str:
    if not url or not url.startswith("http"):
        return ""

    if "rec_idx=" in url:
        rec_idx = url.split("rec_idx=")[-1].split("&")[0]
        return rec_idx
    return ""


def load_grouped_data(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    GT Analysis CSV를 로드하고 query_id별로 그룹화

    Returns:
        {
            "437": {
                "query_text": "전공: 생명공학...",

                "ground_truths": [
                    {"rec_idx": "50436465", "job_title": "...", "url": "..."},
                    {"rec_idx": "50436592", "job_title": "...", "url": "..."},
                ]
            }
        }
    """
    print(f"\n📖 CSV 로드 중: {Path(csv_path).name}")
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
            # Query Text : 첫 등장시
            if not grouped_data[query_id]["query_text"]:
                query_text = row.get("완전한_검색_쿼리", "").strip()
                if not query_text:
                    skipped_rows += 1
                    continue
                grouped_data[query_id]["query_text"] = query_text
            # Ground Truth : URL에서 rec_idx 추출
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

    return dict(grouped_data)


def save_to_jsonl(data: Dict[str, Dict[str, Any]], output_path: str):
    # JSONL 형식으로 저장
    print(f"📝 평가용 데이터 저장: {Path(output_path).name}")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_queries = len(data)
    total_gt = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for query_id, query_data in sorted(
            data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0
        ):
            query_data = data[query_id]

            output_entry = {
                "query_id": query_id,
                "query_text": query_data["query_text"],
                "ground_truth": query_data["ground_truth"],
            }
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            total_gt += len(query_data["ground_truth"])
    print(f"✅ 총 {total_queries}개 쿼리, {total_gt}개 정답 문서 저장 완료")

    print(f" - 평균 정답 문서/쿼리: {total_gt/total_queries:.2f}")


def main():
    # CSV 파일 경로
    base_path = (
        "/Users/haing/Desktop/🏫2025-1/졸업 프로젝트 /fork-experiment/Experiment/data"
    )
    csv_path = f"{base_path}/GT Analysis v3.0 2025-08-25.csv"
    output_jsonl = f"{base_path}/evaluation_queries.jsonl"

    print(f"{'='*60}")
    print(f"GT Analysis CSV → 평가용 데이터 변환")
    print(f"{'='*60}")
    print(f"입력: {Path(csv_path).name}")
    print(f"출력: {Path(output_jsonl).name}")

    # 1. CSV 로드 및 그룹화
    print(f"\n1️⃣ CSV 로드 및 그룹화...")
    grouped_data = load_grouped_data(csv_path)

    if not grouped_data:
        print(f"❌ 쿼리 데이터 로드 실패")
        return

    save_to_jsonl(grouped_data, output_jsonl)

    print(f"\n{'='*60}")
    print(f"✅ 완료!")
    print(f"{'='*60}")
    print(f"📁 파일: {output_jsonl}")
    print(f"\n📊 확인:")
    print(f"  wc -l {output_jsonl}")
    print(f"  head -1 {output_jsonl} | jq .")


if __name__ == "__main__":
    main()
