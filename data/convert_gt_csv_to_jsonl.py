import csv
import json
from pathlib import Path


def convert_gt_csv_to_jsonl(
    csv_path: Path,
    jsonl_path: Path,
) -> None:
    """
    GT CSV를 파이프라인에서 기대하는 JSONL 형식으로 변환한다.

    - query_id: GT_ID
    - query_text: 완전한_검색_쿼리
    - ground_truth: 같은_군집_id들을 rec_idx 리스트로 변환
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
                # 빈 행 등은 스킵
                continue

            try:
                query_id = int(gt_id_raw)
            except ValueError:
                # 숫자가 아니면 그대로 문자열로 사용
                query_id = gt_id_raw

            query_text = row.get("완전한_검색_쿼리", "").strip()

            cluster_ids_raw = row.get("같은_군집_id들", "")
            if cluster_ids_raw is None:
                cluster_ids_raw = ""

            # "50285218, 50463673, ..." 형태를 리스트로 변환
            rec_ids = [
                part.strip()
                for part in cluster_ids_raw.split(",")
                if part.strip()  # 빈 문자열 제거
            ]

            ground_truth = [{"rec_idx": rec_id} for rec_id in rec_ids]

            record = {
                "query_id": query_id,
                "query_text": query_text,
                "ground_truth": ground_truth,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"변환 완료: {count}개 레코드 → {jsonl_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "gt_evaluation_dataset_20251126_010803.csv"
    jsonl_file = base_dir / "gt_eval_fullquery_cluster_ids.jsonl"

    convert_gt_csv_to_jsonl(csv_file, jsonl_file)

