#!/usr/bin/env python3
"""
Ground Truth CSV → JSONL 변환 유틸리티

새로운 Ground Truth CSV 파일을 받았을 때
기존 JSONL 형태로 변환하는 도구입니다.

사용법:
    python utils/gt_converter.py input.csv output.jsonl

CSV 형식 (예상):
    query_id,query_text,ground_truth_doc_ids,user_profile_data,...

JSONL 형식:
    {"query": "...", "ground_truth_docs": ["doc1", "doc2"], "user_profile": {...}, "metadata": {...}}
"""

import csv
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class GTConverter:
    """Ground Truth CSV를 JSONL로 변환하는 클래스"""

    def __init__(self):
        self.required_columns = [
            'query',  # 쿼리 텍스트
            'ground_truth_docs'  # Ground Truth 문서 ID들
        ]

        # 사용자 프로필 관련 컬럼들 (선택사항)
        self.profile_columns = [
            'major',           # 전공
            'interest_job',    # 관심 직무
            'courses',         # 수강 이력
            'certification',   # 자격증
            'club_activities'  # 동아리/대외활동
        ]

        # 메타데이터 컬럼들 (선택사항)
        self.metadata_columns = [
            'gt_id',           # GT ID
            'company_name',    # 회사명
            'job_title',       # 채용공고 제목
            'url',            # 채용공고 URL
            'rec_idx',        # 채용공고 인덱스
            'alternative_query'  # 대체 쿼리
        ]

    def detect_csv_format(self, csv_path: str) -> Dict[str, str]:
        """CSV 파일의 컬럼 구조를 분석하여 매핑 정보 반환"""
        try:
            # 첫 몇 줄만 읽어서 컬럼 구조 파악
            df = pd.read_csv(csv_path, nrows=5)
            columns = df.columns.tolist()

            print(f"📋 CSV 파일 컬럼들: {columns}")

            # 컬럼 매핑 자동 감지
            column_mapping = {}

            # 쿼리 컬럼 감지
            query_candidates = ['query', 'query_text', 'question', 'user_input']
            for col in columns:
                if any(candidate.lower() in col.lower() for candidate in query_candidates):
                    column_mapping['query'] = col
                    break

            # Ground Truth 컬럼 감지
            gt_candidates = ['ground_truth', 'gt_docs', 'relevant_docs', 'answer_docs']
            for col in columns:
                if any(candidate.lower() in col.lower() for candidate in gt_candidates):
                    column_mapping['ground_truth_docs'] = col
                    break

            # 기타 컬럼들 자동 매핑
            for target_col in self.profile_columns + self.metadata_columns:
                for csv_col in columns:
                    if target_col.lower() in csv_col.lower():
                        column_mapping[target_col] = csv_col
                        break

            print(f"🔍 자동 감지된 컬럼 매핑: {column_mapping}")
            return column_mapping

        except Exception as e:
            print(f"❌ CSV 파일 분석 실패: {e}")
            return {}

    def interactive_column_mapping(self, csv_path: str) -> Dict[str, str]:
        """사용자와 상호작용하여 컬럼 매핑 설정"""
        auto_mapping = self.detect_csv_format(csv_path)

        # 필수 컬럼 확인
        final_mapping = {}

        # 쿼리 컬럼 매핑
        if 'query' in auto_mapping:
            query_col = auto_mapping['query']
            confirm = input(f"쿼리 컬럼으로 '{query_col}'을 사용하시겠습니까? (y/n): ").lower()
            if confirm == 'y':
                final_mapping['query'] = query_col
            else:
                query_col = input("쿼리 컬럼명을 직접 입력하세요: ")
                final_mapping['query'] = query_col
        else:
            query_col = input("쿼리 텍스트가 있는 컬럼명을 입력하세요: ")
            final_mapping['query'] = query_col

        # Ground Truth 컬럼 매핑
        if 'ground_truth_docs' in auto_mapping:
            gt_col = auto_mapping['ground_truth_docs']
            confirm = input(f"Ground Truth 컬럼으로 '{gt_col}'을 사용하시겠습니까? (y/n): ").lower()
            if confirm == 'y':
                final_mapping['ground_truth_docs'] = gt_col
            else:
                gt_col = input("Ground Truth 문서 ID 컬럼명을 직접 입력하세요: ")
                final_mapping['ground_truth_docs'] = gt_col
        else:
            gt_col = input("Ground Truth 문서 ID가 있는 컬럼명을 입력하세요: ")
            final_mapping['ground_truth_docs'] = gt_col

        # 선택적 컬럼들 매핑
        print("\n📝 선택적 컬럼들을 매핑하시겠습니까? (스킵하려면 Enter)")
        for col in self.profile_columns + self.metadata_columns:
            if col in auto_mapping:
                csv_col = auto_mapping[col]
                confirm = input(f"{col} → '{csv_col}' 매핑을 사용하시겠습니까? (y/n/Enter=스킵): ").lower()
                if confirm == 'y':
                    final_mapping[col] = csv_col
            else:
                csv_col = input(f"{col} 컬럼명 (Enter=스킵): ")
                if csv_col.strip():
                    final_mapping[col] = csv_col

        return final_mapping

    def parse_list_field(self, value: Any) -> List[str]:
        """리스트 형태의 필드를 파싱 (쉼표 구분, JSON 배열 등)"""
        if pd.isna(value) or value is None or value == '':
            return []

        if isinstance(value, str):
            # JSON 배열 형태인지 확인
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass

            # 쉼표로 구분된 값들
            return [item.strip() for item in value.split(',') if item.strip()]

        elif isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        else:
            return [str(value).strip()] if str(value).strip() else []

    def convert_row(self, row: Dict[str, Any], column_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """CSV 한 행을 JSONL 형태로 변환"""
        try:
            # 필수 필드 확인
            if 'query' not in column_mapping or 'ground_truth_docs' not in column_mapping:
                raise ValueError("필수 컬럼(query, ground_truth_docs)이 매핑되지 않았습니다")

            # 기본 구조 생성
            jsonl_row = {
                "query": str(row[column_mapping['query']]).strip(),
                "ground_truth_docs": self.parse_list_field(row[column_mapping['ground_truth_docs']])
            }

            # 사용자 프로필 생성
            user_profile = {}
            for field in self.profile_columns:
                if field in column_mapping and column_mapping[field] in row:
                    value = row[column_mapping[field]]
                    if field in ['interest_job', 'courses', 'certification', 'club_activities']:
                        user_profile[field] = self.parse_list_field(value)
                    else:
                        user_profile[field] = str(value).strip() if not pd.isna(value) else ""

            if user_profile:
                jsonl_row["user_profile"] = user_profile

            # 메타데이터 생성
            metadata = {}
            for field in self.metadata_columns:
                if field in column_mapping and column_mapping[field] in row:
                    value = row[column_mapping[field]]
                    if not pd.isna(value):
                        metadata[field] = str(value).strip()

            if metadata:
                jsonl_row["metadata"] = metadata

            return jsonl_row

        except Exception as e:
            print(f"⚠️  행 변환 실패: {e}")
            return None

    def convert_csv_to_jsonl(self, csv_path: str, jsonl_path: str, column_mapping: Optional[Dict[str, str]] = None):
        """CSV 파일을 JSONL로 변환"""

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

        # 컬럼 매핑이 제공되지 않은 경우 대화형으로 설정
        if not column_mapping:
            print("🔧 컬럼 매핑을 설정합니다...")
            column_mapping = self.interactive_column_mapping(csv_path)

        print(f"📂 변환 시작: {csv_path} → {jsonl_path}")
        print(f"🗺️  사용할 컬럼 매핑: {column_mapping}")

        # CSV 읽기 및 변환
        converted_count = 0
        error_count = 0

        try:
            df = pd.read_csv(csv_path)

            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for idx, row in df.iterrows():
                    jsonl_row = self.convert_row(row.to_dict(), column_mapping)

                    if jsonl_row:
                        f.write(json.dumps(jsonl_row, ensure_ascii=False) + '\n')
                        converted_count += 1
                    else:
                        error_count += 1

                    if (idx + 1) % 100 == 0:
                        print(f"진행률: {idx + 1}/{len(df)} 행 처리완료")

            print(f"\n✅ 변환 완료!")
            print(f"   📊 성공: {converted_count}개")
            print(f"   ❌ 실패: {error_count}개")
            print(f"   📁 출력 파일: {jsonl_path}")

        except Exception as e:
            print(f"❌ 변환 실패: {e}")
            raise

    def validate_jsonl(self, jsonl_path: str) -> bool:
        """생성된 JSONL 파일의 유효성 검사"""
        try:
            print(f"🔍 JSONL 파일 검증 중: {jsonl_path}")

            total_lines = 0
            valid_lines = 0
            sample_entries = []

            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    try:
                        entry = json.loads(line.strip())

                        # 필수 필드 확인
                        if 'query' in entry and 'ground_truth_docs' in entry:
                            valid_lines += 1

                            # 처음 3개 항목을 샘플로 저장
                            if len(sample_entries) < 3:
                                sample_entries.append(entry)

                    except json.JSONDecodeError:
                        print(f"⚠️  {i+1}번째 줄 JSON 파싱 실패")

            print(f"📊 검증 결과:")
            print(f"   전체 라인: {total_lines}")
            print(f"   유효 라인: {valid_lines}")
            print(f"   유효율: {valid_lines/total_lines*100:.1f}%")

            # 샘플 출력
            if sample_entries:
                print(f"\n📝 샘플 항목들:")
                for i, entry in enumerate(sample_entries, 1):
                    print(f"   {i}. 쿼리: {entry['query'][:50]}...")
                    print(f"      GT 문서 수: {len(entry['ground_truth_docs'])}")
                    if 'user_profile' in entry:
                        print(f"      사용자 프로필: {list(entry['user_profile'].keys())}")

            return valid_lines == total_lines

        except Exception as e:
            print(f"❌ 검증 실패: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Ground Truth CSV를 JSONL 형태로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 대화형 변환
    python utils/gt_converter.py input.csv output.jsonl

    # 컬럼 매핑 사전 정의
    python utils/gt_converter.py input.csv output.jsonl --mapping query:query_text,ground_truth_docs:gt_docs

    # 검증만 실행
    python utils/gt_converter.py --validate output.jsonl
        """
    )

    parser.add_argument('input_file', nargs='?', help='입력 CSV 파일 경로')
    parser.add_argument('output_file', nargs='?', help='출력 JSONL 파일 경로')
    parser.add_argument('--mapping', help='컬럼 매핑 (예: query:query_text,ground_truth_docs:gt_docs)')
    parser.add_argument('--validate', help='JSONL 파일 검증만 실행')

    args = parser.parse_args()

    converter = GTConverter()

    # 검증 모드
    if args.validate:
        if converter.validate_jsonl(args.validate):
            print("✅ 검증 성공!")
            sys.exit(0)
        else:
            print("❌ 검증 실패!")
            sys.exit(1)

    # 변환 모드
    if not args.input_file or not args.output_file:
        parser.print_help()
        sys.exit(1)

    # 컬럼 매핑 파싱
    column_mapping = None
    if args.mapping:
        column_mapping = {}
        for pair in args.mapping.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                column_mapping[key.strip()] = value.strip()

    try:
        # 변환 실행
        converter.convert_csv_to_jsonl(args.input_file, args.output_file, column_mapping)

        # 자동 검증
        print("\n🔍 자동 검증 실행...")
        if converter.validate_jsonl(args.output_file):
            print("🎉 변환 및 검증 완료!")
        else:
            print("⚠️  변환은 완료되었지만 일부 문제가 있습니다.")

    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()