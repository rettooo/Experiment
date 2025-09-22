#!/usr/bin/env python3
"""
간단한 데이터 로더 테스트 스크립트
"""
import os
from utils.data_loader import S3DataLoader

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

def test_s3_data_loader():
    """S3 데이터 로더 기본 기능 테스트"""

    # S3DataLoader 인스턴스 생성 (환경변수에서 bucket_name 자동 로드)
    loader = S3DataLoader()

    print("=== S3 데이터 로더 테스트 시작 ===\n")

    # 1. JSON 메타데이터 로드 테스트
    print("1. JSON 메타데이터 로드 테스트")
    print("-" * 50)
    json_prefix = "initial-dataset/json/"
    metadata_map = loader.load_json_metadata(json_prefix)
    print(f"로드된 JSON 메타데이터 개수: {len(metadata_map)}")

    if metadata_map:
        first_key = list(metadata_map.keys())[0]
        print(f"첫 번째 메타데이터 키: {first_key}")
        print(f"첫 번째 메타데이터 샘플: {list(metadata_map[first_key].keys())}")

    print()

    # 2. PDF 파일 목록 조회 테스트
    print("2. PDF 파일 목록 조회 테스트")
    print("-" * 50)
    pdf_prefix = "initial-dataset/pdf/"
    pdf_files = loader.list_s3_files(pdf_prefix)
    print(f"찾은 PDF 파일 개수: {len(pdf_files)}")
    if pdf_files:
        print(f"첫 5개 PDF 파일: {pdf_files[:5]}")

    print()

    # 3. 전체 문서 로드 테스트 (일부만)
    print("3. 문서 로드 테스트 (처음 2개만)")
    print("-" * 50)

    # 테스트를 위해 처음 2개 PDF만 처리하도록 수정
    if pdf_files:
        test_pdf_files = pdf_files[:2]  # 처음 2개만

        documents = []
        for pdf_file in test_pdf_files:
            if not pdf_file.endswith('.pdf'):
                continue

            # rec_idx 추출 (실제 서비스와 동일한 방식)
            from pathlib import Path
            pdf_filename = Path(pdf_file).name
            pdf_stem = pdf_filename.replace('.pdf', '')

            # 파일명에서 rec_idx 추출 (마지막 '_' 이후 부분)
            if '_' in pdf_stem:
                rec_idx = pdf_stem.split('_')[-1]
            else:
                rec_idx = pdf_stem

            print(f"처리 중: {pdf_filename} (rec_idx: {rec_idx})")

            # 메타데이터 확인
            if rec_idx in metadata_map:
                print(f"  ✅ 메타데이터 찾음: {metadata_map[rec_idx].get('post_title', 'N/A')}")

                # PDF 다운로드 및 텍스트 추출 테스트
                local_path = f"/tmp/{pdf_filename}"
                if loader.download_file_from_s3(pdf_file, local_path):
                    text = loader.extract_text_from_pdf(local_path)
                    print(f"  📄 추출된 텍스트 길이: {len(text)} 문자")
                    print(f"  📄 첫 100자: {text[:100]}")

                    # 파일 정리
                    if os.path.exists(local_path):
                        os.remove(local_path)
                else:
                    print(f"  ❌ PDF 다운로드 실패")
            else:
                print(f"  ⚠️  메타데이터 없음")

    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_s3_data_loader()