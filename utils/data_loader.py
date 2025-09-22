"""
S3 데이터 로더: 현재 서비스와 동일한 방식으로 S3에서 데이터 로드
"""

import boto3
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF


class S3DataLoader:
    """S3에서 채용공고 PDF와 JSON 메타데이터를 로드하는 클래스"""

    def __init__(self, bucket_name: str | None = None):
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'career-hi')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-2')
        )

    def list_s3_files(self, prefix: str) -> List[str]:
        """S3 버킷에서 특정 prefix로 시작하는 파일 목록 조회 (페이지네이션 지원)"""
        try:
            files = []
            continuation_token = None

            while True:
                kwargs = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix
                }

                if continuation_token:
                    kwargs['ContinuationToken'] = continuation_token

                response = self.s3_client.list_objects_v2(**kwargs)

                if 'Contents' in response:
                    files.extend([obj['Key'] for obj in response['Contents']])

                # 더 많은 파일이 있는지 확인
                if response.get('IsTruncated', False):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

            return files

        except Exception as e:
            print(f"S3 파일 목록 조회 실패: {e}")
            return []

    def download_file_from_s3(self, s3_key: str, local_path: str) -> bool:
        """S3에서 파일을 로컬로 다운로드"""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"S3 파일 다운로드 실패 ({s3_key}): {e}")
            return False

    def load_json_metadata(self, json_prefix: str) -> Dict[str, Dict[str, Any]]:
        """JSON 메타데이터 파일들을 로드하여 rec_idx로 매핑"""
        json_files = self.list_s3_files(json_prefix)
        print(f"JSON prefix '{json_prefix}'에서 찾은 파일 수: {len(json_files)}")
        if len(json_files) > 0:
            print(f"첫 5개 파일: {json_files[:5]}")
        metadata_map = {}

        for json_file in json_files:
            # 실제 JSON 파일만 처리 (디렉토리 제외)
            if not json_file.endswith('.json'):
                continue
            try:
                # S3에서 JSON 파일 내용 읽기
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=json_file)
                json_content = response['Body'].read().decode('utf-8')
                json_data = json.loads(json_content)

                # rec_idx를 키로 하는 매핑 생성
                if 'rec_idx' in json_data:
                    metadata_map[str(json_data['rec_idx'])] = json_data

            except Exception as e:
                print(f"JSON 파일 로드 실패 ({json_file}): {e}")
                continue

        print(f"총 {len(metadata_map)}개의 JSON 메타데이터 로드됨")
        return metadata_map

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PyMuPDF를 사용하여 PDF에서 텍스트 추출 (현재 서비스와 동일)"""
        text_list = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_list.append(f"\n--- Page {page_num} ---\n")
                    text_list.append(page.get_text())
            return ''.join(text_list)
        except Exception as e:
            print(f"PDF 텍스트 추출 실패 ({pdf_path}): {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """텍스트 정제 (현재 서비스와 동일한 방식)"""
        import re

        # 현재 서비스의 정제 로직과 동일
        patterns_to_remove = [
            r'최저임금.*?원',
            r'조회수.*?\d+',
            r'신고.*?바로가기',
            # 추가적인 정제 패턴들을 여기에 추가
        ]

        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

        return cleaned_text.strip()

    def load_documents(
        self,
        pdf_prefix: str,
        json_prefix: str
    ) -> List[Dict[str, Any]]:
        """
        S3에서 PDF와 JSON 데이터를 로드하여 문서 리스트 생성

        Returns:
            문서 리스트, 각 문서는 {'text': str, 'metadata': dict} 형태
        """
        print("JSON 메타데이터 로드 중...")
        metadata_map = self.load_json_metadata(json_prefix)

        print("PDF 파일 목록 조회 중...")
        pdf_files = self.list_s3_files(pdf_prefix)
        print(f"PDF prefix '{pdf_prefix}'에서 찾은 파일 수: {len(pdf_files)}")
        if len(pdf_files) > 0:
            print(f"첫 5개 파일: {pdf_files[:5]}")

        documents = []
        processed_count = 0

        for pdf_file in pdf_files:
            # 실제 PDF 파일만 처리 (디렉토리 제외)
            if not pdf_file.endswith('.pdf'):
                continue

            # PDF 파일명에서 rec_idx 추출 (실제 서비스와 동일한 방식)
            pdf_filename = Path(pdf_file).name
            pdf_stem = pdf_filename.replace('.pdf', '')

            # 파일명에서 rec_idx 추출 (마지막 '_' 이후 부분)
            if '_' in pdf_stem:
                rec_idx = pdf_stem.split('_')[-1]
            else:
                rec_idx = pdf_stem

            # 해당하는 JSON 메타데이터 찾기
            if rec_idx not in metadata_map:
                print(f"메타데이터 없음, 기본 메타데이터 사용: {rec_idx}")
                # 기본 메타데이터 생성
                basic_metadata = {
                    "rec_idx": rec_idx,
                    "title": f"문서_{rec_idx}",
                    "company": "알 수 없음",
                    "source": "s3_pdf_only"
                }
                metadata_map[rec_idx] = basic_metadata

            # PDF 파일을 임시로 다운로드
            local_pdf_path = f"/tmp/{pdf_filename}"
            if not self.download_file_from_s3(pdf_file, local_pdf_path):
                continue

            try:
                # PDF에서 텍스트 추출
                raw_text = self.extract_text_from_pdf(local_pdf_path)
                if not raw_text.strip():
                    print(f"텍스트 추출 실패: {rec_idx}")
                    continue

                # 텍스트 정제
                cleaned_text = self.clean_text(raw_text)

                # 메타데이터 준비
                metadata = {
                    **metadata_map[rec_idx],
                    "source": "s3",
                    "pdf_file": pdf_file,
                    "rec_idx": rec_idx
                }

                documents.append({
                    "text": cleaned_text,
                    "metadata": metadata
                })

                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"처리된 문서 수: {processed_count}")

            except Exception as e:
                print(f"문서 처리 실패 ({rec_idx}): {e}")
                continue

            finally:
                # 임시 파일 삭제
                if os.path.exists(local_pdf_path):
                    os.remove(local_pdf_path)

        print(f"총 {len(documents)}개 문서 로드 완료")
        return documents