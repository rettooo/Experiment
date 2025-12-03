"""
StructuredDocumentLoader: JobPostParser 기반 구조화 문서 로더

섹션별 청크 생성 및 ChromaDB 메타데이터 관리
"""

import pickle
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from implementations.parsers.job_post_parser import (
    JobPostParser,
)  # unstructured 기반 파서


class Chunk:
    """
    청크 데이터 클래스
    text + metadata로 구성
    metadata는 ChromaDB 저장용으로 설계됨
    """

    def __init__(self, text: str, metadata: Dict[str, Any]):
        """
        Args:
            text: 청크 텍스트 (섹션 내용)
            metadata: ChromaDB 저장용 메타데이터
                - chunk_id (str): 고유 ID
                - rec_idx (str): 문서 ID
                - company (str): 회사명
                - title (str): 직무명
                - url (str): 상세 URL
                - section_type (str): 섹션 타입
                - section_length (int): 섹션 길이
                - tags (list): 태그 리스트
        """
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {"text": self.text, "metadata": self.metadata}

    def __repr__(self):
        return f"Chunk(chunk_id={self.metadata.get('chunk_id')}, section={self.metadata.get('section_type')}, length={len(self.text)})"


class StructuredDocumentLoader:
    """
    JobPostParser를 활용한 구조화 문서 로더

    특징:
    - 섹션별 청크 생성 (preferred, qualifications, job_duties)
    - ChromaDB 호환 메타데이터 자동 생성
    - 태그는 문서 레벨에서 추출 (모든 청크에 공유)
    - benefits, hiring_process, notes는 제외
    """

    # 임베딩 대상 섹션 (나머지는 제외)
    DEFAULT_TARGET_SECTIONS = ["preferred", "qualifications", "job_duties"]

    def __init__(
        self,
        strategy: str = "fast",
        target_sections: Optional[List[str]] = None,
        include_context: bool = True,
    ):
        """
        Args:
            strategy: unstructured 파싱 전략 ("fast", "hi_res")
            target_sections: 로드할 섹션 타입 리스트
                None이면 DEFAULT_TARGET_SECTIONS 사용
            include_context: JobPostParser의 context 주입 여부
        """
        self.parser = JobPostParser(strategy=strategy, include_context=include_context)
        self.target_sections = target_sections or self.DEFAULT_TARGET_SECTIONS
        self.chunks: List[Chunk] = []

        print(f"✅ StructuredDocumentLoader 초기화")
        print(f"   - 파싱 전략: {strategy}")
        print(f"   - 대상 섹션: {', '.join(self.target_sections)}")
        print(f"   - Context 주입: {include_context}")

    @staticmethod
    def extract_company_and_title(text: str) -> tuple:
        """
        캐시된 텍스트에서 회사명과 직무명 추출

        패턴:
        - 회사명: "관심기업" 키워드 바로 앞 줄
        - 직무명: "관심기업" 키워드 바로 뒤 줄

        Args:
            text: 문서 전체 텍스트

        Returns:
            (company, title) 튜플
        """
        lines = text.split("\n")
        company = "미상"
        title = "미상"

        try:
            # 회사명 추출: "관심기업" 전까지의 첫 번째 의미있는 줄
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or "---" in line or "Page" in line:
                    continue

                # "관심기업" 키워드를 찾으면 그 전 줄이 회사명
                if "관심기업" in line:
                    # 이전 줄들 중 의미있는 줄 찾기
                    for j in range(i - 1, -1, -1):
                        prev_line = lines[j].strip()
                        if (
                            prev_line
                            and "---" not in prev_line
                            and "Page" not in prev_line
                        ):
                            company = prev_line
                            break

                    # 직무명: "관심기업" 다음 의미있는 줄
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and len(next_line) > 5:  # 충분히 긴 줄
                            title = next_line
                            break
                    break

            # "관심기업" 키워드가 없는 경우, 처음 나오는 의미있는 2개 줄
            if company == "미상":
                meaningful_lines = []
                for line in lines:
                    line = line.strip()
                    if (
                        line
                        and "---" not in line
                        and "Page" not in line
                        and len(line) > 2
                    ):
                        meaningful_lines.append(line)
                        if len(meaningful_lines) >= 2:
                            break

                if len(meaningful_lines) >= 1:
                    company = meaningful_lines[0]
                if len(meaningful_lines) >= 2:
                    title = meaningful_lines[1]

        except Exception as e:
            print(f"    ⚠️ 회사/직무 추출 실패: {e}")

        return company, title

    def load_from_cache(
        self, cache_file: str, limit: Optional[int] = None, save_progress: bool = True
    ) -> List[Chunk]:
        """
        캐시된 S3 데이터에서 문서 로드 및 파싱

        Args:
            cache_file: documents.pkl 경로
            limit: 처리할 문서 수 제한 (None이면 전체)
            save_progress: 진행상황 중간 저장 여부

        Returns:
            청크 리스트
        """
        print(f"\n{'='*80}")
        print(f"📂 캐시 파일 로드: {cache_file}")
        print(f"{'='*80}")

        # 캐시 로드
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)

        total_docs = len(cached_data)
        process_count = limit if limit else total_docs

        print(f"✅ 총 문서: {total_docs}개")
        print(f"🎯 처리 대상: {process_count}개")
        print(f"\n{'='*80}")
        print(f"📋 문서 파싱 시작")
        print(f"{'='*80}\n")

        # 문서 파싱
        self.chunks = []
        failed_docs = []

        # 캐시 데이터가 리스트인 경우
        if isinstance(cached_data, list):
            docs_to_process = cached_data[:process_count]
        else:
            # 딕셔너리인 경우 (하위 호환성)
            docs_to_process = list(cached_data.items())[:process_count]

        for idx, doc_item in enumerate(tqdm(docs_to_process, desc="문서 파싱")):
            try:
                # 리스트 형태: {'text': ..., 'metadata': {...}}
                if isinstance(doc_item, dict) and "metadata" in doc_item:
                    doc_metadata = doc_item["metadata"]
                    raw_text = doc_item.get("text", "")
                    rec_id = doc_metadata.get("rec_idx", f"unknown_{idx}")

                    # raw_text에서 회사명과 직무명 추출
                    company, title = self.extract_company_and_title(raw_text)

                    # 메타데이터 추출
                    metadata = {
                        "rec_idx": str(rec_id),
                        "company": company,
                        "title": title,
                        "url": doc_metadata.get(
                            "detail_url",
                            f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_id}",
                        ),
                    }
                # 딕셔너리 형태 (하위 호환성): (rec_id, {data})
                else:
                    rec_id, doc_data = doc_item
                    raw_text = doc_data.get("text", "")

                    # raw_text에서 회사명과 직무명 추출
                    company, title = self.extract_company_and_title(raw_text)

                    metadata = {
                        "rec_idx": str(rec_id),
                        "company": company,
                        "title": title,
                        "url": doc_data.get(
                            "detail_url",
                            f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_id}",
                        ),
                    }

                # 텍스트 검증
                if not raw_text or len(raw_text) < 50:
                    failed_docs.append((rec_id, "텍스트 없음"))
                    continue

                # 임시 파일로 저장하여 파싱
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tmp_file:
                    tmp_file.write(raw_text)
                    tmp_path = tmp_file.name

                # JobPostParser로 파싱
                parsed_chunks = self.parser.process_document(
                    doc_path=tmp_path, original_metadata=metadata
                )

                # 임시 파일 삭제
                Path(tmp_path).unlink()

                # 섹션 필터링 및 Chunk 객체 생성 (raw_text 전달)
                doc_chunks = self._create_chunks_from_parsed(
                    parsed_chunks, base_metadata=metadata, raw_text=raw_text
                )

                self.chunks.extend(doc_chunks)

            except Exception as e:
                failed_docs.append((rec_id, str(e)))
                continue

        # 결과 출력
        print(f"\n{'='*80}")
        print(f"✅ 파싱 완료")
        print(f"{'='*80}")
        print(f"📊 총 청크 수: {len(self.chunks)}개")
        print(f"❌ 실패 문서: {len(failed_docs)}개")

        if failed_docs and len(failed_docs) <= 10:
            print(f"\n실패 문서 목록:")
            for rec_id, reason in failed_docs:
                print(f"  - {rec_id}: {reason}")
        return self.chunks

    def load_from_documents(
        self, documents: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[Chunk]:
        """
        메모리에 이미 로드된 documents 리스트에서 바로 파싱 및 청크 생성

        Args:
            documents: S3DataLoader.load_documents() 형태의 리스트
                - 각 원소: {"text": str, "metadata": dict(rec_idx, detail_url, ...)}
            limit: 처리할 문서 수 제한 (None이면 전체)

        Returns:
            Chunk 객체 리스트
        """
        print(f"\n{'='*80}")
        print(
            f"📂 메모리 문서 리스트에서 구조화 파싱 시작 (JobPostParser + unstructured)"
        )
        print(f"{'='*80}")

        total_docs = len(documents)
        process_count = limit if limit else total_docs

        print(f"✅ 총 문서: {total_docs}개")
        print(f"🎯 처리 대상: {process_count}개")
        print(f"\n{'='*80}")
        print(f"📋 문서 파싱 시작 (in-memory)")
        print(f"{'='*80}\n")

        self.chunks = []
        failed_docs = []

        docs_to_process = documents[:process_count]

        for idx, doc_item in enumerate(tqdm(docs_to_process, desc="문서 파싱")):
            try:
                # S3DataLoader.load_documents() 포맷 가정
                doc_metadata = doc_item.get("metadata", {})
                raw_text = doc_item.get("text", "")
                rec_id = doc_metadata.get("rec_idx", f"unknown_{idx}")

                # raw_text에서 회사명과 직무명 추출 (메타데이터에 없을 때만)
                company, title = self.extract_company_and_title(raw_text)

                # 원본 메타데이터를 최대한 보존하면서 필요한 필드 추가/보완
                # 표준 필드명과 원본 필드명 모두 보존
                base_metadata = {
                    **doc_metadata,  # 원본 메타데이터 전체 보존 (deadline, start_date, crawling_time 등 포함)
                    "rec_idx": str(rec_id),  # rec_idx는 확실히 설정
                    # 표준 필드명 설정 (원본 필드명도 함께 보존)
                    "title": doc_metadata.get("title")
                    or doc_metadata.get("post_title")
                    or title,
                    "company": doc_metadata.get("company")
                    or doc_metadata.get("company_name")
                    or company,
                    "url": doc_metadata.get("url")
                    or doc_metadata.get(
                        "detail_url",
                        f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_id}",
                    ),
                    # 원본 필드명도 명시적으로 보존 (하위 호환성)
                    "post_title": doc_metadata.get("post_title")
                    or doc_metadata.get("title")
                    or title,
                    "company_name": doc_metadata.get("company_name")
                    or doc_metadata.get("company")
                    or company,
                    "detail_url": doc_metadata.get("detail_url")
                    or doc_metadata.get("url")
                    or "",
                }

                # 디버깅: 첫 번째 문서의 메타데이터 확인
                if idx == 0:
                    print(f"\n🔍 첫 번째 문서 메타데이터 확인 (rec_idx: {rec_id}):")
                    print(f"   - deadline: {doc_metadata.get('deadline')}")
                    print(f"   - start_date: {doc_metadata.get('start_date')}")
                    print(f"   - crawling_time: {doc_metadata.get('crawling_time')}")
                    print(f"   - post_title: {doc_metadata.get('post_title')}")
                    print(f"   - company_name: {doc_metadata.get('company_name')}")
                    print(f"   - 전체 메타데이터 키: {list(doc_metadata.keys())[:20]}")

                # S3 JSON 메타데이터 필드명 매핑 (필요한 경우)
                # JSON 파일에서 사용하는 필드명과 내부에서 사용하는 필드명이 다를 수 있음
                # **doc_metadata로 이미 포함되어 있지만, 명시적으로 확인
                # deadline, start_date, crawling_time은 원본 메타데이터에서 가져오기
                if (
                    "deadline" not in base_metadata
                    or base_metadata.get("deadline") is None
                ):
                    base_metadata["deadline"] = doc_metadata.get("deadline")
                if (
                    "start_date" not in base_metadata
                    or base_metadata.get("start_date") is None
                ):
                    base_metadata["start_date"] = doc_metadata.get("start_date")
                if (
                    "crawling_time" not in base_metadata
                    or base_metadata.get("crawling_time") is None
                ):
                    base_metadata["crawling_time"] = doc_metadata.get("crawling_time")

                # 텍스트 검증
                if not raw_text or len(raw_text) < 50:
                    failed_docs.append((rec_id, "텍스트 없음"))
                    continue

                # 임시 파일로 저장하여 JobPostParser 재사용 (unstructured 기반)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as tmp_file:
                    tmp_file.write(raw_text)
                    tmp_path = tmp_file.name

                parsed_chunks = self.parser.process_document(
                    doc_path=tmp_path, original_metadata=base_metadata
                )

                # 임시 파일 삭제
                Path(tmp_path).unlink()

                # 섹션 필터링 및 Chunk 객체 생성 (fallback 포함)
                doc_chunks = self._create_chunks_from_parsed(
                    parsed_chunks, base_metadata=base_metadata, raw_text=raw_text
                )

                self.chunks.extend(doc_chunks)

            except Exception as e:
                error_msg = str(e)
                import traceback

                if idx < 3:  # 처음 3개만 상세 에러 출력
                    print(f"\n❌ 문서 {idx} 파싱 실패 (rec_idx: {rec_id}):")
                    print(f"   에러: {error_msg}")
                    print(f"   상세:")
                    traceback.print_exc()
                failed_docs.append((rec_id, error_msg))
                continue

        print(f"\n{'='*80}")
        print(f"✅ in-memory 파싱 완료")
        print(f"{'='*80}")
        print(f"📊 총 청크 수: {len(self.chunks)}개")
        print(f"❌ 실패 문서: {len(failed_docs)}개")

        if failed_docs:
            print(f"\n실패 문서 목록 (최대 20개):")
            for rec_id, reason in failed_docs[:20]:
                print(f"  - {rec_id}: {reason}")
            if len(failed_docs) > 20:
                print(f"  ... 외 {len(failed_docs) - 20}개 실패")

        return self.chunks

    def load_and_chunk(
        self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        단일 PDF 파일 파싱 및 청크 생성

        Args:
            pdf_path: PDF 파일 경로
            metadata: 기본 메타데이터
                - rec_idx (필수)
                - company, title, url (선택)

        Returns:
            청크 리스트
        """
        if metadata is None:
            metadata = {}

        # rec_idx 필수 체크 (rec_id도 호환성 위해 허용)
        rec_idx = metadata.get("rec_idx") or metadata.get("rec_id")
        if not rec_idx:
            raise ValueError("metadata에 'rec_idx' 또는 'rec_id' 필수")

        # 기본값 설정
        base_metadata = {
            "rec_idx": str(rec_idx),
            "company": metadata.get("company", "미상"),
            "title": metadata.get("title", "미상"),
            "url": metadata.get(
                "url",
                f"https://www.career.go.kr/cnet/front/extr/extrJobView.do?rec_idx={rec_idx}",
            ),
        }

        # JobPostParser로 파싱
        parsed_chunks = self.parser.process_document(
            doc_path=pdf_path, original_metadata=base_metadata
        )

        # Chunk 객체 생성
        doc_chunks = self._create_chunks_from_parsed(
            parsed_chunks, base_metadata=base_metadata
        )

        self.chunks.extend(doc_chunks)

        return doc_chunks

    def _create_fallback_chunk(
        self, raw_text: str, base_metadata: Dict[str, Any], tags: List[str]
    ) -> List[Chunk]:
        """
        Target 섹션이 없을 때 fallback 청크 생성

        하이브리드 접근:
        - 900자 이하: 경량 컨텍스트 (회사+직무+태그만)
        - 900자 초과: 400자 단위 청킹

        Args:
            raw_text: 문서 원본 텍스트
            base_metadata: 기본 메타데이터 (rec_idx, company, title, url)
            tags: 문서 태그

        Returns:
            Fallback Chunk 객체 리스트
        """
        # 전처리된 텍스트 길이 확인
        max_raw_length = 2000
        truncated_raw_text = raw_text[:max_raw_length]

        # Context 추가한 전체 텍스트
        full_text_with_context = (
            f"[회사: {base_metadata['company']}] "
            f"[직무: {base_metadata['title']}]\n\n"
            f"{truncated_raw_text}"
        )

        text_length = len(full_text_with_context)

        # 900자 이하: 경량 컨텍스트 생성
        if text_length <= 900:
            lightweight_text = f"회사: {base_metadata['company']}\n"
            lightweight_text += f"직무: {base_metadata['title']}\n"

            if tags:
                tags_str = ", ".join(tags[:5])
                if len(tags) > 5:
                    tags_str += f" 외 {len(tags)-5}개"
                lightweight_text += f"태그: {tags_str}"

            chunk_metadata = {
                **base_metadata,  # 원본 S3 JSON 메타데이터 전체 포함
                "chunk_id": f"{base_metadata['rec_idx']}_full_text_0",
                "rec_idx": base_metadata["rec_idx"],
                "company": base_metadata.get("company")
                or base_metadata.get("company_name"),
                "title": base_metadata.get("title") or base_metadata.get("post_title"),
                "url": base_metadata.get("url") or base_metadata.get("detail_url"),
                "section_type": "full_text",
                "section_length": len(lightweight_text),
                "tags": tags,
                "is_fallback": True,
                "is_lightweight": True,
                # deadline, start_date, crawling_time 명시적으로 보장
                "deadline": base_metadata.get("deadline"),
                "start_date": base_metadata.get("start_date"),
                "crawling_time": base_metadata.get("crawling_time"),
            }

            return [Chunk(text=lightweight_text, metadata=chunk_metadata)]

        # 900자 초과: 400자 단위 청킹
        else:
            chunks = []
            chunk_size = 400
            num_chunks = (text_length + chunk_size - 1) // chunk_size  # 올림

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, text_length)
                chunk_text = full_text_with_context[start_idx:end_idx]

                chunk_metadata = {
                    **base_metadata,  # 원본 S3 JSON 메타데이터 전체 포함
                    "chunk_id": f"{base_metadata['rec_idx']}_full_text_{i}",
                    "rec_idx": base_metadata["rec_idx"],
                    "company": base_metadata.get("company")
                    or base_metadata.get("company_name"),
                    "title": base_metadata.get("title")
                    or base_metadata.get("post_title"),
                    "url": base_metadata.get("url") or base_metadata.get("detail_url"),
                    "section_type": "full_text",
                    "section_length": len(chunk_text),
                    "tags": tags,
                    "is_fallback": True,
                    "chunk_index": i,
                    "total_chunks": num_chunks,
                    # deadline, start_date, crawling_time 명시적으로 보장
                    "deadline": base_metadata.get("deadline"),
                    "start_date": base_metadata.get("start_date"),
                    "crawling_time": base_metadata.get("crawling_time"),
                }

                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

            return chunks

    def _create_chunks_from_parsed(
        self,
        parsed_chunks: List[Dict[str, Any]],
        base_metadata: Dict[str, Any],
        raw_text: str = "",
    ) -> List[Chunk]:
        """
        JobPostParser 결과를 Chunk 객체로 변환

        Target 섹션이 없으면 fallback 청크 생성

        Args:
            parsed_chunks: JobPostParser.process_document() 결과
            base_metadata: 기본 메타데이터 (rec_idx, company, title, url)
            raw_text: 원본 텍스트 (fallback용)

        Returns:
            Chunk 객체 리스트
        """
        # 문서 레벨 태그 추출 (첫 번째 청크에서 가져옴)
        doc_tags = []
        if parsed_chunks:
            doc_tags = parsed_chunks[0].get("metadata", {}).get("tags", [])

        # 섹션별로 청크 생성
        chunks = []
        section_counters = defaultdict(int)

        for parsed_chunk in parsed_chunks:
            section_type = parsed_chunk.get("metadata", {}).get(
                "section_type", "unknown"
            )

            # 대상 섹션만 처리
            if section_type not in self.target_sections:
                continue

            text = parsed_chunk.get("text", "")
            if not text or len(text.strip()) < 10:
                continue

            # chunk_id 생성
            chunk_idx = section_counters[section_type]
            section_counters[section_type] += 1
            chunk_id = f"{base_metadata['rec_idx']}_{section_type}_{chunk_idx}"

            # 메타데이터 구성 (원본 S3 JSON 메타데이터 전체 포함)
            chunk_metadata = {
                **base_metadata,  # 원본 S3 JSON 메타데이터 전체 포함 (deadline, start_date, crawling_time 등)
                "chunk_id": chunk_id,
                "rec_idx": base_metadata["rec_idx"],  # 확실히 설정
                "company": base_metadata.get("company")
                or base_metadata.get("company_name"),
                "title": base_metadata.get("title") or base_metadata.get("post_title"),
                "url": base_metadata.get("url") or base_metadata.get("detail_url"),
                "section_type": section_type,
                "section_length": len(text),
                "tags": doc_tags,  # 문서 레벨 태그
                # deadline, start_date, crawling_time 명시적으로 보장
                "deadline": base_metadata.get("deadline"),
                "start_date": base_metadata.get("start_date"),
                "crawling_time": base_metadata.get("crawling_time"),
            }

            chunks.append(Chunk(text=text, metadata=chunk_metadata))

        # Target 섹션이 하나도 없으면 fallback 청크 생성
        if not chunks and raw_text:
            fallback_chunks = self._create_fallback_chunk(
                raw_text=raw_text, base_metadata=base_metadata, tags=doc_tags
            )
            chunks.extend(fallback_chunks)

            # 로깅
            if len(fallback_chunks) == 1 and fallback_chunks[0].metadata.get(
                "is_lightweight"
            ):
                print(f"   💡 경량 컨텍스트 생성: {base_metadata['rec_idx']}")
            else:
                print(
                    f"   ✂️  Fallback 청킹 ({len(fallback_chunks)}개): {base_metadata['rec_idx']}"
                )

        return chunks

    def get_section_chunks(self, section_type: str) -> List[Chunk]:
        """
        특정 섹션 타입의 청크만 필터링

        Args:
            section_type: "preferred", "qualifications", "job_duties"

        Returns:
            필터링된 청크 리스트
        """
        return [
            chunk
            for chunk in self.chunks
            if chunk.metadata.get("section_type") == section_type
        ]

    def save_chunks(
        self, output_dir: str = "structured_chunks", format: str = "json"
    ) -> Dict[str, str]:
        """
        청크 데이터 저장

        Args:
            output_dir: 저장 디렉토리
            format: "json" (CSV는 미지원)

        Returns:
            저장된 파일 경로 딕셔너리
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # 1. 전체 청크 저장
        all_chunks_file = output_path / f"all_chunks_{timestamp}.json"
        with open(all_chunks_file, "w", encoding="utf-8") as f:
            json.dump(
                [chunk.to_dict() for chunk in self.chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )
        saved_files["all_chunks"] = str(all_chunks_file)
        print(f"✅ 전체 청크 저장: {all_chunks_file}")

        # 2. 통계 저장
        stats = self.get_statistics()
        stats_file = output_path / f"statistics_{timestamp}.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        saved_files["statistics"] = str(stats_file)
        print(f"✅ 통계 저장: {stats_file}")

        # 3. 샘플 청크 저장 (확인용, 각 섹션별 3개씩)
        sample_chunks = []
        for section in self.target_sections:
            section_chunks = self.get_section_chunks(section)
            sample_chunks.extend([c.to_dict() for c in section_chunks[:3]])

        sample_file = output_path / f"sample_chunks_{timestamp}.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_chunks, f, ensure_ascii=False, indent=2)
        saved_files["sample"] = str(sample_file)
        print(f"✅ 샘플 저장: {sample_file}")

        return saved_files

    def get_statistics(self) -> Dict[str, Any]:
        """
        청크 통계 반환

        Returns:
            통계 정보 딕셔너리
        """
        if not self.chunks:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "section_distribution": {},
                "avg_chunk_length": 0,
                "avg_chunks_per_doc": 0,
                "total_unique_tags": 0,
                "target_sections": self.target_sections,
            }

        # 문서 수 (고유 rec_idx)
        unique_docs = set(chunk.metadata["rec_idx"] for chunk in self.chunks)

        # 섹션별 분포
        section_counts = Counter(
            chunk.metadata["section_type"] for chunk in self.chunks
        )

        # 평균 청크 길이
        avg_length = sum(len(chunk.text) for chunk in self.chunks) / len(self.chunks)

        # 문서당 평균 청크 수
        chunks_per_doc = defaultdict(int)
        for chunk in self.chunks:
            chunks_per_doc[chunk.metadata["rec_idx"]] += 1
        avg_chunks = sum(chunks_per_doc.values()) / len(chunks_per_doc)

        # 총 태그 수 (중복 제거)
        all_tags = set()
        for chunk in self.chunks:
            all_tags.update(chunk.metadata.get("tags", []))

        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(unique_docs),
            "section_distribution": dict(section_counts),
            "avg_chunk_length": round(avg_length, 2),
            "avg_chunks_per_doc": round(avg_chunks, 2),
            "total_unique_tags": len(all_tags),
            "target_sections": self.target_sections,
        }

    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()

        print(f"\n{'='*80}")
        print(f"📊 청크 통계")
        print(f"{'='*80}")
        print(f"총 청크 수: {stats['total_chunks']:,}개")
        print(f"총 문서 수: {stats['total_documents']:,}개")
        print(f"평균 청크 길이: {stats['avg_chunk_length']:.1f}자")
        print(f"문서당 평균 청크: {stats['avg_chunks_per_doc']:.2f}개")
        print(f"고유 태그 수: {stats['total_unique_tags']:,}개")
        print(f"\n섹션별 분포:")
        for section, count in stats["section_distribution"].items():
            pct = count / stats["total_chunks"] * 100
            print(f"  - {section}: {count:,}개 ({pct:.1f}%)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # 간단한 테스트
    loader = StructuredDocumentLoader()
    print("✅ StructuredDocumentLoader 모듈 로드 성공")
