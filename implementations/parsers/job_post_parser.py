"""
JobPostParser: 채용 공고 PDF 파싱 및 섹션 기반 청킹

3단계 파이프라인:
  1. 정의 (__init__)
  2. 파싱 (_parse_document, _group_by_sections)
  3. 후처리 (_sections_to_chunks)

주요 기능:
- 표준화된 섹션 감지 (주요업무, 자격요건 등)
- #태그 추출 및 metadata 저장
- 섹션 누락 시 컨텍스트 주입 (Fallback)
- rec_idx 기반 문서 구분
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ Warning: unstructured library not installed.")


class JobPostParser:
    """
    채용 공고 문서 파싱 및 RAG용 청크 생성

    Args:
        strategy: unstructured 파싱 전략 ("fast", "hi_res", "auto")
        include_context: 각 청크에 회사명/직무명 컨텍스트 주입 여부
        min_section_length: 최소 섹션 길이 (너무 짧은 섹션 제외)
    """

    def __init__(
        self,
        strategy: str = "fast",
        include_context: bool = True,
        min_section_length: int = 30,
    ):
        """1. 정의 단계: 파서 초기화 및 키워드 정의"""

        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "unstructured library is required. "
                "Install with: pip install 'unstructured[pdf]'"
            )

        self.strategy = strategy
        self.include_context = include_context
        self.min_section_length = min_section_length

        # 섹션 키워드 정의 (확장 버전 - 감지율 향상)
        self.section_keywords = {
            "company_intro": [
                "조직 소개",
                "팀소개",
                "회사 소개",
                "팀을 소개합니다",
                "회사소개",
                "조직소개",
                "Introduction",
                "ABOUT",
            ],
            "job_duties": [
                "담당업무",
                "주요업무",
                "이런 일을 하게",
                "함께 할 업무에요",
                "모집부문",
                "업무내용",
                "담당 업무",
                "주요 업무",
                "하실 일",
                "Responsibility",
                "responsibility",
                "responsibilities",
                "KEY PURPOSE OF ROLE",
            ],
            "qualifications": [
                "자격요건",
                "지원자격",
                "자격조건",
                "공통 자격요건",
                "이런 분을 찾고 있어요",
                "이런 분을 찾고",
                "필수 사항",
                "필수사항",
                "지원 자격",
                "자격 요건",
                "필수조건",
                "필수 조건",
                "자격",
                "requirement",
                "KEY RESPONSIBILITIES",
            ],
            "preferred": [
                "우대사항",
                "이런 분이면 더 좋아요",
                "이런 분이면",
                "우대 사항",
                "우대조건",
                "우대 조건",
                "우대 요건",
                "우대요건",
            ],
            "benefits": [
                "근무조건",
                "복리후생",
                "혜택 및 복지",
                "함께하면 받는 혜택 및 복지",
                "이런 혜택과 복지를",
                "근무 조건",
                "복리 후생",
                "혜택",
            ],
            "hiring_process": [
                "전형절차",
                "접수방법 및",
                "접수기간 및 방법",
                "제출서류",
                "이렇게 합류해요",
                "전형 절차",
                "제출 서류",
                "접수 방법",
            ],
            "notes": [
                "유의사항",
                "기타",
                "참고해 주세요",
                "채용서류 반환에 관한 고지",
                "유의 사항",
                "기타사항",
            ],
        }

        print(f"✅ JobPostParser 초기화:")
        print(f"   - 파싱 전략: {strategy}")
        print(f"   - 컨텍스트 주입: {include_context}")
        print(f"   - 최소 섹션 길이: {min_section_length}자")

    def process_document(
        self, doc_path: str, original_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        메인 오케스트레이터: 문서 파싱 → 섹션 그룹화 → 청크 생성

        Args:
            doc_path: 문서 파일 경로 (PDF, DOCX 등)
            original_metadata: 원본 메타데이터
                - rec_idx (필수): 문서 고유 식별자
                - company (필수): 회사명
                - title (필수): 직무명

        Returns:
            청크 리스트 (각 청크는 text와 metadata 포함)
        """
        # 필수 필드 검증 (rec_idx 또는 rec_id 허용, 하위 호환성)
        rec_idx = original_metadata.get("rec_idx") or original_metadata.get("rec_id")
        if not rec_idx:
            raise ValueError("original_metadata must contain 'rec_idx' or 'rec_id'")
        
        required_fields = ["company", "title"]
        for field in required_fields:
            if field not in original_metadata:
                raise ValueError(f"original_metadata must contain '{field}'")
        
        # rec_idx가 없으면 rec_id를 rec_idx로 설정 (하위 호환성)
        if "rec_idx" not in original_metadata and "rec_id" in original_metadata:
            original_metadata["rec_idx"] = original_metadata["rec_id"]

        print(f"\n{'='*70}")
        print(f"📄 문서 처리: {original_metadata.get('rec_idx')}")
        print(f"   회사: {original_metadata.get('company')}")
        print(f"   직무: {original_metadata.get('title')}")
        print(f"{'='*70}")

        # Step 1: 문서 파싱
        elements = self._parse_document(doc_path)
        print(f"✅ {len(elements)}개 elements 추출")

        # Step 2: 섹션 그룹화 및 태그 감지
        sections, tags = self._group_by_sections(elements)
        print(f"✅ {len(sections)}개 섹션 감지")
        print(f"✅ {len(tags)}개 태그 추출: {tags[:5]}...")  # 처음 5개만

        # Step 3: 메타데이터 업데이트
        doc_metadata = original_metadata.copy()
        doc_metadata["tags"] = tags

        # Step 4: 청크 생성 (후처리)
        chunks = self._sections_to_chunks(sections, doc_metadata)
        print(f"✅ {len(chunks)}개 청크 생성")

        return chunks

    def _parse_document(self, doc_path: str) -> List:
        """
        2. 파싱 단계: unstructured로 문서 로드

        Args:
            doc_path: 문서 파일 경로

        Returns:
            elements 리스트
        """
        print(f"📥 문서 로드 중... (strategy: {self.strategy})")

        # PDF 파일인지 확인
        file_path = Path(doc_path)

        if file_path.suffix.lower() == ".pdf":
            elements = partition_pdf(
                filename=doc_path,
                strategy=self.strategy,
                languages=["kor", "eng"],
                infer_table_structure=False,
                extract_images_in_pdf=False,
            )
        else:
            # 기타 파일 형식 (DOCX, TXT 등)
            elements = partition(filename=doc_path, languages=["kor", "eng"])

        return elements

    def _normalize_text(self, text: str) -> str:
        """
        2. 파싱 단계 (보조): 텍스트 정규화

        줄 시작 부분의 특수문자 및 공백 제거:
        - 제거 대상: ■, ◆, ●, ▶, ◾, ▪, *, [], (), 공백 등

        Args:
            text: 원본 텍스트

        Returns:
            정규화된 텍스트
        """
        # 줄 시작 부분 특수문자 제거 (대괄호, 소괄호 추가)
        normalized = re.sub(r"^[\[\]()■◆●▶◾▪*\s]+", "", text)

        # 앞뒤 공백 및 강조 문자 제거 (느낌표, 대괄호 추가)
        normalized = normalized.strip().strip(" *-_[]()!?")

        return normalized

    def _detect_section(self, elem) -> Optional[str]:
        """
        2. 파싱 단계 (핵심): 섹션 감지

        elem.text를 정규화한 후, section_keywords와 매칭하여
        표준 섹션 타입을 반환합니다.

        Args:
            elem: unstructured Element 객체

        Returns:
            섹션 타입 (예: "job_duties") 또는 None
        """
        if not elem.text:
            return None

        normalized_text = self._normalize_text(elem.text)

        # section_keywords를 순회하며 startswith 비교
        for section_type, keywords in self.section_keywords.items():
            for keyword in keywords:
                if normalized_text.startswith(keyword):
                    return section_type

        return None

    def _group_by_sections(self, elements: List) -> Tuple[Dict[str, List], List[str]]:
        """
        2. 파싱 단계 (그룹화): 섹션별 그룹화 및 태그 감지

        각 element를 순회하며:
        1. #태그 감지 → detected_tags에 추가
        2. 섹션 헤더 감지 → current_section 업데이트
        3. 현재 섹션에 element 귀속 (Fallback 1)

        Args:
            elements: unstructured elements 리스트

        Returns:
            (sections, tags) 튜플
            - sections: {section_type: [elem1, elem2, ...]}
            - tags: ["#태그1", "#태그2", ...]
        """
        sections = {}
        detected_tags = []
        current_section = "header"  # 기본 섹션

        for elem in elements:
            if not elem.text or not elem.text.strip():
                continue

            text = elem.text.strip()

            # ========================================
            # [태그 감지 로직]
            # ========================================
            if text.startswith("#"):
                # 해당 라인의 모든 태그 추출
                tags_in_line = re.findall(r"#\S+", text)
                detected_tags.extend(tags_in_line)
                continue  # 태그는 섹션에 포함하지 않음

            # ========================================
            # [섹션 감지 로직]
            # ========================================
            detected_section = self._detect_section(elem)

            if detected_section:
                current_section = detected_section
                print(f"   🔖 섹션 감지: {current_section} - '{elem.text[:40]}...'")

            # ========================================
            # [섹션 귀속 로직 / Fallback 1]
            # ========================================
            # 키워드 미감지 시 current_section이 바뀌지 않으므로
            # 자동으로 header 또는 직전 섹션에 귀속됨
            if current_section not in sections:
                sections[current_section] = []

            sections[current_section].append(elem)

        # 중복 태그 제거
        unique_tags = list(set(detected_tags))

        return sections, unique_tags

    def _sections_to_chunks(
        self, sections: Dict[str, List], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        3. 후처리 단계: 섹션을 청크로 변환

        각 섹션에 대해:
        1. 컨텍스트 주입 (Fallback 2)
        2. Elements를 텍스트로 결합
        3. 청크 딕셔너리 생성 (metadata 확장)

        Args:
            sections: 섹션별 elements
            metadata: 문서 메타데이터 (rec_idx, company, title, tags 포함)

        Returns:
            청크 리스트
        """
        chunks = []

        # ========================================
        # [Fallback 2: 컨텍스트 주입]
        # ========================================
        company = metadata.get("company", "")
        title = metadata.get("title", "")

        context = ""
        if self.include_context and (company or title):
            context = f"[회사: {company}] [직무: {title}]\n\n"

        # 섹션별 청크 생성
        for section_type, elements in sections.items():
            # Elements를 텍스트로 결합
            section_text = "\n".join(
                [elem.text for elem in elements if elem.text and elem.text.strip()]
            )

            # 최소 길이 체크
            if len(section_text.strip()) < self.min_section_length:
                print(
                    f"   ⚠️ 섹션 '{section_type}' 너무 짧음 ({len(section_text)}자), 스킵"
                )
                continue

            # 컨텍스트 주입
            final_text = context + section_text if context else section_text

            # ========================================
            # [청크 딕셔너리 생성]
            # ========================================
            chunk = {
                "text": final_text,
                "metadata": {
                    **metadata,  # rec_idx, company, title, tags 포함
                    "section_type": section_type,
                    "section_length": len(section_text),
                    "chunk_method": "structured_parsing",
                    "has_context": self.include_context,
                },
            }

            chunks.append(chunk)
            print(f"   ✅ Chunk 생성: {section_type} ({len(section_text)}자)")

        return chunks


# ========================================
# 디버깅 및 테스트 유틸리티
# ========================================


def preview_document_structure(doc_path: str, strategy: str = "fast"):
    """
    문서 구조 미리보기 (디버깅용)

    Usage:
        preview_document_structure("/tmp/sample.pdf")
    """
    from unstructured.partition.pdf import partition_pdf

    print(f"\n{'='*70}")
    print(f"📄 문서 구조 분석: {doc_path}")
    print(f"{'='*70}")

    elements = partition_pdf(
        filename=doc_path, strategy=strategy, languages=["kor", "eng"]
    )

    print(f"\n총 {len(elements)}개 elements\n")

    for i, elem in enumerate(elements[:20], 1):  # 처음 20개만
        print(f"[{i:2d}] {elem.category:15s} | {elem.text[:60]}")

    if len(elements) > 20:
        print(f"... (나머지 {len(elements) - 20}개 생략)")


def analyze_section_distribution(chunks: List[Dict[str, Any]]):
    """
    청크의 섹션 분포 분석 (디버깅용)

    Usage:
        chunks = parser.process_document(...)
        analyze_section_distribution(chunks)
    """
    from collections import Counter

    print(f"\n{'='*70}")
    print("📊 섹션 분포 분석")
    print(f"{'='*70}")

    section_types = [chunk["metadata"]["section_type"] for chunk in chunks]
    distribution = Counter(section_types)

    print(f"\n총 {len(chunks)}개 청크\n")

    for section_type, count in distribution.most_common():
        percentage = (count / len(chunks)) * 100
        print(f"  {section_type:20s}: {count:3d}개 ({percentage:5.1f}%)")

    # 태그 통계
    all_tags = []
    for chunk in chunks:
        all_tags.extend(chunk["metadata"].get("tags", []))

    if all_tags:
        unique_tags = set(all_tags)
        print(f"\n태그 통계:")
        print(f"  - 총 태그 수: {len(all_tags)}개")
        print(f"  - 고유 태그 수: {len(unique_tags)}개")
        print(f"  - 상위 10개: {list(unique_tags)[:10]}")
