"""
Career-HY RAG 실험 파이프라인 메인 로직
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional

from .config import ExperimentConfig
from .interfaces.evaluator import QueryResult
from utils.factory import ComponentFactory
from utils.data_loader import S3DataLoader
from utils.embedding_cache import embedding_cache
from utils.document_cache import document_cache
from utils.sampler import (
    StratifiedSampler,
    generate_reproducible_seed,
    analyze_sample_distribution,
)
from implementations.evaluators import RetrieverEvaluator
from implementations.loaders import StructuredDocumentLoader, Chunk

# 임베딩 생성 전에 섹션별 분포 확인
from collections import Counter

from implementations.evaluators.evaluators_back.langsmith_evaluator import (
    CareerHYLangSmithEvaluator,
)


class ExperimentPipeline:
    """RAG 실험을 실행하는 메인 파이프라인"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}

        # 실험 ID 및 출력 디렉토리 설정
        self.experiment_id = config.get_experiment_id()
        self.output_dir = config.get_output_path()

        print(f"실험 시작: {config.experiment_name}")
        print(f"실험 ID: {self.experiment_id}")
        print(f"출력 디렉토리: {self.output_dir}")

    async def run(self) -> Dict[str, Any]:
        """
        전체 실험 파이프라인 실행

        Returns:
            실험 결과 딕셔너리
        """
        start_time = time.time()

        try:
            # 1. 컴포넌트 초기화
            print("\n=== 1. 컴포넌트 초기화 ===")
            components = self._initialize_components()

            # 2. 데이터 로드
            print("\n=== 2. 데이터 로드 ===")
            # S3의 모든 데이터 사용
            documents = self._load_documents()

            # 3. 문서 처리 및 임베딩
            print("\n=== 3. 문서 처리 및 임베딩 ===")
            processed_docs, embeddings = self._process_documents(documents, components)

            # 4. 검색 시스템 구축
            print("\n=== 4. 검색 시스템 구축 ===")
            self._build_retrieval_system(processed_docs, embeddings, components)

            eval_mode = self.config.evaluation.mode
            print(f"🔍 평가 모드: {eval_mode}")

            # 평가 없이 인덱스 구축까지만 수행하는 모드
            if eval_mode == "build_only":
                print("\n=== 5. 평가 스킵 (build_only 모드) ===")
                build_only_results = {
                    "experiment_info": {
                        "name": self.config.experiment_name,
                        "description": self.config.description,
                        "experiment_id": self.experiment_id,
                        "timestamp": datetime.now().isoformat(),
                        "duration_seconds": time.time() - start_time,
                        "evaluation_type": "build_only",
                    },
                    "config": {
                        "embedder": asdict(self.config.embedder),
                        "chunker": asdict(self.config.chunker),
                        "retriever": asdict(self.config.retriever),
                    },
                    "document_count": components["retriever"].get_document_count(),
                }
                print(
                    f"\n📦 build_only 완료 - Chroma/검색 시스템에 저장된 청크 수: {build_only_results['document_count']}"
                )
                return build_only_results

            # 5. Ground Truth 쿼리 로드
            print("\n=== 5. Ground Truth 쿼리 로드 ===")
            test_queries = self._load_test_queries()

            # 6. 평가 실행 (모드에 따라 분기)
            if eval_mode == "retrieval_only":
                # 검색 성능 평가만 수행
                print("\n=== 6. 검색 성능 평가 (Retrieval Only) ===")
                eval_results = await self._run_retrieval_only_evaluation(
                    test_queries, components, start_time
                )

                # 7. 결과 저장
                print("\n=== 7. 결과 저장 ===")
                results = self._save_retrieval_results(eval_results, start_time)
            else:
                # 기존 이중 평가 시스템
                print("\n=== 6. 이중 평가 시스템 (Dual Evaluation) ===")
                dual_results = await self._run_dual_evaluation(test_queries, components)

                # 7. 결과 저장
                print("\n=== 7. 결과 저장 ===")
                results = await self._save_dual_results(
                    dual_results, components, start_time
                )

            print(f"\n실험 완료! 총 소요시간: {time.time() - start_time:.2f}초")
            return results

        except Exception as e:
            print(f"\n실험 실행 중 오류 발생: {e}")
            raise

    def _initialize_components(self) -> Dict[str, Any]:
        """설정에 따라 컴포넌트들 초기화"""
        components = {}

        # 임베딩 모델 초기화
        print(
            f"임베딩 모델 초기화: {self.config.embedder.type} - {self.config.embedder.model_name}"
        )
        components["embedder"] = ComponentFactory.create_embedder(self.config.embedder)

        # 청킹 전략 초기화
        print(f"청킹 전략 초기화: {self.config.chunker.type}")
        components["chunker"] = ComponentFactory.create_chunker(self.config.chunker)

        # 검색 시스템 초기화
        print(f"검색 시스템 초기화: {self.config.retriever.type}")
        components["retriever"] = ComponentFactory.create_retriever(
            self.config.retriever
        )

        # LLM 모델 초기화 (필요한 경우)
        if hasattr(self.config, "llm") and self.config.llm:
            print(
                f"LLM 모델 초기화: {self.config.llm.type} - {self.config.llm.model_name}"
            )
            components["llm"] = ComponentFactory.create_llm(self.config.llm)

        # 응답 생성기 초기화 (선택적)
        if (
            hasattr(self.config, "response_generator")
            and self.config.response_generator
        ):
            print(f"응답 생성기 초기화: {self.config.response_generator.type}")
            components["response_generator"] = (
                ComponentFactory.create_response_generator(
                    self.config.response_generator
                )
            )

        # 평가기 초기화 (dual evaluation 모드용)
        from implementations.evaluators import RetrieverEvaluator

        if hasattr(self.config, "evaluation") and self.config.evaluation:
            components["evaluator"] = RetrieverEvaluator()
            print("✅ 평가기 초기화 완료: RetrieverEvaluator")

        return components

    def _load_documents(self) -> List[Dict[str, Any]]:
        """S3에서 모든 문서 데이터 로드 (캐싱 지원)"""

        # 캐시 키 생성 (data_version 포함)
        cache_key = document_cache.generate_cache_key(
            self.config.data.s3_bucket,
            self.config.data.pdf_prefix,
            self.config.data.json_prefix,
            self.config.data.data_version,
        )

        # 캐시 확인
        if document_cache.exists(cache_key):
            print(f"✅ 기존 문서 캐시 사용: {cache_key}")
            documents = document_cache.load(cache_key)
            return documents

        print(f"🔄 S3에서 문서 로드 중: {cache_key}")

        # S3에서 로드
        data_loader = S3DataLoader(bucket_name=self.config.data.s3_bucket)
        documents = data_loader.load_documents(
            pdf_prefix=self.config.data.pdf_prefix,
            json_prefix=self.config.data.json_prefix,
        )

        print(f"로드된 문서 수: {len(documents)}")

        # 캐시에 저장
        s3_config = {
            "s3_bucket": self.config.data.s3_bucket,
            "pdf_prefix": self.config.data.pdf_prefix,
            "json_prefix": self.config.data.json_prefix,
            "data_version": self.config.data.data_version,
        }

        try:
            document_cache.save(cache_key, documents, s3_config)
        except Exception as e:
            print(f"⚠️  문서 캐시 저장 실패 (실험은 계속 진행): {e}")

        return documents

    def _process_documents(
        self, documents: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> tuple:
        """문서 청킹 및 임베딩 처리 (캐싱 지원)"""

        # StructuredDocumentLoader 기반 텍스트 파서 사용 여부에 따라 분기
        if getattr(self.config.data, "use_structured_loader", False):
            return self._process_documents_structured(documents, components)

        chunker = components["chunker"]
        embedder = components["embedder"]

        # 캐시 키 생성
        cache_key = embedding_cache.generate_cache_key(
            self.config.embedder, self.config.chunker
        )

        # 캐시 확인
        if embedding_cache.exists(cache_key):
            print(f"✅ 기존 임베딩 캐시 사용: {cache_key}")
            cached_documents, cached_embeddings = embedding_cache.load(cache_key)
            return cached_documents, cached_embeddings

        print(f"🔄 새로운 임베딩 생성: {cache_key}")

        all_chunks = []
        all_texts = []

        print("문서 청킹 중...")
        for i, doc in enumerate(documents):
            # 청킹 수행
            chunks = chunker.chunk(doc["text"], doc["metadata"])
            all_chunks.extend(chunks)

            # 임베딩용 텍스트 추출
            for chunk in chunks:
                all_texts.append(chunk["text"])

            if (i + 1) % 50 == 0:
                print(f"청킹 완료: {i + 1}/{len(documents)} 문서")

        print(f"총 청크 수: {len(all_chunks)}")

        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = embedder.embed(all_texts)

        print(f"임베딩 완료: {len(embeddings)}개 벡터")

        # 캐시에 저장
        additional_info = {
            "original_document_count": len(documents),
            "embedder_config": self.config.embedder.__dict__,
            "chunker_config": self.config.chunker.__dict__,
        }

        try:
            embedding_cache.save(cache_key, all_chunks, embeddings, additional_info)
        except Exception as e:
            print(f"⚠️  캐시 저장 실패 (실험은 계속 진행): {e}")

        return all_chunks, embeddings

    def _process_documents_structured(
        self, documents: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> tuple:
        """
        StructuredDocumentLoader + 텍스트 기반 JobPostParser를 사용한
        섹션별 청킹 및 임베딩 처리 (캐싱 지원)
        """
        embedder = components["embedder"]
        data_cfg = self.config.data

        # 🔑 임베딩 캐시 키 생성 (임베더+청커 설정 + structured + data_version)
        base_cache_key = embedding_cache.generate_cache_key(
            self.config.embedder, self.config.chunker
        )
        cache_key = f"{base_cache_key}_structured_{data_cfg.data_version}"

        # 캐시 확인
        if embedding_cache.exists(cache_key):
            print(f"✅ 기존 structured 임베딩 캐시 사용: {cache_key}")
            cached_documents, cached_embeddings = embedding_cache.load(cache_key)
            return cached_documents, cached_embeddings

        # StructuredDocumentLoader 초기화
        loader = StructuredDocumentLoader(
            strategy=getattr(data_cfg, "structured_parser_strategy", "fast"),
            target_sections=(
                data_cfg.structured_target_sections
                if getattr(data_cfg, "structured_target_sections", None)
                else None
            ),
            include_context=True,
        )

        print("문서 구조화 청킹 중 (StructuredDocumentLoader + TextJobPostParser)...")
        chunks: List[Chunk] = loader.load_from_documents(documents)

        # Text 파서 결과 통계 출력 및 중간 결과 저장
        loader.print_statistics()
        try:
            saved_files = loader.save_chunks(output_dir="structured_chunks")
            print(f"📁 Text 파서 중간 결과 저장 완료: {saved_files}")
        except Exception as e:
            print(f"⚠️ Text 파서 중간 결과 저장 실패 (실험은 계속 진행): {e}")

        # Recursive chunking 추가 적용 (chunker가 recursive 타입이고 설정된 경우)
        chunker = components.get("chunker")
        apply_recursive_chunking = (
            chunker is not None
            and hasattr(chunker, "chunk")
            and self.config.chunker.type == "recursive"
            and self.config.chunker.chunk_size is not None
        )

        if apply_recursive_chunking:
            print(f"\n🔄 Recursive Chunking 추가 적용 중...")
            print(f"   - Chunk size: {self.config.chunker.chunk_size}")
            print(f"   - Chunk overlap: {self.config.chunker.chunk_overlap}")

            final_chunks = []
            original_count = len(chunks)
            section_stats = {}  # 섹션별 통계

            for chunk in chunks:
                section_type = chunk.metadata.get("section_type", "unknown")

                # 각 섹션 청크에 대해 recursive chunking 적용
                sub_chunks = chunker.chunk(chunk.text, chunk.metadata)

                # 섹션별 통계 수집
                if section_type not in section_stats:
                    section_stats[section_type] = {
                        "original_chunks": 0,
                        "final_chunks": 0,
                        "split_count": 0,  # 나뉜 청크 수
                    }
                section_stats[section_type]["original_chunks"] += 1
                section_stats[section_type]["final_chunks"] += len(sub_chunks)
                if len(sub_chunks) > 1:
                    section_stats[section_type]["split_count"] += 1

                # 원본 메타데이터 유지하면서 chunk_index 추가
                for i, sub_chunk in enumerate(sub_chunks):
                    # 원본 섹션 정보 유지
                    final_metadata = {
                        **chunk.metadata,
                        **sub_chunk["metadata"],
                        "original_chunk_id": chunk.metadata.get("chunk_id"),
                        "recursive_chunk_index": i,
                        "recursive_chunk_count": len(sub_chunks),
                    }
                    final_chunks.append(
                        {"text": sub_chunk["text"], "metadata": final_metadata}
                    )

            print(
                f"   ✅ Recursive chunking 완료: {original_count}개 → {len(final_chunks)}개 청크"
            )

            # 섹션별 통계 출력
            print(f"\n   📊 섹션별 Recursive Chunking 통계:")
            for section_type, stats in sorted(section_stats.items()):
                avg_final = (
                    stats["final_chunks"] / stats["original_chunks"]
                    if stats["original_chunks"] > 0
                    else 0
                )
                split_pct = (
                    (stats["split_count"] / stats["original_chunks"] * 100)
                    if stats["original_chunks"] > 0
                    else 0
                )
                print(f"      - {section_type}:")
                print(
                    f"        • 원본: {stats['original_chunks']}개 → 최종: {stats['final_chunks']}개 (평균 {avg_final:.2f}배)"
                )
                print(
                    f"        • 분할된 청크: {stats['split_count']}개 ({split_pct:.1f}%)"
                )

            structured_docs = final_chunks
        else:
            # Recursive chunking 미적용: 원본 청크 그대로 사용
            structured_docs: List[Dict[str, Any]] = [
                {"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks
            ]

        print(f"총 최종 청크 수: {len(structured_docs)}")

        # 임베딩 생성
        texts = [doc["text"] for doc in structured_docs]
        print("임베딩 생성 중 (structured)...")
        embeddings = embedder.embed(texts)

        print(f"임베딩 완료: {len(embeddings)}개 벡터")

        # 캐시에 저장 (각 문서 ID(rec_idx)도 함께 기록)
        doc_ids = [
            doc.get("metadata", {}).get("rec_idx")
            for doc in structured_docs
            if doc.get("metadata", {}).get("rec_idx") is not None
        ]
        additional_info = {
            "original_document_count": len(documents),
            "structured": True,
            "target_sections": getattr(data_cfg, "structured_target_sections", None),
            "embedder_config": self.config.embedder.__dict__,
            "data_version": data_cfg.data_version,
            "document_ids": doc_ids,
        }

        try:
            embedding_cache.save(
                cache_key, structured_docs, embeddings, additional_info
            )
        except Exception as e:
            print(f"⚠️  structured 임베딩 캐시 저장 실패 (실험은 계속 진행): {e}")

        return structured_docs, embeddings

    def _build_retrieval_system(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        components: Dict[str, Any],
    ) -> None:
        """검색 시스템에 문서와 임베딩 추가"""
        retriever = components["retriever"]

        print("검색 시스템에 문서 추가 중...")

        # 기존 컬렉션 초기화 (실험용)
        if hasattr(retriever, "clear_collection"):
            retriever.clear_collection()

        # 문서와 임베딩을 검색 시스템에 추가
        retriever.add_documents(documents, embeddings)

        # FAISS인 경우 인덱스 저장 (ChromaDB는 자동 저장)
        if hasattr(retriever, "save_index"):
            retriever.save_index()
            print("💾 FAISS 인덱스 저장 완료")

        doc_count = retriever.get_document_count()
        print(f"검색 시스템 구축 완료: {doc_count}개 문서")

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Ground Truth 테스트 쿼리 로드 (기존 방식 + evaluation_queries.jsonl 지원)"""
        test_queries_path = self.config.data.test_queries_path

        if not Path(test_queries_path).exists():
            print(f"⚠️  테스트 쿼리 파일이 없습니다: {test_queries_path}")
            print("샘플 테스트 쿼리를 생성합니다...")
            return self._create_sample_queries()

        # 기존 방식: 직접 JSONL 파일 읽기
        queries = []
        try:
            with open(test_queries_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    try:
                        query_data = json.loads(line.strip())

                        # query_text에서 질문과 프로필 분리
                        if "query_text" in query_data:
                            parsed = self._parse_query_text(query_data["query_text"])
                            query_data["query"] = parsed["query"]
                            query_data["user_profile"] = parsed["user_profile"]
                            # query_text는 그대로 유지 (하위 호환성)

                            # 첫 번째 쿼리만 디버깅 정보 출력
                            if line_num == 1:
                                print(f"🔍 첫 번째 쿼리 파싱 결과:")
                                print(f"   - 질문: {parsed['query'][:100]}...")
                                print(
                                    f"   - 프로필 키: {list(parsed['user_profile'].keys())}"
                                )
                                if "catalogs" in parsed["user_profile"]:
                                    print(
                                        f"   - 수강 이력: {len(parsed['user_profile']['catalogs'])}개 강의"
                                    )

                        queries.append(query_data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON 파싱 오류 (Line {line_num}): {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️  쿼리 파싱 오류 (Line {line_num}): {e}")
                        continue

            # 데이터 형식 확인
            if queries and "ground_truth" in queries[0]:
                print(f"📊 새로운 평가 데이터 형식 로드: {len(queries)}개 쿼리")
                print(
                    f"   - GT 포함 (평균 {sum(len(q.get('ground_truth', [])) for q in queries) / len(queries):.1f}개/쿼리)"
                )
                if queries[0].get("user_profile"):
                    print(f"   - 사용자 프로필 자동 파싱 완료")
            else:
                print(f"📊 기존 테스트 쿼리 형식 로드: {len(queries)}개 쿼리")

            return queries

        except Exception as e:
            print(f"❌ 쿼리 로드 실패: {e}")
            return []

    def _parse_query_text(self, query_text: str) -> Dict[str, Any]:
        """
        query_text에서 질문과 사용자 프로필을 분리

        형식:
        질문: ...
        전공: ...
        관심 직무: ...
        자격증: ...
        동아리/대외활동: ...
        수강 이력:
        ...
        """
        result = {"query": "", "user_profile": {}}

        if not query_text:
            return result

        lines = query_text.split("\n")
        current_section = None
        query_parts = []
        profile_parts = {}
        course_history_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 섹션 헤더 확인
            if line.startswith("질문:"):
                current_section = "query"
                query_parts.append(line.replace("질문:", "").strip())
            elif line.startswith("전공:"):
                current_section = "profile"
                profile_parts["major"] = line.replace("전공:", "").strip()
            elif line.startswith("관심 직무:"):
                current_section = "profile"
                interest_job = line.replace("관심 직무:", "").strip()
                # 쉼표로 구분된 경우 리스트로 변환
                if "," in interest_job:
                    profile_parts["interest_job"] = [
                        j.strip() for j in interest_job.split(",")
                    ]
                else:
                    profile_parts["interest_job"] = interest_job
            elif line.startswith("자격증:"):
                current_section = "profile"
                certification = line.replace("자격증:", "").strip()
                # 쉼표로 구분된 경우 리스트로 변환
                if "," in certification:
                    profile_parts["certification"] = [
                        c.strip() for c in certification.split(",")
                    ]
                else:
                    profile_parts["certification"] = certification
            elif line.startswith("동아리/대외활동:"):
                current_section = "profile"
                activities = line.replace("동아리/대외활동:", "").strip()
                # 쉼표로 구분된 경우 리스트로 변환
                if "," in activities:
                    profile_parts["club_activities"] = [
                        a.strip() for a in activities.split(",")
                    ]
                else:
                    profile_parts["club_activities"] = activities
            elif line.startswith("수강 이력:"):
                current_section = "course_history"
                course_history_lines = []
            else:
                # 현재 섹션에 따라 내용 추가
                if current_section == "query":
                    query_parts.append(line)
                elif current_section == "course_history":
                    course_history_lines.append(line)

        # 질문 조합
        result["query"] = " ".join(query_parts).strip()

        # 수강 이력 파싱
        if course_history_lines:
            catalogs = []
            current_course = {}

            for line in course_history_lines:
                if line.startswith("강의명:"):
                    if current_course:
                        catalogs.append(current_course)
                    current_course = {
                        "course_name": line.replace("강의명:", "").strip()
                    }
                elif line.startswith("핵심 역량:"):
                    if current_course:
                        current_course["core_competency"] = line.replace(
                            "핵심 역량:", ""
                        ).strip()
                elif line.startswith("강의 개요:"):
                    if current_course:
                        current_course["course_overview"] = line.replace(
                            "강의 개요:", ""
                        ).strip()
                elif line.startswith("학습 목표:"):
                    if current_course:
                        current_course["learning_objectives"] = line.replace(
                            "학습 목표:", ""
                        ).strip()
                elif current_course and line:
                    # 이전 필드에 내용 추가
                    if "course_overview" in current_course and not current_course.get(
                        "learning_objectives"
                    ):
                        current_course["course_overview"] += " " + line
                    elif "learning_objectives" in current_course:
                        current_course["learning_objectives"] += " " + line

            if current_course:
                catalogs.append(current_course)

            if catalogs:
                profile_parts["catalogs"] = catalogs

        result["user_profile"] = profile_parts

        return result

    def _create_sample_queries(self) -> List[Dict[str, Any]]:
        """샘플 테스트 쿼리 생성 (Ground Truth가 없을 때)"""
        sample_queries = [
            {
                "query": "컴퓨터공학 전공 신입 개발자 채용공고",
                "ground_truth_docs": [],  # 실제로는 관련 문서 ID들이 들어가야 함
                "user_profile": {
                    "major": "컴퓨터공학과",
                    "interest_job": ["개발자", "프로그래머"],
                },
            },
            {
                "query": "데이터 사이언스 관련 직무",
                "ground_truth_docs": [],
                "user_profile": {
                    "major": "데이터사이언스학과",
                    "interest_job": ["데이터 분석가", "데이터 사이언티스트"],
                },
            },
        ]

        print(f"샘플 쿼리 생성: {len(sample_queries)}개")
        return sample_queries

    def _print_retrieval_evaluation_statistics(
        self, test_queries: List[Dict[str, Any]]
    ) -> None:
        """Retrieval 평가 전 통계 출력"""
        from collections import Counter

        print("\n" + "=" * 80)
        print("📊 Retrieval 평가 전 통계")
        print("=" * 80)

        # 총 쿼리 개수
        total_queries = len(test_queries)
        print(f"\n📝 총 쿼리 개수: {total_queries}개")

        # 정답 분포 분석
        gt_counts = []
        valid_queries = 0

        for query_data in test_queries:
            # 필드명 호환성
            ground_truth = query_data.get("ground_truth_docs") or query_data.get(
                "ground_truth", []
            )

            if ground_truth:
                gt_count = len(ground_truth)
                gt_counts.append(gt_count)
                valid_queries += 1

        if gt_counts:
            print(f"\n✅ 유효한 쿼리 개수: {valid_queries}개 (정답이 있는 쿼리)")
            print(f"⚠️  정답 없는 쿼리: {total_queries - valid_queries}개")

            # 정답 개수 통계
            print(f"\n📊 정답 개수 통계:")
            print(f"   - 평균: {sum(gt_counts) / len(gt_counts):.2f}개")
            print(f"   - 중앙값: {sorted(gt_counts)[len(gt_counts) // 2]}개")
            print(f"   - 최소: {min(gt_counts)}개")
            print(f"   - 최대: {max(gt_counts)}개")

            # 정답 개수 분포
            gt_distribution = Counter(gt_counts)
            print(f"\n📈 정답 개수 분포:")
            sorted_dist = sorted(gt_distribution.items(), key=lambda x: x[0])
            for count, freq in sorted_dist[:20]:  # 상위 20개만 표시
                percentage = (freq / len(gt_counts)) * 100
                print(f"   - {count}개 정답: {freq}개 쿼리 ({percentage:.1f}%)")

            # Retrieval 설정 정보
            print(f"\n🔍 Retrieval 설정:")
            print(
                f"   - top_k: {self.config.retriever.top_k}개 (동적 top_k: max(기존 top_k, 정답개수), 최대 60)"
            )
            # 평가 지표 출력 (사용자 요청 순서)
            if (
                hasattr(self.config, "evaluation")
                and self.config.evaluation
                and self.config.evaluation.metrics
            ):
                metrics_list = self.config.evaluation.metrics
                # 사용자 요청 순서대로 정렬
                metric_order = [
                    "ndcg@10",
                    "mrr@10",
                    "precision@3",
                    "precision@5",
                    "precision@10",
                    "precision@20",
                    "hit@10_count",
                    "r_recall",
                ]
                ordered_metrics = [m for m in metric_order if m in metrics_list]
                other_metrics = [m for m in metrics_list if m not in metric_order]
                all_metrics = ordered_metrics + other_metrics
                print(f"   - 평가 지표: {', '.join(all_metrics)}")
            else:
                print(f"   - 평가 지표: (설정되지 않음)")
        else:
            print(f"\n⚠️  정답이 있는 쿼리가 없습니다!")

        print("=" * 80 + "\n")

    def _print_single_query_details(self, query_result, evaluator) -> None:
        """단일 쿼리에 대한 상세 결과 출력 (디버깅용)"""
        print("\n" + "=" * 80)
        print("🔍 단일 쿼리 상세 결과 (첫 번째 쿼리)")
        print("=" * 80)

        # 쿼리 정보
        query_text = query_result.query
        print(f"\n📝 쿼리:")
        print(
            f"   {query_text[:200]}..." if len(query_text) > 200 else f"   {query_text}"
        )

        # Ground Truth 문서들
        gt_docs = query_result.ground_truth_docs
        print(f"\n✅ Ground Truth 문서 ({len(gt_docs)}개):")
        gt_rec_idxs = set()
        for idx, gt_doc in enumerate(gt_docs, 1):
            if isinstance(gt_doc, dict):
                rec_idx = gt_doc.get("rec_idx") or gt_doc.get("rec_id", "unknown")
                title = (
                    gt_doc.get("title")
                    or gt_doc.get("job_title")
                    or gt_doc.get("post_title", "제목 없음")
                )
                url = gt_doc.get("url") or gt_doc.get("detail_url", "")
                company = gt_doc.get("company") or gt_doc.get("company_name", "")
                deadline = gt_doc.get("deadline", "")
                start_date = gt_doc.get("start_date", "")
                crawling_time = gt_doc.get("crawling_time", "")
            else:
                rec_idx = str(gt_doc)
                title = "제목 없음"
                url = ""
                company = ""
                deadline = ""
                start_date = ""
                crawling_time = ""

            gt_rec_idxs.add(str(rec_idx))
            print(f"   {idx}. rec_idx: {rec_idx}")
            print(f"      - 제목: {title}")
            print(f"      - 회사: {company}")
            print(f"      - URL: {url}")
            if deadline:
                print(f"      - 마감일: {deadline}")
            if start_date:
                print(f"      - 시작일: {start_date}")
            if crawling_time:
                print(f"      - 크롤링시간: {crawling_time}")

        # 검색된 문서들
        retrieved_docs = query_result.retrieved_docs
        print(f"\n🔍 검색된 문서 (상위 {len(retrieved_docs)}개):")

        # 평가 지표 계산 (이 쿼리만)
        retrieved_rec_idxs = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                rec_idx = metadata.get("rec_idx") or metadata.get("rec_id", "unknown")
                retrieved_rec_idxs.append(str(rec_idx))

        if retrieved_rec_idxs and gt_rec_idxs:
            metrics = evaluator.evaluate_query(retrieved_rec_idxs, list(gt_rec_idxs))
            print(f"\n📊 이 쿼리의 평가 지표:")
            print(f"   - NDCG@10: {metrics.get('ndcg@10', 0.0):.4f}")
            print(f"   - MRR@10: {metrics.get('mrr@10', 0.0):.4f}")
            print(f"   - Precision@3: {metrics.get('precision@3', 0.0):.4f}")
            print(f"   - Precision@5: {metrics.get('precision@5', 0.0):.4f}")
            print(f"   - Precision@10: {metrics.get('precision@10', 0.0):.4f}")
            print(f"   - Precision@20: {metrics.get('precision@20', 0.0):.4f}")
            print(f"   - Hit@10_count: {metrics.get('hit@10_count', 0.0):.0f}")
            print(f"   - R-recall: {metrics.get('r_recall', 0.0):.4f}")
            print(f"   - Recall@10: {metrics.get('recall@10', 0.0):.4f}")
            print(f"   - Recall@20: {metrics.get('recall@20', 0.0):.4f}")

        # 검색된 문서 상세 정보
        for rank, doc in enumerate(retrieved_docs[:20], 1):  # 상위 20개만
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                rec_idx = metadata.get("rec_idx") or metadata.get("rec_id", "unknown")
                title = metadata.get("title") or metadata.get("post_title", "제목 없음")
                company = metadata.get("company") or metadata.get("company_name", "")
                url = metadata.get("url") or metadata.get("detail_url", "")
                deadline = metadata.get("deadline", "")
                start_date = metadata.get("start_date", "")
                crawling_time = metadata.get("crawling_time", "")
                score = doc.get("score", 0.0)

                # GT 매칭 여부
                is_gt = str(rec_idx) in gt_rec_idxs
                match_marker = "✅" if is_gt else "❌"

                print(
                    f"\n   {rank}. {match_marker} rec_idx: {rec_idx} (score: {score:.4f})"
                )
                print(f"      - 제목: {title}")
                print(f"      - 회사: {company}")
                print(f"      - URL: {url}")
                if deadline:
                    print(f"      - 마감일: {deadline}")
                if start_date:
                    print(f"      - 시작일: {start_date}")
                if crawling_time:
                    print(f"      - 크롤링시간: {crawling_time}")

        # 매칭 통계
        matched_count = sum(
            1
            for doc in retrieved_docs[:20]
            if isinstance(doc, dict)
            and str(
                doc.get("metadata", {}).get("rec_idx")
                or doc.get("metadata", {}).get("rec_id", "")
            )
            in gt_rec_idxs
        )
        print(f"\n📈 매칭 통계:")
        print(f"   - GT 문서 수: {len(gt_rec_idxs)}개")
        print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
        print(f"   - 상위 20개 중 매칭: {matched_count}개")
        print(
            f"   - 매칭률: {matched_count / len(gt_rec_idxs) * 100:.1f}%"
            if gt_rec_idxs
            else "   - 매칭률: N/A"
        )

        print("=" * 80 + "\n")

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산 (tiktoken 사용)"""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"토큰 카운팅 실패: {e}")
            # 대략적인 추정 (1 토큰 ≈ 4 글자)
            return len(text) // 4

    def trim_courses_if_needed(self, query_text: str, max_tokens: int = 7500) -> str:
        """토큰 초과 시 수강 이력에서 과목을 순차적으로 제거"""

        # 1. 현재 토큰 수 체크
        if self.count_tokens(query_text) <= max_tokens:
            return query_text

        # 2. 수강 이력 부분 분리
        lines = query_text.split("\n")
        course_history_start = -1

        for i, line in enumerate(lines):
            if line.startswith("수강 이력:"):
                course_history_start = i
                break

        if course_history_start == -1:
            return query_text  # 수강 이력이 없으면 그대로 반환

        # 3. 기본 정보 부분과 수강 이력 부분 분리
        basic_info = lines[: course_history_start + 1]  # "수강 이력:" 까지 포함
        course_lines = lines[course_history_start + 1 :]

        # 4. 각 강의 블록 파싱 (강의명으로 시작하는 블록들)
        courses = []
        current_course = []

        for line in course_lines:
            if line.startswith("강의명:"):
                if current_course:  # 이전 강의 저장
                    courses.append("\n".join(current_course))
                current_course = [line]
            else:
                current_course.append(line)

        if current_course:  # 마지막 강의 저장
            courses.append("\n".join(current_course))

        # 5. 뒤에서부터 과목을 하나씩 제거하면서 토큰 수 체크
        while courses and len(courses) > 5:  # 최소 5개는 유지
            # 현재 상태로 텍스트 재구성
            trimmed_text = "\n".join(basic_info + ["\n".join(courses)])

            if self.count_tokens(trimmed_text) <= max_tokens:
                return trimmed_text

            # 마지막 과목 제거
            courses.pop()

        # 6. 최종 텍스트 반환 (5개 이하가 되어도 토큰이 초과하면 그대로 반환)
        final_text = "\n".join(basic_info + ["\n".join(courses)])
        return final_text

    async def _generate_response_for_query(
        self,
        query_data: Dict[str, Any],
        query_text: str,
        retrieved_docs: List[Dict[str, Any]],
        response_generator,
    ) -> Optional[Dict[str, Any]]:
        """개별 쿼리에 대한 응답 생성"""
        try:
            # 사용자 프로필 추출
            user_profile = query_data.get("user_profile", {})

            # 대화 이력 추출 (있다면)
            chat_history = query_data.get("chat_history", [])

            # 응답 생성
            generated_response = await response_generator.generate(
                query=query_text,
                retrieved_docs=retrieved_docs,
                user_profile=user_profile,
                chat_history=chat_history,
                config_tags=self.config.langsmith.tags,
            )

            return {
                "content": generated_response.content,
                "recommended_jobs": [
                    job.dict() for job in generated_response.recommended_jobs
                ],
            }

        except Exception as e:
            print(f"응답 생성 실패: {e}")
            return None

    async def _evaluate_retrieval(
        self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> List[QueryResult]:
        """검색 성능 평가 수행"""
        embedder = components["embedder"]
        retriever = components["retriever"]
        evaluator = components["evaluator"]

        # 디버깅 모드: 단일 쿼리만 평가
        debug_single_query = getattr(
            self.config.evaluation, "debug_single_query", False
        )
        if debug_single_query:
            print("\n" + "=" * 80)
            print("🔍 디버깅 모드: 단일 쿼리만 평가합니다")
            print("=" * 80)
            test_queries = test_queries[:1]

        # 평가 전 통계 출력
        self._print_retrieval_evaluation_statistics(test_queries)

        query_results = []
        skipped_queries = 0
        TOKEN_LIMIT = 8000  # 안전 마진 포함

        for i, query_data in enumerate(test_queries):
            # 첫 번째 쿼리 데이터 구조 확인 (디버깅용)
            if i == 0:
                print(f"첫 번째 쿼리 데이터 타입: {type(query_data)}")
                print(f"첫 번째 쿼리 내용: {str(query_data)[:200]}...")

            # 타입 체크 및 파싱
            if isinstance(query_data, str):
                try:
                    import json

                    query_data = json.loads(query_data)
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 실패, 쿼리 스킵: {e}")
                    skipped_queries += 1
                    continue

            # 딕셔너리가 아닌 경우 스킵
            if not isinstance(query_data, dict):
                print(f"잘못된 데이터 타입, 쿼리 스킵: {type(query_data)}")
                skipped_queries += 1
                continue

            # 필드명 호환성: query_text 또는 query
            query_text = query_data.get("query") or query_data.get("query_text")
            if not query_text:
                print(f"'query' 또는 'query_text' 필드 없음, 쿼리 스킵")
                skipped_queries += 1
                continue

            # 필드명 호환성: ground_truth_docs 또는 ground_truth
            ground_truth = query_data.get("ground_truth_docs") or query_data.get(
                "ground_truth", []
            )

            # ground_truth가 rec_idx만 있는 경우 dict 형태로 변환
            if (
                ground_truth
                and isinstance(ground_truth[0], dict)
                and "rec_idx" in ground_truth[0]
            ):
                ground_truth = [
                    {
                        "rec_idx": (
                            gt.get("rec_idx", gt) if isinstance(gt, dict) else gt
                        ),
                        "url": gt.get(
                            "url",
                            f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt.get('rec_idx', gt) if isinstance(gt, dict) else gt}",
                        ),
                        "job_title": gt.get("job_title", ""),
                    }
                    for gt in ground_truth
                ]
            elif ground_truth and isinstance(ground_truth[0], str):
                ground_truth = [
                    {
                        "rec_idx": gt,
                        "url": f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt}",
                        "job_title": "",
                    }
                    for gt in ground_truth
                ]

            # 토큰 수 체크 및 필요시 수강 이력 트리밍
            original_token_count = self.count_tokens(query_text)
            if original_token_count > TOKEN_LIMIT:
                print(
                    f"토큰 초과 감지 ({original_token_count}), 수강 이력 트리밍 시도..."
                )
                query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)
                new_token_count = self.count_tokens(query_text)

                if new_token_count > TOKEN_LIMIT:
                    print(
                        f"쿼리 스킵 (트리밍 후에도 토큰 초과: {new_token_count}): {query_text[:50]}..."
                    )
                    skipped_queries += 1
                    continue
                else:
                    print(
                        f"트리밍 성공: {original_token_count} → {new_token_count} 토큰"
                    )

            try:
                # 쿼리 임베딩 생성
                query_embedding = embedder.embed([query_text])[0]

                # 동적 top_k 설정: R-recall 계산을 위해 정답 개수만큼은 검색
                base_top_k = self.config.retriever.top_k
                gt_count = len(ground_truth) if ground_truth else 0
                # max(기존 top_k, 정답개수)로 설정 (최대 60개로 제한)
                # R-recall을 정확히 계산하려면 최소한 GT count만큼은 검색해야 함
                evaluation_top_k = min(max(base_top_k, gt_count), 60)

                # 디버깅: GT count가 base_top_k보다 큰 경우 로그 출력
                if gt_count > base_top_k and len(query_results) == 0:
                    print(
                        f"  ⚠️  GT count({gt_count}) > base_top_k({base_top_k}), evaluation_top_k={evaluation_top_k}로 검색"
                    )

                # 검색 수행
                search_results = retriever.search(
                    query_embedding, top_k=evaluation_top_k
                )

                # 디버깅: 실제 검색된 문서 수 확인
                if len(search_results) < evaluation_top_k and len(query_results) == 0:
                    print(
                        f"  ⚠️  검색 요청: {evaluation_top_k}개, 실제 검색: {len(search_results)}개"
                    )

                # 검색 결과 디버깅 (첫 번째 쿼리만)
                if len(query_results) == 0:
                    print(f"검색 결과 구조 디버깅:")
                    print(f"  search_results 타입: {type(search_results)}")
                    print(f"  search_results 길이: {len(search_results)}")
                    if len(search_results) > 0:
                        print(f"  첫 번째 결과 타입: {type(search_results[0])}")
                        print(f"  첫 번째 결과 내용: {str(search_results[0])[:200]}...")
                        if isinstance(search_results[0], tuple):
                            doc, score = search_results[0]
                            print(f"  doc 타입: {type(doc)}")
                            print(f"  doc 내용: {str(doc)[:200]}...")
                            print(f"  score 타입: {type(score)}")
                            print(f"  score 값: {score}")

                # QueryResult 객체 생성
                try:
                    retrieved_docs = []
                    for item in search_results:
                        if isinstance(item, tuple) and len(item) == 2:
                            doc, score = item
                            if isinstance(doc, dict):
                                metadata = doc.get("metadata", {})
                                retrieved_docs.append(
                                    {
                                        "text": doc.get("text", ""),
                                        "metadata": {
                                            **metadata,  # 모든 메타데이터 포함 (deadline, start_date, crawling_time 등)
                                            "score": score,  # 유사도 점수도 포함
                                        },
                                    }
                                )
                            else:
                                print(f"예상과 다른 doc 타입: {type(doc)}, 내용: {doc}")
                        else:
                            print(f"예상과 다른 item 구조: {type(item)}, 내용: {item}")

                    # 응답 생성 (선택적)
                    generated_response = None
                    if "response_generator" in components:
                        generated_response = await self._generate_response_for_query(
                            query_data,
                            query_text,
                            retrieved_docs,
                            components["response_generator"],
                        )

                    query_result = QueryResult(
                        query=query_text,
                        retrieved_docs=retrieved_docs,
                        ground_truth_docs=ground_truth,
                    )

                    # 생성된 응답을 query_result에 추가 (기존 구조 유지)
                    if generated_response:
                        query_result.generated_response = generated_response
                except Exception as e:
                    print(f"QueryResult 생성 실패: {e}")
                    print(f"search_results: {search_results}")
                    continue

                query_results.append(query_result)

                if (i + 1) % 10 == 0:
                    print(f"쿼리 평가 완료: {len(query_results)}/{len(test_queries)}")

            except Exception as e:
                print(f"쿼리 처리 실패: {e}")
                print(f"쿼리 인덱스: {i}")
                print(f"쿼리 텍스트: {query_text[:100]}...")
                import traceback

                traceback.print_exc()
                skipped_queries += 1
                continue

        print(f"\n처리 완료: {len(query_results)}개, 스킵: {skipped_queries}개")

        # 첫 번째 쿼리에 대한 상세 결과 출력
        if query_results:
            self._print_single_query_details(query_results[0], evaluator)

        # 평가 지표 계산
        evaluation_results = evaluator.evaluate(query_results)

        # 디버깅 모드인 경우 여기서 종료 (전체 통계 생략)
        if debug_single_query:
            print("\n" + "=" * 80)
            print("✅ 디버깅 모드: 단일 쿼리 평가 완료")
            print("=" * 80)
            return query_results

        print("\n=== 평가 결과 (사용자 요청 순서) ===")
        # 평가 지표 순서 정의 (사용자 요청 순서)
        metric_order = [
            "ndcg@10",
            "mrr@10",
            "precision@3",
            "precision@5",
            "precision@10",
            "precision@20",
            "hit@10_count",
            "r_recall",
        ]

        # 순서대로 출력
        for metric_name in metric_order:
            result = next(
                (r for r in evaluation_results if r.metric_name == metric_name), None
            )
            if result:
                if metric_name == "hit@10_count":
                    print(f"{result.metric_name}: {result.score:.2f}")
                else:
                    print(f"{result.metric_name}: {result.score:.4f}")

        # 추가 지표 출력 (하위 호환성)
        other_results = [
            r for r in evaluation_results if r.metric_name not in metric_order
        ]
        if other_results:
            print("\n=== 추가 지표 (하위 호환성) ===")
            for result in other_results:
                if result.metric_name == "hit@20_count":
                    print(f"{result.metric_name}: {result.score:.2f}")
                else:
                    print(f"{result.metric_name}: {result.score:.4f}")

        return query_results

    async def _run_dual_evaluation(
        self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """이중 평가 시스템: 전체 검색 평가 + 샘플 생성 평가"""

        dual_results = {}

        # 1. 전체 쿼리 검색 성능 평가
        print("🔍 1단계: 전체 쿼리 검색 성능 평가")
        print(f"   대상: {len(test_queries)}개 쿼리")

        retrieval_results = await self._evaluate_retrieval_only(
            test_queries, components
        )
        dual_results["retrieval_evaluation"] = {
            "query_results": retrieval_results,
            "query_count": len(retrieval_results),
        }

        # 2. 샘플 쿼리 선택 (응답 생성기가 있는 경우만)
        if "response_generator" in components:
            # 샘플링 설정 확인
            evaluation_config = getattr(self.config, "evaluation")
            generation_config = evaluation_config.generation

            # generation.target이 "none"이면 생성 평가 건너뛰기
            if generation_config.target == "none":
                print("\n⚠️  생성 평가가 비활성화되어 있습니다 (target: none).")
                dual_results["generation_evaluation"] = None
                return dual_results

            print("\n🎯 2단계: 샘플 쿼리 선택")

            # 기본값 설정
            sample_size = generation_config.sample_size
            sample_strategy = generation_config.sample_strategy
            sample_seed = generation_config.sample_seed

            # 시드 생성 (설정되지 않은 경우)
            if sample_seed is None:
                config_dict = {
                    "embedder": self.config.embedder.__dict__,
                    "chunker": self.config.chunker.__dict__,
                    "retriever": self.config.retriever.__dict__,
                }
                sample_seed = generate_reproducible_seed(config_dict)

            print(f"   샘플 크기: {sample_size}")
            print(f"   샘플링 전략: {sample_strategy}")
            print(f"   시드: {sample_seed}")

            # 샘플링 수행
            sampler = StratifiedSampler(seed=sample_seed)
            sampled_queries = sampler.sample_queries(
                test_queries, sample_size=sample_size, strategy=sample_strategy
            )

            # 샘플링 분포 분석
            distribution_analysis = analyze_sample_distribution(
                test_queries, sampled_queries
            )
            print(f"   샘플링 비율: {distribution_analysis['sampling_ratio']:.2%}")

            # 3. 샘플 쿼리 검색 + 응답 생성 평가
            print("\n🤖 3단계: 샘플 쿼리 응답 생성 평가")
            print(f"   대상: {len(sampled_queries)}개 쿼리")

            generation_results = await self._evaluate_generation_for_samples(
                sampled_queries, components
            )

            dual_results["generation_evaluation"] = {
                "sampled_queries": sampled_queries,
                "query_results": generation_results,
                "sample_config": {
                    "sample_size": len(sampled_queries),
                    "sample_strategy": sample_strategy,
                    "sample_seed": sample_seed,
                },
                "distribution_analysis": distribution_analysis,
            }

        else:
            print("\n⚠️  응답 생성기가 설정되지 않아 생성 평가를 건너뜁니다.")
            dual_results["generation_evaluation"] = None

        return dual_results

    async def _evaluate_retrieval_only(
        self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> List[QueryResult]:
        """검색 성능만 평가 (응답 생성 없음)"""

        embedder = components["embedder"]
        retriever = components["retriever"]

        # 평가 전 통계 출력
        self._print_retrieval_evaluation_statistics(test_queries)

        query_results = []
        skipped_queries = 0
        TOKEN_LIMIT = 8000

        for i, query_data in enumerate(test_queries):
            try:
                # 기존 _evaluate_retrieval과 동일한 전처리
                if isinstance(query_data, str):
                    try:
                        import json

                        query_data = json.loads(query_data)
                    except json.JSONDecodeError as e:
                        skipped_queries += 1
                        continue

                if not isinstance(query_data, dict):
                    skipped_queries += 1
                    continue

                # 필드명 호환성: query_text 또는 query
                query_text = query_data.get("query") or query_data.get("query_text")
                if not query_text:
                    skipped_queries += 1
                    continue

                # 필드명 호환성: ground_truth_docs 또는 ground_truth
                ground_truth = query_data.get("ground_truth_docs") or query_data.get(
                    "ground_truth", []
                )

                # ground_truth가 rec_idx만 있는 경우 dict 형태로 변환
                if (
                    ground_truth
                    and isinstance(ground_truth[0], dict)
                    and "rec_idx" in ground_truth[0]
                ):
                    # 이미 dict 형태이지만 rec_idx만 있는 경우, url이 없으면 추가
                    ground_truth = [
                        {
                            "rec_idx": (
                                gt.get("rec_idx", gt) if isinstance(gt, dict) else gt
                            ),
                            "url": gt.get(
                                "url",
                                f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt.get('rec_idx', gt) if isinstance(gt, dict) else gt}",
                            ),
                            "job_title": gt.get("job_title", ""),
                        }
                        for gt in ground_truth
                    ]
                elif ground_truth and isinstance(ground_truth[0], str):
                    # rec_idx 문자열 리스트인 경우
                    ground_truth = [
                        {
                            "rec_idx": gt,
                            "url": f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt}",
                            "job_title": "",
                        }
                        for gt in ground_truth
                    ]

                # 토큰 수 체크 및 트리밍
                original_token_count = self.count_tokens(query_text)
                if original_token_count > TOKEN_LIMIT:
                    query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)
                    new_token_count = self.count_tokens(query_text)

                    if new_token_count > TOKEN_LIMIT:
                        skipped_queries += 1
                        continue

                # 검색만 수행 (응답 생성 없음)
                query_embedding = embedder.embed([query_text])[0]
                search_results = retriever.search(
                    query_embedding, top_k=self.config.retriever.top_k
                )

                # QueryResult 생성
                retrieved_docs = []
                for item in search_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        if isinstance(doc, dict):
                            retrieved_docs.append(
                                {
                                    "text": doc.get("text", ""),
                                    "metadata": doc.get("metadata", {}),
                                }
                            )

                query_result = QueryResult(
                    query=query_text,
                    retrieved_docs=retrieved_docs,
                    ground_truth_docs=ground_truth,
                )

                query_results.append(query_result)

                if (i + 1) % 50 == 0:
                    print(
                        f"   검색 평가 진행률: {len(query_results)}/{len(test_queries)}"
                    )

            except Exception as e:
                print(f"   쿼리 {i} 검색 평가 실패: {e}")
                skipped_queries += 1
                continue

        print(
            f"   검색 평가 완료: {len(query_results)}개 성공, {skipped_queries}개 스킵"
        )
        return query_results

    async def _evaluate_generation_for_samples(
        self, sampled_queries: List[Dict[str, Any]], components: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """샘플 쿼리들에 대한 검색 + 응답 생성 평가"""

        embedder = components["embedder"]
        retriever = components["retriever"]
        response_generator = components["response_generator"]

        generation_results = []
        TOKEN_LIMIT = 8000

        for i, query_data in enumerate(sampled_queries):
            try:
                # 필드명 호환성: query_text 또는 query
                query_text = query_data.get("query") or query_data.get("query_text")
                if not query_text:
                    print(
                        f"   샘플 쿼리 {i} 생성 평가 실패: 'query' 또는 'query_text' 필드 없음"
                    )
                    continue

                # 필드명 호환성: ground_truth_docs 또는 ground_truth
                ground_truth = query_data.get("ground_truth_docs") or query_data.get(
                    "ground_truth", []
                )

                # ground_truth가 rec_idx만 있는 경우 dict 형태로 변환
                if (
                    ground_truth
                    and isinstance(ground_truth[0], dict)
                    and "rec_idx" in ground_truth[0]
                ):
                    ground_truth = [
                        {
                            "rec_idx": (
                                gt.get("rec_idx", gt) if isinstance(gt, dict) else gt
                            ),
                            "url": gt.get(
                                "url",
                                f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt.get('rec_idx', gt) if isinstance(gt, dict) else gt}",
                            ),
                            "job_title": gt.get("job_title", ""),
                        }
                        for gt in ground_truth
                    ]
                elif ground_truth and isinstance(ground_truth[0], str):
                    ground_truth = [
                        {
                            "rec_idx": gt,
                            "url": f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt}",
                            "job_title": "",
                        }
                        for gt in ground_truth
                    ]

                # 토큰 수 체크 및 트리밍
                original_token_count = self.count_tokens(query_text)
                if original_token_count > TOKEN_LIMIT:
                    query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)

                # 검색 수행
                query_embedding = embedder.embed([query_text])[0]

                # 동적 top_k 설정: R-recall 계산을 위해 정답 개수만큼은 검색
                base_top_k = self.config.retriever.top_k
                gt_count = len(ground_truth) if ground_truth else 0
                # max(기존 top_k, 정답개수)로 설정 (최대 60개로 제한)
                evaluation_top_k = min(max(base_top_k, gt_count), 60)

                search_results = retriever.search(
                    query_embedding, top_k=evaluation_top_k
                )

                # 검색 결과 정리 (메타데이터 전체 포함)
                retrieved_docs = []
                for item in search_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        if isinstance(doc, dict):
                            metadata = doc.get("metadata", {})

                            # 디버깅: 첫 번째 검색 결과의 메타데이터 확인
                            if len(retrieved_docs) == 0:
                                print(
                                    f"\n🔍 [생성 평가] 첫 번째 검색 결과 메타데이터 확인:"
                                )
                                print(f"   - rec_idx: {metadata.get('rec_idx', 'N/A')}")
                                print(
                                    f"   - deadline: {metadata.get('deadline', 'N/A')}"
                                )
                                print(
                                    f"   - start_date: {metadata.get('start_date', 'N/A')}"
                                )
                                print(
                                    f"   - crawling_time: {metadata.get('crawling_time', 'N/A')}"
                                )
                                print(
                                    f"   - 전체 메타데이터 키: {list(metadata.keys())[:15]}"
                                )

                            retrieved_docs.append(
                                {
                                    "text": doc.get("text", ""),
                                    "metadata": {
                                        **metadata,  # 모든 메타데이터 포함 (deadline, start_date, crawling_time 등)
                                        "score": score,  # 유사도 점수도 포함
                                    },
                                }
                            )

                # 응답 생성
                generated_response = await self._generate_response_for_query(
                    query_data, query_text, retrieved_docs, response_generator
                )

                # 결과 저장
                result = {
                    "query": query_text,
                    "user_profile": query_data.get("user_profile", {}),
                    "ground_truth_docs": ground_truth,
                    "retrieved_docs": retrieved_docs,
                    "generated_response": generated_response,
                    "alternative_query": query_data.get("metadata", {}).get(
                        "alternative_query", ""
                    ),
                }

                generation_results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"   생성 평가 진행률: {i + 1}/{len(sampled_queries)}")

            except Exception as e:
                print(f"   샘플 쿼리 {i} 생성 평가 실패: {e}")
                continue

        print(f"   생성 평가 완료: {len(generation_results)}개 성공")
        return generation_results

    async def _run_langsmith_evaluation_if_enabled(
        self,
        generation_query_results: List[Dict[str, Any]],
        langsmith_evaluation_results: List,
    ):
        """LangSmith 평가 실행 (설정된 경우만)"""

        # LangSmith 설정 확인
        langsmith_config = getattr(self.config, "langsmith", None)
        print(f"🔍 LangSmith 설정 확인: {langsmith_config}")

        if not langsmith_config or not langsmith_config.enabled:
            print("\n⚠️  LangSmith 평가가 비활성화되어 있어 건너뜁니다.")
            return

        # 환경변수 확인
        import os

        api_key = os.getenv("LANGCHAIN_API_KEY")
        print(f"🔍 LANGCHAIN_API_KEY 존재: {bool(api_key)}")
        if not api_key:
            print("\n⚠️  LANGCHAIN_API_KEY가 설정되지 않아 LangSmith 평가를 건너뜁니다.")
            return

        try:
            print("\n=== LangSmith 고품질 평가 ===")

            # LangSmith 평가기 초기화
            judge_model = langsmith_config.judge_model
            project_name = langsmith_config.project_name
            print(f"🔍 Judge 모델: {judge_model}, 프로젝트: {project_name}")
            print(f"🔍 평가할 쿼리 수: {len(generation_query_results)}")

            langsmith_evaluator = CareerHYLangSmithEvaluator(
                judge_model=judge_model, project_name=project_name
            )
            print("✅ LangSmith 평가기 초기화 완료")

            # 평가 실행
            print("🚀 LangSmith 평가 시작...")
            evaluation_results = await langsmith_evaluator.evaluate_batch(
                generation_query_results, experiment_name=self.config.experiment_name
            )
            print(f"✅ LangSmith 평가 완료: {len(evaluation_results)}개 결과")

            langsmith_evaluation_results.extend(evaluation_results)

        except Exception as e:
            print(f"❌ LangSmith 평가 실패: {e}")
            print("자동화된 평가 결과만 사용합니다.")

    async def _save_dual_results(
        self,
        dual_results: Dict[str, Any],
        components: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """이중 평가 결과 저장"""

        # 1. 검색 성능 평가
        retrieval_evaluation = dual_results["retrieval_evaluation"]
        retrieval_evaluation_results = []
        retrieval_query_results = []  # 기본값 설정

        if retrieval_evaluation is not None:
            # 검색 평가가 활성화된 경우만 실행
            retrieval_query_results = retrieval_evaluation["query_results"]

            # evaluator가 없으면 직접 생성
            if "evaluator" not in components:
                from implementations.evaluators import RetrieverEvaluator

                evaluator = RetrieverEvaluator()
            else:
                evaluator = components["evaluator"]

            retrieval_evaluation_results = evaluator.evaluate(retrieval_query_results)

            print("\n=== 검색 성능 평가 결과 (사용자 요청 순서) ===")
            # 평가 지표 순서 정의 (사용자 요청 순서)
            metric_order = [
                "ndcg@10",
                "mrr@10",
                "precision@3",
                "precision@5",
                "precision@10",
                "precision@20",
                "hit@10_count",
                "r_recall",
            ]

            # 순서대로 출력
            for metric_name in metric_order:
                result = next(
                    (
                        r
                        for r in retrieval_evaluation_results
                        if r.metric_name == metric_name
                    ),
                    None,
                )
                if result:
                    if metric_name == "hit@10_count":
                        print(f"{result.metric_name}: {result.score:.2f}")
                    else:
                        print(f"{result.metric_name}: {result.score:.4f}")

            # 추가 지표 출력 (하위 호환성)
            other_results = [
                r
                for r in retrieval_evaluation_results
                if r.metric_name not in metric_order
            ]
            if other_results:
                print("\n=== 추가 지표 (하위 호환성) ===")
                for result in other_results:
                    if result.metric_name == "hit@20_count":
                        print(f"{result.metric_name}: {result.score:.2f}")
                    else:
                        print(f"{result.metric_name}: {result.score:.4f}")
        else:
            print("\n=== 검색 성능 평가 생략됨 ===")
            print("   (프로필 기반 검색 시스템 - GT 준비 필요)")

        # 2. 생성 품질 평가 (응답 생성기가 있는 경우)
        langsmith_evaluation_results = []

        if dual_results["generation_evaluation"] is not None:
            generation_evaluation = dual_results["generation_evaluation"]
            generation_query_results = generation_evaluation["query_results"]

            if generation_query_results:
                print("\n=== LangSmith 정성평가 실행 ===")
                # LangSmith 평가만 실행
                await self._run_langsmith_evaluation_if_enabled(
                    generation_query_results, langsmith_evaluation_results
                )

        # 3. 결과 딕셔너리 구성
        results = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "description": self.config.description,
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "evaluation_type": "dual",  # 이중 평가 표시
            },
            "config": {
                "embedder": asdict(self.config.embedder),
                "chunker": asdict(self.config.chunker),
                "retriever": asdict(self.config.retriever),
                "evaluation": {
                    "retrieval": asdict(self.config.evaluation.retrieval),
                    "generation": asdict(self.config.evaluation.generation),
                },
                "langsmith": (
                    asdict(self.config.langsmith) if self.config.langsmith else None
                ),
            },
            "component_info": {
                name: (
                    comp.get_model_info()
                    if hasattr(comp, "get_model_info")
                    else (
                        comp.get_chunker_info()
                        if hasattr(comp, "get_chunker_info")
                        else (
                            comp.get_retriever_info()
                            if hasattr(comp, "get_retriever_info")
                            else {}
                        )
                    )
                )
                for name, comp in components.items()
                if hasattr(comp, "__dict__")
            },
            "retrieval_evaluation": {
                "query_count": len(retrieval_query_results),
                "metrics": [
                    {
                        "metric": result.metric_name,
                        "score": result.score,
                        "details": result.details,
                    }
                    for result in retrieval_evaluation_results
                ],
            },
            "document_count": components["retriever"].get_document_count(),
        }

        # 생성 평가 결과 추가
        if dual_results["generation_evaluation"] is not None:
            generation_evaluation = dual_results["generation_evaluation"]

            results["generation_evaluation"] = {
                "sample_count": len(generation_evaluation["query_results"]),
                "sample_config": generation_evaluation["sample_config"],
                "distribution_analysis": generation_evaluation["distribution_analysis"],
                "langsmith_metrics": (
                    [
                        {
                            "metric": result.metric_name,
                            "score": result.score,
                            "reasoning": result.reasoning,
                            "details": result.details,
                        }
                        for result in langsmith_evaluation_results
                    ]
                    if langsmith_evaluation_results
                    else []
                ),
            }

            # 응답 생성기 설정 추가
            if hasattr(self.config, "response_generator"):
                results["config"]["response_generator"] = asdict(
                    self.config.response_generator
                )

            # LangSmith 설정 추가 (이미 위에서 처리되었음)
            # if hasattr(self.config, 'langsmith'):
            #     results["config"]["langsmith"] = asdict(self.config.langsmith)

        # 4. 결과 파일 저장
        try:
            # 디렉토리 확인 및 생성
            self.output_dir.mkdir(parents=True, exist_ok=True)

            results_file = self.output_dir / f"results_{self.experiment_id}.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 요약 결과 저장: {results_file.absolute()}")
        except Exception as e:
            print(f"❌ 요약 결과 저장 실패: {e}")
            import traceback

            traceback.print_exc()

        # 5. 상세 결과 저장
        # 검색 결과
        try:
            retrieval_detailed_file = (
                self.output_dir / f"retrieval_detailed_{self.experiment_id}.jsonl"
            )
            with open(retrieval_detailed_file, "w", encoding="utf-8") as f:
                for qr in retrieval_query_results:
                    # 검색된 문서 메타데이터
                    retrieved_docs_detail = []
                    for doc in qr.retrieved_docs:
                        if isinstance(doc, dict):
                            metadata = doc.get("metadata", {})
                            rec_idx = metadata.get("rec_idx") or metadata.get(
                                "rec_id", "unknown"
                            )
                            retrieved_docs_detail.append(
                                {
                                    "rec_idx": rec_idx,
                                    "title": (
                                        metadata.get("title")
                                        or metadata.get("post_title")
                                        or ""
                                    ),
                                    "company": (
                                        metadata.get("company")
                                        or metadata.get("company_name")
                                        or ""
                                    ),
                                    "url": (
                                        metadata.get("url")
                                        or metadata.get("detail_url")
                                        or (
                                            f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_idx}"
                                            if rec_idx != "unknown"
                                            else ""
                                        )
                                    ),
                                    "deadline": metadata.get("deadline"),
                                    "start_date": metadata.get("start_date"),
                                    "crawling_time": metadata.get("crawling_time"),
                                    "score": doc.get("score", 0.0),
                                }
                            )

                    # GT 문서 메타데이터
                    # retriever에서 메타데이터 조회 가능 여부 확인
                    retriever = components.get("retriever")
                    has_get_metadata = retriever and hasattr(
                        retriever, "get_metadata_by_rec_idx"
                    )

                    ground_truth_detail = []
                    for gt in qr.ground_truth_docs:
                        if isinstance(gt, dict):
                            rec_idx = gt.get("rec_idx") or gt.get("rec_id", "unknown")

                            # 기존 메타데이터에서 title과 company 가져오기
                            title = (
                                gt.get("title")
                                or gt.get("job_title")
                                or gt.get("post_title")
                                or ""
                            )
                            company = gt.get("company") or gt.get("company_name") or ""

                            # title이나 company가 비어있고 retriever에서 조회 가능하면 조회
                            if (
                                (not title or not company)
                                and has_get_metadata
                                and rec_idx != "unknown"
                            ):
                                try:
                                    metadata = retriever.get_metadata_by_rec_idx(
                                        rec_idx
                                    )
                                    if metadata:
                                        if not title:
                                            title = (
                                                metadata.get("title")
                                                or metadata.get("post_title")
                                                or ""
                                            )
                                        if not company:
                                            company = (
                                                metadata.get("company")
                                                or metadata.get("company_name")
                                                or ""
                                            )
                                except Exception as e:
                                    # 조회 실패해도 계속 진행
                                    pass

                            ground_truth_detail.append(
                                {
                                    "rec_idx": rec_idx,
                                    "title": title,
                                    "company": company,
                                    "url": (
                                        gt.get("url")
                                        or gt.get("detail_url")
                                        or (
                                            f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_idx}"
                                            if rec_idx != "unknown"
                                            else ""
                                        )
                                    ),
                                    "deadline": gt.get("deadline"),
                                    "start_date": gt.get("start_date"),
                                    "crawling_time": gt.get("crawling_time"),
                                }
                            )
                        else:
                            rec_idx_str = str(gt)

                            # retriever에서 메타데이터 조회
                            title = ""
                            company = ""
                            if has_get_metadata and rec_idx_str != "unknown":
                                try:
                                    metadata = retriever.get_metadata_by_rec_idx(
                                        rec_idx_str
                                    )
                                    if metadata:
                                        title = (
                                            metadata.get("title")
                                            or metadata.get("post_title")
                                            or ""
                                        )
                                        company = (
                                            metadata.get("company")
                                            or metadata.get("company_name")
                                            or ""
                                        )
                                except Exception as e:
                                    # 조회 실패해도 계속 진행
                                    pass

                            ground_truth_detail.append(
                                {
                                    "rec_idx": rec_idx_str,
                                    "title": title,
                                    "company": company,
                                    "url": f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={rec_idx_str}",
                                    "deadline": None,
                                    "start_date": None,
                                    "crawling_time": None,
                                }
                            )

                    query_detail = {
                        "query": qr.query,
                        "ground_truth_count": len(qr.ground_truth_docs),
                        "retrieved_count": len(qr.retrieved_docs),
                        "retrieved_docs": retrieved_docs_detail,
                        "ground_truth_docs": ground_truth_detail,
                    }
                    f.write(json.dumps(query_detail, ensure_ascii=False) + "\n")
            print(f"✅ 검색 상세 결과 저장: {retrieval_detailed_file.absolute()}")
        except Exception as e:
            print(f"❌ 검색 상세 결과 저장 실패: {e}")
            import traceback

            traceback.print_exc()

        # 생성 결과 (있다면)
        if dual_results["generation_evaluation"] is not None:
            generation_evaluation = dual_results["generation_evaluation"]
            generation_results = generation_evaluation["query_results"]

            if generation_results:
                try:
                    generation_detailed_file = (
                        self.output_dir
                        / f"generation_detailed_{self.experiment_id}.jsonl"
                    )
                    with open(generation_detailed_file, "w", encoding="utf-8") as f:
                        for result in generation_results:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(
                        f"✅ 생성 상세 결과 저장: {generation_detailed_file.absolute()}"
                    )
                except Exception as e:
                    print(f"❌ 생성 상세 결과 저장 실패: {e}")
                    import traceback

                    traceback.print_exc()

                try:
                    # 생성된 응답만 별도 저장
                    responses_file = (
                        self.output_dir
                        / f"generated_responses_{self.experiment_id}.json"
                    )
                    responses_only = []
                    for result in generation_results:
                        if result.get("generated_response"):
                            responses_only.append(
                                {
                                    "query": result["query"],
                                    "response": result["generated_response"],
                                }
                            )

                    with open(responses_file, "w", encoding="utf-8") as f:
                        json.dump(responses_only, f, ensure_ascii=False, indent=2)
                    print(f"✅ 생성된 응답 저장: {responses_file.absolute()}")
                except Exception as e:
                    print(f"❌ 생성된 응답 저장 실패: {e}")
                    import traceback

                    traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"✅ 결과 저장 완료")
        print(f"{'='*60}")
        print(f"📁 저장 디렉토리: {self.output_dir.absolute()}")
        if "results_file" in locals():
            print(f"📄 요약 결과: {results_file.name} ({results_file.absolute()})")
        if "retrieval_detailed_file" in locals():
            print(
                f"📄 검색 상세 결과: {retrieval_detailed_file.name} ({retrieval_detailed_file.absolute()})"
            )
        if dual_results["generation_evaluation"] is not None:
            generation_evaluation = dual_results["generation_evaluation"]
            if generation_evaluation.get("query_results"):
                print(
                    f"📄 생성 상세 결과: generation_detailed_{self.experiment_id}.jsonl"
                )
                print(f"📄 생성된 응답: generated_responses_{self.experiment_id}.json")
        print(f"{'='*60}\n")

        return results

    async def _run_retrieval_only_evaluation(
        self,
        test_queries: List[Dict[str, Any]],
        components: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """
        검색 성능만 평가 (생성 평가 제외)

        Args:
            test_queries: evaluation_queries.jsonl에서 로드된 쿼리 (GT 포함)
            components: 파이프라인 컴포넌트들
            start_time: 실험 시작 시간

        Returns:
            평가 결과 딕셔너리
        """
        print(f"📊 검색 성능 평가 시작: {len(test_queries)}개 쿼리")

        retriever = components["retriever"]
        embedder = components["embedder"]
        evaluator = RetrieverEvaluator(ground_truth_size=5)

        results = []
        total_search_time = 0.0

        for idx, query_data in enumerate(test_queries, start=1):
            query_id = query_data["query_id"]
            query_text = query_data["query_text"]
            ground_truth = query_data["ground_truth"]

            if idx % 10 == 0:
                print(f"  처리 중: {idx}/{len(test_queries)}")

            try:
                # ⏱️ 검색 시간 측정 시작
                search_start = time.time()

                # 1. 토큰 초과 시 수강 이력 트리밍 (기존 로직 활용)
                TOKEN_LIMIT = 8000
                original_token_count = self.count_tokens(query_text)

                if original_token_count > TOKEN_LIMIT:
                    query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)
                    new_token_count = self.count_tokens(query_text)

                    if idx == 1:  # 첫 번째 쿼리에서만 로그 출력
                        print(
                            f"  ⚠️  토큰 초과 감지: 트리밍 {original_token_count} → {new_token_count} 토큰"
                        )

                # 2. 쿼리 임베딩 변환 (embed는 List[str]을 받으므로 리스트로 감싸기)
                query_embedding = embedder.embed([query_text])[0]

                # 3. 동적 top_k 설정: R-recall 계산을 위해 정답 개수만큼은 검색
                base_top_k = self.config.retriever.top_k
                gt_count = len(ground_truth) if ground_truth else 0
                # max(기존 top_k, 정답개수)로 설정 (최대 60개로 제한)
                # R-recall을 정확히 계산하려면 최소한 GT count만큼은 검색해야 함
                evaluation_top_k = min(max(base_top_k, gt_count), 60)

                # 디버깅: GT count가 base_top_k보다 큰 경우 로그 출력
                if gt_count > base_top_k and idx == 1:
                    print(
                        f"  ⚠️  Query {query_id}: GT count({gt_count}) > base_top_k({base_top_k}), evaluation_top_k={evaluation_top_k}로 검색"
                    )

                # 4. 검색 실행
                search_results = retriever.search(
                    query_embedding, top_k=evaluation_top_k
                )

                # 디버깅: 실제 검색된 문서 수 확인
                if len(search_results) < evaluation_top_k and idx == 1:
                    print(
                        f"  ⚠️  Query {query_id}: 검색 요청 {evaluation_top_k}개, 실제 검색 {len(search_results)}개"
                    )

                # ⏱️ 검색 시간 측정 종료
                search_time = time.time() - search_start
                total_search_time += search_time

                # 3. rec_idx 추출 (search는 List[Tuple[doc, score]] 반환)
                retrieved_rec_idxs = []
                retrieved_docs = []
                for doc_tuple in search_results:
                    doc = doc_tuple[0]  # (doc, score) 튜플의 첫 번째 요소
                    score = doc_tuple[1]  # 유사도 점수

                    rec_idx = doc.get("metadata", {}).get("rec_idx")
                    if rec_idx:
                        retrieved_rec_idxs.append(str(rec_idx))
                        retrieved_docs.append(
                            {
                                "metadata": doc.get("metadata", {}),
                                "score": score,
                                "text": doc.get("text", ""),
                            }
                        )

                gt_rec_idxs = [str(gt["rec_idx"]) for gt in ground_truth]

                # 지표 계산
                metrics = evaluator.evaluate_query(
                    retrieved_rec_idxs, gt_rec_idxs, search_time=search_time
                )

                # retriever에서 메타데이터 조회 가능 여부 확인
                has_get_metadata = retriever and hasattr(
                    retriever, "get_metadata_by_rec_idx"
                )

                # GT 문서에 title과 company 추가 (비어있는 경우)
                enriched_ground_truth = []
                for gt in ground_truth:
                    gt_rec_idx = str(gt.get("rec_idx", gt))

                    # 기존 메타데이터에서 title과 company 가져오기
                    title = (
                        gt.get("title")
                        or gt.get("job_title")
                        or gt.get("post_title")
                        or ""
                    )
                    company = gt.get("company") or gt.get("company_name") or ""

                    # title이나 company가 비어있고 retriever에서 조회 가능하면 조회
                    if (
                        (not title or not company)
                        and has_get_metadata
                        and gt_rec_idx != "unknown"
                    ):
                        try:
                            metadata = retriever.get_metadata_by_rec_idx(gt_rec_idx)
                            if metadata:
                                if not title:
                                    title = (
                                        metadata.get("title")
                                        or metadata.get("post_title")
                                        or ""
                                    )
                                if not company:
                                    company = (
                                        metadata.get("company")
                                        or metadata.get("company_name")
                                        or ""
                                    )
                        except Exception as e:
                            # 조회 실패해도 계속 진행
                            pass

                    enriched_ground_truth.append(
                        {
                            "rec_idx": gt_rec_idx,
                            "title": title,
                            "company": company,
                            "url": gt.get("url")
                            or gt.get(
                                "detail_url",
                                f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={gt_rec_idx}",
                            ),
                            "deadline": gt.get("deadline"),
                            "start_date": gt.get("start_date"),
                            "crawling_time": gt.get("crawling_time"),
                        }
                    )

                # 결과 저장 (메타데이터 포함)
                # 메타데이터에서 모든 필드를 정확히 추출
                result = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "retrieved_docs": [
                        {
                            "rank": i + 1,
                            "rec_idx": retrieved_rec_idxs[i],
                            "score": retrieved_docs[i]["score"],
                            "title": (
                                retrieved_docs[i]["metadata"].get("title")
                                or retrieved_docs[i]["metadata"].get("post_title")
                                or "제목 없음"
                            ),
                            "company": (
                                retrieved_docs[i]["metadata"].get("company")
                                or retrieved_docs[i]["metadata"].get("company_name")
                                or ""
                            ),
                            "url": (
                                retrieved_docs[i]["metadata"].get("url")
                                or retrieved_docs[i]["metadata"].get("detail_url")
                                or f"https://www.saramin.co.kr/zf_user/jobs/relay/view?view_type=public-recruit&rec_idx={retrieved_rec_idxs[i]}"
                            ),
                            "deadline": retrieved_docs[i]["metadata"].get("deadline"),
                            "start_date": retrieved_docs[i]["metadata"].get(
                                "start_date"
                            ),
                            "crawling_time": retrieved_docs[i]["metadata"].get(
                                "crawling_time"
                            ),
                        }
                        for i in range(len(retrieved_rec_idxs))
                    ],
                    "ground_truth": enriched_ground_truth,
                    "metrics": metrics,
                }
                results.append(result)

            except Exception as e:
                print(f"  ❌ 쿼리 {query_id} 처리 실패: {e}")
                continue

        # 전체 집계
        print(f"\n📈 평가 결과 집계 중...")
        summary = evaluator.evaluate_all_queries(results)

        # 검색 시간 통계 추가
        if results:
            search_times = [
                r["metrics"]["search_time"]
                for r in results
                if "search_time" in r["metrics"]
            ]
            summary["total_search_time"] = round(total_search_time, 3)
            summary["average_search_time_per_query"] = round(
                total_search_time / len(results), 3
            )
            summary["search_time_stats"] = {
                "min": round(min(search_times), 3) if search_times else 0,
                "max": round(max(search_times), 3) if search_times else 0,
            }

        # 전체 실험 시간
        total_experiment_time = time.time() - start_time
        summary["total_experiment_time"] = round(total_experiment_time, 3)

        # 결과 출력
        from implementations.evaluators.retrieval_evaluator import (
            print_evaluation_summary,
        )

        print_evaluation_summary(summary)

        return {"results": results, "summary": summary}

    def _save_retrieval_results(
        self, eval_results: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """
        검색 평가 결과 저장

        Args:
            eval_results: _run_retrieval_only_evaluation의 결과
            start_time: 실험 시작 시간

        Returns:
            저장된 결과 정보
        """
        results = eval_results["results"]
        summary = eval_results["summary"]

        # 결과 저장 디렉토리
        save_results = self.config.evaluation.save_results

        if save_results:
            # 타임스탬프
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 상세 결과 JSONL 저장
            results_dir = Path(self.config.evaluation.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            search_results_file = (
                results_dir
                / f"search_results_{self.config.experiment_name}_{timestamp}.jsonl"
            )

            with open(search_results_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"  ✅ 검색 결과 저장: {search_results_file}")

            # 2. 요약 결과 JSON 저장
            summary_file = (
                results_dir / f"summary_{self.config.experiment_name}_{timestamp}.json"
            )

            summary_data = {
                "experiment_name": self.config.experiment_name,
                "experiment_id": self.experiment_id,
                "timestamp": timestamp,
                "config": {
                    "embedder": self.config.embedder.model_name,
                    "retriever": self.config.retriever.type,
                    "top_k": self.config.retriever.top_k,
                },
                **summary,
            }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

            print(f"  ✅ 요약 결과 저장: {summary_file}")

            return {
                "summary": summary_data,
                "search_results_file": str(search_results_file),
                "summary_file": str(summary_file),
            }
        else:
            print("  ⚠️  결과 저장 스킵 (save_results=false)")
            return {"summary": summary}
