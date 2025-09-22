"""
Career-HY RAG 실험 파이프라인 메인 로직
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import ExperimentConfig
from .interfaces.evaluator import QueryResult
from utils.factory import ComponentFactory
from utils.data_loader import S3DataLoader
from utils.embedding_cache import embedding_cache
from implementations.evaluators import SearchMetricsEvaluator


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

    def run(self) -> Dict[str, Any]:
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

            # 5. Ground Truth 쿼리 로드
            print("\n=== 5. Ground Truth 쿼리 로드 ===")
            test_queries = self._load_test_queries()

            # 6. 검색 성능 평가
            print("\n=== 6. 검색 성능 평가 ===")
            query_results = self._evaluate_retrieval(test_queries, components)

            # 7. 결과 저장
            print("\n=== 7. 결과 저장 ===")
            results = self._save_results(query_results, components, start_time)

            print(f"\n실험 완료! 총 소요시간: {time.time() - start_time:.2f}초")
            return results

        except Exception as e:
            print(f"\n실험 실행 중 오류 발생: {e}")
            raise

    def _initialize_components(self) -> Dict[str, Any]:
        """설정에 따라 컴포넌트들 초기화"""
        components = {}

        # 임베딩 모델 초기화
        print(f"임베딩 모델 초기화: {self.config.embedder.type} - {self.config.embedder.model_name}")
        components['embedder'] = ComponentFactory.create_embedder(self.config.embedder)

        # 청킹 전략 초기화
        print(f"청킹 전략 초기화: {self.config.chunker.type}")
        components['chunker'] = ComponentFactory.create_chunker(self.config.chunker)

        # 검색 시스템 초기화
        print(f"검색 시스템 초기화: {self.config.retriever.type}")
        components['retriever'] = ComponentFactory.create_retriever(self.config.retriever)

        # LLM 모델 초기화 (필요한 경우)
        if hasattr(self.config, 'llm') and self.config.llm:
            print(f"LLM 모델 초기화: {self.config.llm.type} - {self.config.llm.model_name}")
            components['llm'] = ComponentFactory.create_llm(self.config.llm)

        # 평가기 초기화
        components['evaluator'] = SearchMetricsEvaluator(
            k_values=self.config.evaluation.k_values
        )

        return components

    def _load_documents(self) -> List[Dict[str, Any]]:
        """S3에서 모든 문서 데이터 로드"""
        data_loader = S3DataLoader(bucket_name=self.config.data.s3_bucket)

        documents = data_loader.load_documents(
            pdf_prefix=self.config.data.pdf_prefix,
            json_prefix=self.config.data.json_prefix
        )

        print(f"로드된 문서 수: {len(documents)}")
        return documents

    def _process_documents(self, documents: List[Dict[str, Any]], components: Dict[str, Any]) -> tuple:
        """문서 청킹 및 임베딩 처리 (캐싱 지원)"""
        chunker = components['chunker']
        embedder = components['embedder']

        # 캐시 키 생성
        cache_key = embedding_cache.generate_cache_key(
            self.config.embedder,
            self.config.chunker
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
            chunks = chunker.chunk(doc['text'], doc['metadata'])
            all_chunks.extend(chunks)

            # 임베딩용 텍스트 추출
            for chunk in chunks:
                all_texts.append(chunk['text'])

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
            "chunker_config": self.config.chunker.__dict__
        }

        try:
            embedding_cache.save(cache_key, all_chunks, embeddings, additional_info)
        except Exception as e:
            print(f"⚠️  캐시 저장 실패 (실험은 계속 진행): {e}")

        return all_chunks, embeddings

    def _build_retrieval_system(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], components: Dict[str, Any]) -> None:
        """검색 시스템에 문서와 임베딩 추가"""
        retriever = components['retriever']

        print("검색 시스템에 문서 추가 중...")

        # 기존 컬렉션 초기화 (실험용)
        if hasattr(retriever, 'clear_collection'):
            retriever.clear_collection()

        # 문서와 임베딩을 검색 시스템에 추가
        retriever.add_documents(documents, embeddings)

        doc_count = retriever.get_document_count()
        print(f"검색 시스템 구축 완료: {doc_count}개 문서")

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Ground Truth 테스트 쿼리 로드"""
        test_queries_path = self.config.data.test_queries_path

        if not Path(test_queries_path).exists():
            print(f"⚠️  테스트 쿼리 파일이 없습니다: {test_queries_path}")
            print("샘플 테스트 쿼리를 생성합니다...")
            return self._create_sample_queries()

        queries = []
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                query_data = json.loads(line.strip())
                queries.append(query_data)

        print(f"테스트 쿼리 로드 완료: {len(queries)}개")
        return queries

    def _create_sample_queries(self) -> List[Dict[str, Any]]:
        """샘플 테스트 쿼리 생성 (Ground Truth가 없을 때)"""
        sample_queries = [
            {
                "query": "컴퓨터공학 전공 신입 개발자 채용공고",
                "ground_truth_docs": [],  # 실제로는 관련 문서 ID들이 들어가야 함
                "user_profile": {
                    "major": "컴퓨터공학과",
                    "interest_job": ["개발자", "프로그래머"]
                }
            },
            {
                "query": "데이터 사이언스 관련 직무",
                "ground_truth_docs": [],
                "user_profile": {
                    "major": "데이터사이언스학과",
                    "interest_job": ["데이터 분석가", "데이터 사이언티스트"]
                }
            }
        ]

        print(f"샘플 쿼리 생성: {len(sample_queries)}개")
        return sample_queries

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
        lines = query_text.split('\n')
        course_history_start = -1

        for i, line in enumerate(lines):
            if line.startswith('수강 이력:'):
                course_history_start = i
                break

        if course_history_start == -1:
            return query_text  # 수강 이력이 없으면 그대로 반환

        # 3. 기본 정보 부분과 수강 이력 부분 분리
        basic_info = lines[:course_history_start+1]  # "수강 이력:" 까지 포함
        course_lines = lines[course_history_start+1:]

        # 4. 각 강의 블록 파싱 (강의명으로 시작하는 블록들)
        courses = []
        current_course = []

        for line in course_lines:
            if line.startswith('강의명:'):
                if current_course:  # 이전 강의 저장
                    courses.append('\n'.join(current_course))
                current_course = [line]
            else:
                current_course.append(line)

        if current_course:  # 마지막 강의 저장
            courses.append('\n'.join(current_course))

        # 5. 뒤에서부터 과목을 하나씩 제거하면서 토큰 수 체크
        while courses and len(courses) > 5:  # 최소 5개는 유지
            # 현재 상태로 텍스트 재구성
            trimmed_text = '\n'.join(basic_info + ['\n'.join(courses)])

            if self.count_tokens(trimmed_text) <= max_tokens:
                return trimmed_text

            # 마지막 과목 제거
            courses.pop()

        # 6. 최종 텍스트 반환 (5개 이하가 되어도 토큰이 초과하면 그대로 반환)
        final_text = '\n'.join(basic_info + ['\n'.join(courses)])
        return final_text

    def _evaluate_retrieval(self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]) -> List[QueryResult]:
        """검색 성능 평가 수행"""
        embedder = components['embedder']
        retriever = components['retriever']
        evaluator = components['evaluator']

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

            # 필수 필드 체크
            if 'query' not in query_data:
                print(f"'query' 필드 없음, 쿼리 스킵")
                skipped_queries += 1
                continue

            query_text = query_data['query']
            ground_truth = query_data.get('ground_truth_docs', [])

            # 토큰 수 체크 및 필요시 수강 이력 트리밍
            original_token_count = self.count_tokens(query_text)
            if original_token_count > TOKEN_LIMIT:
                print(f"토큰 초과 감지 ({original_token_count}), 수강 이력 트리밍 시도...")
                query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)
                new_token_count = self.count_tokens(query_text)

                if new_token_count > TOKEN_LIMIT:
                    print(f"쿼리 스킵 (트리밍 후에도 토큰 초과: {new_token_count}): {query_text[:50]}...")
                    skipped_queries += 1
                    continue
                else:
                    print(f"트리밍 성공: {original_token_count} → {new_token_count} 토큰")

            try:
                # 쿼리 임베딩 생성
                query_embedding = embedder.embed([query_text])[0]

                # 검색 수행
                search_results = retriever.search(
                    query_embedding,
                    top_k=self.config.retriever.top_k
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
                                retrieved_docs.append({"text": doc.get("text", ""), "metadata": doc.get("metadata", {})})
                            else:
                                print(f"예상과 다른 doc 타입: {type(doc)}, 내용: {doc}")
                        else:
                            print(f"예상과 다른 item 구조: {type(item)}, 내용: {item}")

                    query_result = QueryResult(
                        query=query_text,
                        retrieved_docs=retrieved_docs,
                        ground_truth_docs=ground_truth
                    )
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

        # 평가 지표 계산
        evaluation_results = evaluator.evaluate(query_results)

        print("\n=== 평가 결과 ===")
        for result in evaluation_results:
            print(f"{result.metric_name}: {result.score:.4f}")

        return query_results

    def _save_results(self, query_results: List[QueryResult], components: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """실험 결과 저장"""
        # 평가 결과 계산
        evaluator = components['evaluator']
        evaluation_results = evaluator.evaluate(query_results)

        # 결과 딕셔너리 구성
        results = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "description": self.config.description,
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time
            },
            "config": {
                "embedder": self.config.embedder.__dict__,
                "chunker": self.config.chunker.__dict__,
                "retriever": self.config.retriever.__dict__,
                "evaluation": self.config.evaluation.__dict__
            },
            "component_info": {
                name: comp.get_model_info() if hasattr(comp, 'get_model_info')
                      else comp.get_chunker_info() if hasattr(comp, 'get_chunker_info')
                      else comp.get_retriever_info() if hasattr(comp, 'get_retriever_info')
                      else {}
                for name, comp in components.items() if hasattr(comp, '__dict__')
            },
            "evaluation_results": [
                {
                    "metric": result.metric_name,
                    "score": result.score,
                    "details": result.details
                }
                for result in evaluation_results
            ],
            "query_count": len(query_results),
            "document_count": components['retriever'].get_document_count()
        }

        # 결과 파일 저장
        results_file = self.output_dir / f"results_{self.experiment_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 상세 쿼리 결과 저장 (옵션)
        detailed_results_file = self.output_dir / f"detailed_results_{self.experiment_id}.jsonl"
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            for qr in query_results:
                query_detail = {
                    "query": qr.query,
                    "ground_truth_count": len(qr.ground_truth_docs),
                    "retrieved_count": len(qr.retrieved_docs),
                    "retrieved_doc_ids": [
                        doc.get('metadata', {}).get('rec_idx', 'unknown')
                        for doc in qr.retrieved_docs
                    ]
                }
                f.write(json.dumps(query_detail, ensure_ascii=False) + '\n')

        print(f"결과 저장 완료:")
        print(f"  - 요약 결과: {results_file}")
        print(f"  - 상세 결과: {detailed_results_file}")

        return results