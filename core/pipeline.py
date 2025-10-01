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
from utils.sampler import StratifiedSampler, generate_reproducible_seed, analyze_sample_distribution
from implementations.evaluators import SearchMetricsEvaluator
from implementations.evaluators.langsmith_evaluator import CareerHYLangSmithEvaluator


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

            # 5. Ground Truth 쿼리 로드
            print("\n=== 5. Ground Truth 쿼리 로드 ===")
            test_queries = self._load_test_queries()

            # 6. 이중 평가 실행
            print("\n=== 6. 이중 평가 시스템 ===")
            dual_results = await self._run_dual_evaluation(test_queries, components)

            # 7. 결과 저장
            print("\n=== 7. 결과 저장 ===")
            results = await self._save_dual_results(dual_results, components, start_time)

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

        # 응답 생성기 초기화 (선택적)
        if hasattr(self.config, 'response_generator') and self.config.response_generator:
            print(f"응답 생성기 초기화: {self.config.response_generator.type}")
            components['response_generator'] = ComponentFactory.create_response_generator(self.config.response_generator)

        # 평가기 초기화
        components['evaluator'] = SearchMetricsEvaluator(
            k_values=self.config.evaluation.retrieval.k_values
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

    async def _generate_response_for_query(
        self,
        query_data: Dict[str, Any],
        query_text: str,
        retrieved_docs: List[Dict[str, Any]],
        response_generator
    ) -> Optional[Dict[str, Any]]:
        """개별 쿼리에 대한 응답 생성"""
        try:
            # 사용자 프로필 추출
            user_profile = query_data.get('user_profile', {})

            # 대화 이력 추출 (있다면)
            chat_history = query_data.get('chat_history', [])

            # 응답 생성
            generated_response = await response_generator.generate(
                query=query_text,
                retrieved_docs=retrieved_docs,
                user_profile=user_profile,
                chat_history=chat_history
            )

            return {
                "content": generated_response.content,
                "recommended_jobs": [job.dict() for job in generated_response.recommended_jobs]
            }

        except Exception as e:
            print(f"응답 생성 실패: {e}")
            return None

    async def _evaluate_retrieval(self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]) -> List[QueryResult]:
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

                    # 응답 생성 (선택적)
                    generated_response = None
                    if 'response_generator' in components:
                        generated_response = await self._generate_response_for_query(
                            query_data, query_text, retrieved_docs, components['response_generator']
                        )

                    query_result = QueryResult(
                        query=query_text,
                        retrieved_docs=retrieved_docs,
                        ground_truth_docs=ground_truth
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

        # 평가 지표 계산
        evaluation_results = evaluator.evaluate(query_results)

        print("\n=== 평가 결과 ===")
        for result in evaluation_results:
            print(f"{result.metric_name}: {result.score:.4f}")

        return query_results

    async def _run_dual_evaluation(self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]) -> Dict[str, Any]:
        """이중 평가 시스템: 전체 검색 평가 + 샘플 생성 평가"""

        dual_results = {}

        # 1. 전체 쿼리 검색 성능 평가
        print("🔍 1단계: 전체 쿼리 검색 성능 평가")
        print(f"   대상: {len(test_queries)}개 쿼리")

        retrieval_results = await self._evaluate_retrieval_only(test_queries, components)
        dual_results['retrieval_evaluation'] = {
            'query_results': retrieval_results,
            'query_count': len(retrieval_results)
        }

        # 2. 샘플 쿼리 선택 (응답 생성기가 있는 경우만)
        if 'response_generator' in components:
            print("\n🎯 2단계: 샘플 쿼리 선택")

            # 샘플링 설정 확인
            evaluation_config = getattr(self.config, 'evaluation')
            generation_config = evaluation_config.generation

            # 기본값 설정
            sample_size = generation_config.sample_size
            sample_strategy = generation_config.sample_strategy
            sample_seed = generation_config.sample_seed

            # 시드 생성 (설정되지 않은 경우)
            if sample_seed is None:
                config_dict = {
                    'embedder': self.config.embedder.__dict__,
                    'chunker': self.config.chunker.__dict__,
                    'retriever': self.config.retriever.__dict__
                }
                sample_seed = generate_reproducible_seed(config_dict)

            print(f"   샘플 크기: {sample_size}")
            print(f"   샘플링 전략: {sample_strategy}")
            print(f"   시드: {sample_seed}")

            # 샘플링 수행
            sampler = StratifiedSampler(seed=sample_seed)
            sampled_queries = sampler.sample_queries(
                test_queries,
                sample_size=sample_size,
                strategy=sample_strategy
            )

            # 샘플링 분포 분석
            distribution_analysis = analyze_sample_distribution(test_queries, sampled_queries)
            print(f"   샘플링 비율: {distribution_analysis['sampling_ratio']:.2%}")

            # 3. 샘플 쿼리 검색 + 응답 생성 평가
            print("\n🤖 3단계: 샘플 쿼리 응답 생성 평가")
            print(f"   대상: {len(sampled_queries)}개 쿼리")

            generation_results = await self._evaluate_generation_for_samples(sampled_queries, components)

            dual_results['generation_evaluation'] = {
                'sampled_queries': sampled_queries,
                'query_results': generation_results,
                'sample_config': {
                    'sample_size': len(sampled_queries),
                    'sample_strategy': sample_strategy,
                    'sample_seed': sample_seed
                },
                'distribution_analysis': distribution_analysis
            }

        else:
            print("\n⚠️  응답 생성기가 설정되지 않아 생성 평가를 건너뜁니다.")
            dual_results['generation_evaluation'] = None

        return dual_results

    async def _evaluate_retrieval_only(self, test_queries: List[Dict[str, Any]], components: Dict[str, Any]) -> List[QueryResult]:
        """검색 성능만 평가 (응답 생성 없음)"""

        embedder = components['embedder']
        retriever = components['retriever']

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

                if not isinstance(query_data, dict) or 'query' not in query_data:
                    skipped_queries += 1
                    continue

                query_text = query_data['query']
                ground_truth = query_data.get('ground_truth_docs', [])

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
                search_results = retriever.search(query_embedding, top_k=self.config.retriever.top_k)

                # QueryResult 생성
                retrieved_docs = []
                for item in search_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        if isinstance(doc, dict):
                            retrieved_docs.append({"text": doc.get("text", ""), "metadata": doc.get("metadata", {})})

                query_result = QueryResult(
                    query=query_text,
                    retrieved_docs=retrieved_docs,
                    ground_truth_docs=ground_truth
                )

                query_results.append(query_result)

                if (i + 1) % 50 == 0:
                    print(f"   검색 평가 진행률: {len(query_results)}/{len(test_queries)}")

            except Exception as e:
                print(f"   쿼리 {i} 검색 평가 실패: {e}")
                skipped_queries += 1
                continue

        print(f"   검색 평가 완료: {len(query_results)}개 성공, {skipped_queries}개 스킵")
        return query_results

    async def _evaluate_generation_for_samples(
        self,
        sampled_queries: List[Dict[str, Any]],
        components: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """샘플 쿼리들에 대한 검색 + 응답 생성 평가"""

        embedder = components['embedder']
        retriever = components['retriever']
        response_generator = components['response_generator']

        generation_results = []
        TOKEN_LIMIT = 8000

        for i, query_data in enumerate(sampled_queries):
            try:
                query_text = query_data['query']
                ground_truth = query_data.get('ground_truth_docs', [])

                # 토큰 수 체크 및 트리밍
                original_token_count = self.count_tokens(query_text)
                if original_token_count > TOKEN_LIMIT:
                    query_text = self.trim_courses_if_needed(query_text, TOKEN_LIMIT)

                # 검색 수행
                query_embedding = embedder.embed([query_text])[0]
                search_results = retriever.search(query_embedding, top_k=self.config.retriever.top_k)

                # 검색 결과 정리
                retrieved_docs = []
                for item in search_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        if isinstance(doc, dict):
                            retrieved_docs.append({"text": doc.get("text", ""), "metadata": doc.get("metadata", {})})

                # 응답 생성
                generated_response = await self._generate_response_for_query(
                    query_data, query_text, retrieved_docs, response_generator
                )

                # 결과 저장
                result = {
                    'query': query_text,
                    'user_profile': query_data.get('user_profile', {}),
                    'ground_truth_docs': ground_truth,
                    'retrieved_docs': retrieved_docs,
                    'generated_response': generated_response
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
        langsmith_evaluation_results: List
    ):
        """LangSmith 평가 실행 (설정된 경우만)"""

        # LangSmith 설정 확인
        langsmith_config = getattr(self.config, 'langsmith', None)
        print(f"🔍 LangSmith 설정 확인: {langsmith_config}")

        if not langsmith_config or not langsmith_config.enabled:
            print("\n⚠️  LangSmith 평가가 비활성화되어 있어 건너뜁니다.")
            return

        # 환경변수 확인
        import os
        api_key = os.getenv('LANGCHAIN_API_KEY')
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
                judge_model=judge_model,
                project_name=project_name
            )
            print("✅ LangSmith 평가기 초기화 완료")

            # 평가 실행
            print("🚀 LangSmith 평가 시작...")
            evaluation_results = await langsmith_evaluator.evaluate_batch(
                generation_query_results,
                experiment_name=self.config.experiment_name
            )
            print(f"✅ LangSmith 평가 완료: {len(evaluation_results)}개 결과")

            langsmith_evaluation_results.extend(evaluation_results)

        except Exception as e:
            print(f"❌ LangSmith 평가 실패: {e}")
            print("자동화된 평가 결과만 사용합니다.")

    async def _save_dual_results(self, dual_results: Dict[str, Any], components: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """이중 평가 결과 저장"""

        # 1. 검색 성능 평가
        retrieval_evaluation = dual_results['retrieval_evaluation']
        retrieval_query_results = retrieval_evaluation['query_results']

        evaluator = components['evaluator']
        retrieval_evaluation_results = evaluator.evaluate(retrieval_query_results)

        print("\n=== 검색 성능 평가 결과 ===")
        for result in retrieval_evaluation_results:
            print(f"{result.metric_name}: {result.score:.4f}")

        # 2. 생성 품질 평가 (응답 생성기가 있는 경우)
        langsmith_evaluation_results = []

        if dual_results['generation_evaluation'] is not None:
            generation_evaluation = dual_results['generation_evaluation']
            generation_query_results = generation_evaluation['query_results']

            if generation_query_results:
                print("\n=== LangSmith 정성평가 실행 ===")
                # LangSmith 평가만 실행
                await self._run_langsmith_evaluation_if_enabled(
                    generation_query_results,
                    langsmith_evaluation_results
                )

        # 3. 결과 딕셔너리 구성
        results = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "description": self.config.description,
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "evaluation_type": "dual"  # 이중 평가 표시
            },
            "config": {
                "embedder": asdict(self.config.embedder),
                "chunker": asdict(self.config.chunker),
                "retriever": asdict(self.config.retriever),
                "evaluation": {
                    "retrieval": asdict(self.config.evaluation.retrieval),
                    "generation": asdict(self.config.evaluation.generation)
                },
                "langsmith": asdict(self.config.langsmith) if self.config.langsmith else None
            },
            "component_info": {
                name: comp.get_model_info() if hasattr(comp, 'get_model_info')
                      else comp.get_chunker_info() if hasattr(comp, 'get_chunker_info')
                      else comp.get_retriever_info() if hasattr(comp, 'get_retriever_info')
                      else {}
                for name, comp in components.items() if hasattr(comp, '__dict__')
            },
            "retrieval_evaluation": {
                "query_count": len(retrieval_query_results),
                "metrics": [
                    {
                        "metric": result.metric_name,
                        "score": result.score,
                        "details": result.details
                    }
                    for result in retrieval_evaluation_results
                ]
            },
            "document_count": components['retriever'].get_document_count()
        }

        # 생성 평가 결과 추가
        if dual_results['generation_evaluation'] is not None:
            generation_evaluation = dual_results['generation_evaluation']

            results["generation_evaluation"] = {
                "sample_count": len(generation_evaluation['query_results']),
                "sample_config": generation_evaluation['sample_config'],
                "distribution_analysis": generation_evaluation['distribution_analysis'],
                "langsmith_metrics": [
                    {
                        "metric": result.metric_name,
                        "score": result.score,
                        "reasoning": result.reasoning,
                        "details": result.details
                    }
                    for result in langsmith_evaluation_results
                ] if langsmith_evaluation_results else []
            }

            # 응답 생성기 설정 추가
            if hasattr(self.config, 'response_generator'):
                results["config"]["response_generator"] = asdict(self.config.response_generator)

            # LangSmith 설정 추가 (이미 위에서 처리되었음)
            # if hasattr(self.config, 'langsmith'):
            #     results["config"]["langsmith"] = asdict(self.config.langsmith)

        # 4. 결과 파일 저장
        results_file = self.output_dir / f"results_{self.experiment_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 5. 상세 결과 저장
        # 검색 결과
        retrieval_detailed_file = self.output_dir / f"retrieval_detailed_{self.experiment_id}.jsonl"
        with open(retrieval_detailed_file, 'w', encoding='utf-8') as f:
            for qr in retrieval_query_results:
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

        # 생성 결과 (있다면)
        if dual_results['generation_evaluation'] is not None:
            generation_evaluation = dual_results['generation_evaluation']
            generation_results = generation_evaluation['query_results']

            if generation_results:
                generation_detailed_file = self.output_dir / f"generation_detailed_{self.experiment_id}.jsonl"
                with open(generation_detailed_file, 'w', encoding='utf-8') as f:
                    for result in generation_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                # 생성된 응답만 별도 저장
                responses_file = self.output_dir / f"generated_responses_{self.experiment_id}.json"
                responses_only = []
                for result in generation_results:
                    if result.get('generated_response'):
                        responses_only.append({
                            "query": result['query'],
                            "response": result['generated_response']
                        })

                with open(responses_file, 'w', encoding='utf-8') as f:
                    json.dump(responses_only, f, ensure_ascii=False, indent=2)

                print(f"  - 생성 상세 결과: {generation_detailed_file}")
                print(f"  - 생성된 응답: {responses_file}")

        print(f"결과 저장 완료:")
        print(f"  - 요약 결과: {results_file}")
        print(f"  - 검색 상세 결과: {retrieval_detailed_file}")

        return results