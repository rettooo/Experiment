"""
실험 실행 메인 스크립트
"""

import sys
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from core.config import ExperimentConfig
from core.pipeline import ExperimentPipeline

# .env 파일 로드 (프로젝트 루트에서)
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Career-HY RAG 실험 실행")
    parser.add_argument("config_path", help="실험 설정 YAML 파일 경로")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 설정 파일 확인
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    try:
        # 설정 로드
        print(f"설정 파일 로드: {config_path}")
        config = ExperimentConfig.from_yaml(str(config_path))

        # 파이프라인 실행
        pipeline = ExperimentPipeline(config)
        results = await pipeline.run()

        # 결과 요약 출력
        print("\n" + "=" * 50)
        print("실험 결과 요약")
        print("=" * 50)

        # retrieval_only 모드 결과
        if "summary" in results and "average_metrics" in results["summary"]:
            summary = results["summary"]
            print("=== 검색 성능 지표 (평균) ===")
            avg_metrics = summary["average_metrics"]
            print(f"NDCG@10:      {avg_metrics['ndcg@10']:.4f}")
            print(f"Recall@20:    {avg_metrics['recall@20']:.4f}")
            print(f"MRR@10:       {avg_metrics['mrr@10']:.4f}")
            print(f"Precision@3:  {avg_metrics['precision@3']:.4f}")
            print(f"Precision@5:  {avg_metrics['precision@5']:.4f}")

            if "total_search_time" in summary:
                print(f"\n총 검색 시간:   {summary['total_search_time']:.3f}초")
                print(
                    f"평균 검색 시간: {summary['average_search_time_per_query']:.3f}초/쿼리"
                )

            print(f"\n평가된 쿼리 수: {summary['total_queries']}개")

            if "search_results_file" in results:
                print(f"\n✅ 결과 저장:")
                print(f"   상세: {results['search_results_file']}")
                print(f"   요약: {results['summary_file']}")

        # dual 모드 결과 (기존)
        elif "retrieval_evaluation" in results:
            print("=== 검색 성능 지표 ===")
            for metric in results["retrieval_evaluation"]["metrics"]:
                print(f"{metric['metric']}: {metric['score']:.4f}")

            # 생성 평가 결과 (LangSmith 정성평가)
            if (
                "generation_evaluation" in results
                and "langsmith_metrics" in results["generation_evaluation"]
            ):
                print("\n=== LangSmith 정성평가 지표 ===")
                for metric in results["generation_evaluation"]["langsmith_metrics"]:
                    print(f"{metric['metric']}: {metric['score']:.4f}")

            print(f"\n처리된 문서 수: {results['document_count']}")
            print(f"평가된 쿼리 수: {results['retrieval_evaluation']['query_count']}")
            print(
                f"총 소요시간: {results['experiment_info']['duration_seconds']:.2f}초"
            )

    except Exception as e:
        print(f"실험 실행 중 오류 발생: {e}")
        # 항상 상세한 에러 정보 출력
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
