"""
실험 실행 메인 스크립트
"""

import sys
import argparse
from pathlib import Path

from core.config import ExperimentConfig
from core.pipeline import ExperimentPipeline


def main():
    parser = argparse.ArgumentParser(description='Career-HY RAG 실험 실행')
    parser.add_argument(
        'config_path',
        help='실험 설정 YAML 파일 경로'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 로그 출력'
    )

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
        results = pipeline.run()

        # 결과 요약 출력
        print("\n" + "="*50)
        print("실험 결과 요약")
        print("="*50)

        for eval_result in results['evaluation_results']:
            print(f"{eval_result['metric']}: {eval_result['score']:.4f}")

        print(f"\n처리된 문서 수: {results['document_count']}")
        print(f"평가된 쿼리 수: {results['query_count']}")
        print(f"총 소요시간: {results['experiment_info']['duration_seconds']:.2f}초")

    except Exception as e:
        print(f"실험 실행 중 오류 발생: {e}")
        # 항상 상세한 에러 정보 출력
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()