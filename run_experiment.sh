#!/bin/bash

# Career-HY RAG 실험 실행 스크립트 (단순화 버전)

set -e

# 사용법 체크
if [ $# -lt 1 ]; then
    echo "사용법: $0 <config_file>"
    echo ""
    echo "예시:"
    echo "  $0 configs/baseline.yaml"
    echo "  $0 configs/chunking_test.yaml"
    echo "  $0 configs/embedding_3large.yaml"
    exit 1
fi

CONFIG_FILE=$1
shift  # 첫 번째 인자 제거

# 설정 파일 존재 여부 확인
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

# 환경 변수 파일 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다. .env 파일을 생성해주세요."
    exit 1
fi

echo "🚀 Career-HY RAG 실험 시작"
echo "설정 파일: $CONFIG_FILE"

echo "="*50

# 결과 디렉토리 생성
mkdir -p results cache

# 환경 변수 로드
source .env

# Docker 컨테이너에서 실험 실행
echo "🐳 Docker 컨테이너에서 실험 실행 중..."
docker compose run --rm experiment python run_experiment.py "$CONFIG_FILE"

echo ""
echo "✅ 실험 완료!"
echo "📁 결과 확인: results/ 디렉토리"
echo "💾 캐시 확인: python cache_manager.py list"