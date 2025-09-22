"""
환경 변수 로드 유틸리티
"""

import os
from pathlib import Path


def load_env(env_path: str = ".env"):
    """
    .env 파일에서 환경 변수 로드
    """
    env_file = Path(env_path)
    if not env_file.exists():
        print(f"⚠️  환경 변수 파일을 찾을 수 없습니다: {env_path}")
        return

    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 주석이나 빈 줄 스킵
            if not line or line.startswith('#'):
                continue

            # KEY=VALUE 형태 파싱
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # 따옴표 제거
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                os.environ[key] = value

    print(f"환경 변수 로드 완료: {env_path}")


def check_required_env_vars():
    """
    필수 환경 변수들이 설정되어 있는지 확인
    """
    required_vars = [
        'OPENAI_API_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_DEFAULT_REGION',
        'S3_BUCKET_NAME'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ 다음 환경 변수들이 설정되지 않았습니다: {missing_vars}")
        return False

    print("✅ 모든 필수 환경 변수가 설정되었습니다")
    return True