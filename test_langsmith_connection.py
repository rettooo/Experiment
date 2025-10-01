#!/usr/bin/env python3
"""
LangSmith 연결 테스트 스크립트

LangSmith 설정이 올바르게 되었는지 확인하는 간단한 테스트
"""

import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


def test_langsmith_connection():
    """LangSmith 연결 및 설정 테스트"""

    print("🔍 LangSmith 설정 확인 중...")
    print("-" * 50)

    # 1. 환경변수 확인
    required_vars = [
        'LANGCHAIN_TRACING_V2',
        'LANGCHAIN_API_KEY',
        'LANGCHAIN_PROJECT',
        'LANGCHAIN_ENDPOINT'
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == 'LANGCHAIN_API_KEY':
                print(f"✅ {var}: {value[:8]}...")
            else:
                print(f"✅ {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}: 설정되지 않음")

    if missing_vars:
        print(f"\n⚠️  누락된 환경변수: {missing_vars}")
        print("LANGSMITH_SETUP.md를 참고하여 설정을 완료해주세요.")
        return False

    # 2. LangSmith 패키지 확인
    print("\n📦 LangSmith 패키지 확인 중...")
    try:
        import langsmith
        print(f"✅ langsmith 버전: {langsmith.__version__}")
    except ImportError:
        print("❌ langsmith 패키지가 설치되지 않았습니다.")
        print("설치 명령: pip install langsmith")
        return False

    try:
        import langchain
        print(f"✅ langchain 버전: {langchain.__version__}")
    except ImportError:
        print("❌ langchain 패키지가 설치되지 않았습니다.")
        print("설치 명령: pip install langchain")
        return False

    # 3. LangSmith 클라이언트 연결 테스트
    print("\n🌐 LangSmith 서버 연결 테스트...")
    try:
        from langsmith import Client

        client = Client()

        # 프로젝트 목록 조회 테스트
        projects = list(client.list_projects(limit=5))
        print(f"✅ LangSmith 서버 연결 성공!")
        print(f"📁 프로젝트 수: {len(projects)}")

        # 현재 프로젝트 확인
        current_project = os.getenv('LANGCHAIN_PROJECT')
        print(f"🎯 현재 프로젝트: {current_project}")

        # 프로젝트 존재 여부 확인
        project_exists = any(p.name == current_project for p in projects)
        if project_exists:
            print(f"✅ 프로젝트 '{current_project}' 존재 확인")
        else:
            print(f"⚠️  프로젝트 '{current_project}'가 존재하지 않습니다.")
            print("LangSmith 웹사이트에서 프로젝트를 생성해주세요.")

    except Exception as e:
        print(f"❌ LangSmith 연결 실패: {e}")
        print("API 키와 엔드포인트를 확인해주세요.")
        return False

    # 4. 간단한 추적 테스트
    print("\n🔄 추적 기능 테스트...")
    try:
        from langchain_openai import ChatOpenAI
        from langsmith import traceable

        # OpenAI API 키 확인
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
            print("추적 테스트를 건너뜁니다.")
        else:
            @traceable(name="langsmith_test")
            def simple_llm_test():
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                response = llm.invoke("Hello, this is a LangSmith connection test.")
                return response.content

            # 테스트 실행
            test_response = simple_llm_test()
            print(f"✅ 추적 테스트 성공!")
            print(f"📝 응답: {test_response[:50]}...")
            print(f"🔗 LangSmith 프로젝트에서 추적 로그를 확인하세요.")

    except Exception as e:
        print(f"⚠️  추적 테스트 실패: {e}")
        print("기본 연결은 성공했지만 추적에 문제가 있을 수 있습니다.")

    print("\n" + "="*50)
    print("🎉 LangSmith 설정 테스트 완료!")
    print("실험 파이프라인에서 LangSmith를 사용할 준비가 되었습니다.")
    return True


def print_next_steps():
    """다음 단계 안내"""
    print("\n📋 다음 단계:")
    print("1. LangSmith 웹사이트에 로그인하여 프로젝트를 확인하세요.")
    print("2. 추적 로그가 정상적으로 기록되는지 확인하세요.")
    print("3. 실험 설정에서 LangSmith 평가를 활성화하세요.")
    print("\n🌐 LangSmith 대시보드:")
    print("   https://smith.langchain.com/")


if __name__ == "__main__":
    success = test_langsmith_connection()

    if success:
        print_next_steps()
    else:
        print("\n❌ 설정을 완료한 후 다시 테스트해주세요.")
        print("📖 자세한 설정 방법: LANGSMITH_SETUP.md 참고")