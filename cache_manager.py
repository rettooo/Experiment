"""
임베딩 캐시 관리 스크립트
"""

import sys
import argparse
from pathlib import Path

# 상위 디렉토리를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from utils.embedding_cache import embedding_cache
from utils.document_cache import document_cache
from utils.env_loader import load_env


def main():
    parser = argparse.ArgumentParser(description="임베딩 캐시 관리")
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")

    # list 명령어
    list_parser = subparsers.add_parser("list", help="캐시 목록 조회")

    # stats 명령어
    stats_parser = subparsers.add_parser("stats", help="캐시 통계 출력")

    # delete 명령어
    delete_parser = subparsers.add_parser("delete", help="캐시 삭제")
    delete_parser.add_argument("cache_key", help="삭제할 캐시 키")

    # info 명령어
    info_parser = subparsers.add_parser("info", help="특정 캐시 정보 조회")
    info_parser.add_argument("cache_key", help="조회할 캐시 키")

    # clear 명령어
    clear_parser = subparsers.add_parser("clear", help="모든 캐시 삭제")
    clear_parser.add_argument("--confirm", action="store_true", help="삭제 확인")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 환경 변수 로드 (캐시 조회에는 불필요하지만 일관성을 위해)
    load_env()

    if args.command == "list":
        list_caches()
    elif args.command == "stats":
        embedding_cache.print_cache_stats()
    elif args.command == "delete":
        delete_cache(args.cache_key)
    elif args.command == "info":
        show_cache_info(args.cache_key)
    elif args.command == "clear":
        clear_all_caches(args.confirm)


def list_caches():
    """캐시 목록 출력"""
    # 임베딩 캐시
    embedding_caches = embedding_cache.list_caches()

    # 문서 캐시
    document_caches = document_cache.list_caches()

    if not embedding_caches and not document_caches:
        print("캐시된 데이터가 없습니다.")
        return

    # 임베딩 캐시
    if embedding_caches:
        print("\n📦 임베딩 캐시 목록")
        print("=" * 80)
        print(
            f"{'캐시 키':<40} {'문서 수':<10} {'차원':<10} {'크기(MB)':<10} {'생성일'}"
        )
        print("-" * 80)

        for cache_info in embedding_caches:
            cache_key = cache_info["cache_key"]
            metadata = cache_info["metadata"]
            size_mb = embedding_cache.get_cache_size(cache_key) / 1024 / 1024

            print(
                f"{cache_key:<40} "
                f"{metadata['document_count']:<10} "
                f"{metadata['embedding_dimension']:<10} "
                f"{size_mb:<10.2f} "
                f"{metadata['created_at'][:19]}"
            )

    # 문서 캐시
    if document_caches:
        print("\n📄 문서 캐시 목록")
        print("=" * 80)
        print(f"{'캐시 키':<40} {'문서 수':<10} {'S3 Bucket':<20}")
        print("-" * 80)

        for cache_info in document_caches:
            cache_key = cache_info["cache_key"]
            doc_count = cache_info["document_count"]
            s3_bucket = cache_info["s3_config"]["s3_bucket"]

            print(f"{cache_key:<40} " f"{doc_count:<10} " f"{s3_bucket:<20}")


def delete_cache(cache_key: str):
    """특정 캐시 삭제"""
    if not embedding_cache.exists(cache_key):
        print(f"❌ 캐시가 존재하지 않습니다: {cache_key}")
        return

    # 삭제 확인
    response = input(f"정말로 '{cache_key}' 캐시를 삭제하시겠습니까? (y/N): ")
    if response.lower() != "y":
        print("삭제가 취소되었습니다.")
        return

    if embedding_cache.delete_cache(cache_key):
        print(f"✅ 캐시 삭제 완료: {cache_key}")
    else:
        print(f"❌ 캐시 삭제 실패: {cache_key}")


def show_cache_info(cache_key: str):
    """특정 캐시 상세 정보 출력"""
    if not embedding_cache.exists(cache_key):
        print(f"❌ 캐시가 존재하지 않습니다: {cache_key}")
        return

    metadata = embedding_cache.get_metadata(cache_key)
    size_mb = embedding_cache.get_cache_size(cache_key) / 1024 / 1024

    print(f"\n📦 캐시 정보: {cache_key}")
    print("=" * 60)
    print(f"문서 수: {metadata['document_count']}")
    print(f"임베딩 차원: {metadata['embedding_dimension']}")
    print(f"크기: {size_mb:.2f} MB")
    print(f"생성일: {metadata['created_at']}")
    print(f"저장 경로: {embedding_cache.get_cache_path(cache_key)}")

    if "embedder_config" in metadata.get("additional_info", {}):
        print(f"\n임베딩 설정:")
        embedder_config = metadata["additional_info"]["embedder_config"]
        for key, value in embedder_config.items():
            print(f"  {key}: {value}")

    if "chunker_config" in metadata.get("additional_info", {}):
        print(f"\n청킹 설정:")
        chunker_config = metadata["additional_info"]["chunker_config"]
        for key, value in chunker_config.items():
            print(f"  {key}: {value}")


def clear_all_caches(confirm: bool = False):
    """모든 캐시 삭제"""
    caches = embedding_cache.list_caches()

    if not caches:
        print("삭제할 캐시가 없습니다.")
        return

    if not confirm:
        print(f"⚠️  {len(caches)}개의 캐시가 삭제됩니다:")
        for cache_info in caches:
            print(f"  - {cache_info['cache_key']}")

        response = input("\n정말로 모든 캐시를 삭제하시겠습니까? (y/N): ")
        if response.lower() != "y":
            print("삭제가 취소되었습니다.")
            return

    deleted_count = 0
    for cache_info in caches:
        cache_key = cache_info["cache_key"]
        if embedding_cache.delete_cache(cache_key):
            deleted_count += 1

    print(f"✅ {deleted_count}개 캐시 삭제 완료")


if __name__ == "__main__":
    main()
