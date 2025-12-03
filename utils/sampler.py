"""
고유 프로필 기반 쿼리 샘플링

전체 test_queries 중에서 고유한 프로필(해시 기반)이 겹치지 않도록
서로 다른 사용자의 질문을 각각 1개씩 샘플링하는 로직
"""

import json
import random
import hashlib
from typing import List, Dict, Any, Optional
from collections import defaultdict


class StratifiedSampler:
    """고유 프로필 해시 기반 샘플링을 통한 대표성 있는 쿼리 선택"""

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 재현 가능한 샘플링을 위한 시드값
        """
        self.seed = seed
        # 전역 random 대신 독립적인 Random 인스턴스 사용 (재현성 보장)
        self.rng = random.Random(seed) if seed is not None else random

    def _generate_profile_hash(self, user_profile: Dict[str, Any]) -> str:
        """
        프로필 전체 내용을 기반으로 고유 해시 생성

        Args:
            user_profile: 사용자 프로필 딕셔너리

        Returns:
            8자리 해시 문자열 (고유 프로필 식별자)
        """
        # 프로필의 모든 정보를 정렬된 JSON 문자열로 변환
        profile_str = json.dumps(user_profile, sort_keys=True, ensure_ascii=False)

        # MD5 해시 생성 후 첫 8자리 사용
        hash_object = hashlib.md5(profile_str.encode('utf-8'))
        return hash_object.hexdigest()[:8]

    def sample_queries(
        self,
        all_queries: List[Dict[str, Any]],
        sample_size: int = 15,
        strategy: str = "profile_based"
    ) -> List[Dict[str, Any]]:
        """
        쿼리 샘플링 수행

        Args:
            all_queries: 전체 쿼리 목록
            sample_size: 샘플 크기 (고유 프로필 수)
            strategy: 샘플링 전략 ("profile_based", "random")

        Returns:
            샘플링된 쿼리 목록 (각 고유 프로필에서 1개씩)
        """

        print(f"🎯 샘플링 시작: {len(all_queries)}개 중 {sample_size}개 선택 (전략: {strategy})")

        if len(all_queries) <= sample_size:
            print(f"⚠️  전체 쿼리가 샘플 크기보다 작습니다. 전체 쿼리 사용.")
            return all_queries

        if strategy == "profile_based":
            return self._profile_based_sampling(all_queries, sample_size)
        elif strategy == "random":
            return self._random_sampling(all_queries, sample_size)
        else:
            raise ValueError(f"지원하지 않는 샘플링 전략: {strategy}. 지원되는 전략: profile_based, random")

    def _profile_based_sampling(self, queries: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        프로필 기반 샘플링: 고유한 프로필이 겹치지 않게 선택
        각 고유 프로필에서 1개 쿼리만 선택하여 서로 다른 사용자의 질문을 평가
        """

        # 1. 프로필 해시를 기반으로 고유 프로필별 쿼리 그룹화
        profile_groups = defaultdict(list)
        for query in queries:
            user_profile = query.get('user_profile', {})
            profile_hash = self._generate_profile_hash(user_profile)
            profile_groups[profile_hash].append(query)

        print(f"📊 고유 프로필별 쿼리 분포:")
        profile_info = {}
        for profile_hash, profile_queries in profile_groups.items():
            # 각 프로필의 전공 정보 표시 (해시 대신)
            first_query = profile_queries[0]
            major = first_query.get('user_profile', {}).get('major', 'unknown')
            profile_info[profile_hash] = {
                'major': major,
                'query_count': len(profile_queries)
            }
            print(f"  {profile_hash} ({major}): {len(profile_queries)}개")

        # 2. 사용 가능한 고유 프로필 수 확인
        available_profiles = len(profile_groups)
        print(f"\n🔍 사용 가능한 고유 프로필 수: {available_profiles}개")

        if available_profiles < sample_size:
            print(f"⚠️  고유 프로필 수({available_profiles})가 샘플 크기({sample_size})보다 적습니다.")
            print(f"⚠️  {available_profiles}개 프로필에서 각각 1개씩 선택합니다.")
            target_profiles = available_profiles
        else:
            target_profiles = sample_size

        print(f"🎯 목표: {target_profiles}개 고유 프로필에서 각각 1개 쿼리씩 선택")

        # 3. 랜덤하게 프로필 선택
        profile_hashes = list(profile_groups.keys())
        selected_profile_hashes = self.rng.sample(profile_hashes, target_profiles)

        print(f"✅ 선택된 프로필:")
        for hash_key in selected_profile_hashes:
            major = profile_info[hash_key]['major']
            print(f"  {hash_key} ({major})")

        # 4. 각 선택된 프로필에서 1개 쿼리만 랜덤 선택
        selected_queries = []
        for profile_hash in selected_profile_hashes:
            profile_queries = profile_groups[profile_hash]
            selected_query = self.rng.sample(profile_queries, 1)[0]
            selected_queries.append(selected_query)

            major = profile_info[profile_hash]['major']
            print(f"  {profile_hash} ({major}): {len(profile_queries)}개 중 1개 선택")

        print(f"🎉 최종 샘플링 결과: {len(selected_queries)}개 쿼리 선택 (서로 다른 {len(selected_queries)}명의 사용자)")

        # 5. 선택된 쿼리의 전공 분포 확인 (참고용)
        final_major_dist = defaultdict(int)
        for query in selected_queries:
            major = query.get('user_profile', {}).get('major', 'unknown')
            final_major_dist[major] += 1

        print(f"📊 최종 전공 분포 (참고용):")
        for major, count in final_major_dist.items():
            print(f"  {major}: {count}개")

        return selected_queries

    def _random_sampling(self, queries: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """단순 랜덤 샘플링"""
        return self.rng.sample(queries, sample_size)


def generate_reproducible_seed(config_dict: Dict[str, Any]) -> int:
    """실험 설정을 기반으로 재현 가능한 시드 생성"""

    # 설정을 JSON 문자열로 변환 후 해시 생성
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_object = hashlib.md5(config_str.encode())

    # 해시의 첫 8자리를 정수로 변환
    seed = int(hash_object.hexdigest()[:8], 16)
    return seed % 100000  # 0-99999 범위로 제한


def analyze_sample_distribution(
    original_queries: List[Dict[str, Any]],
    sampled_queries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """샘플링 결과의 분포 분석 (고유 프로필 기반)"""

    def get_profile_distribution(queries):
        # 임시 샘플러 인스턴스로 해시 생성 함수 사용
        temp_sampler = StratifiedSampler()

        profile_hashes = []
        majors = []
        gt_counts = []

        for query in queries:
            user_profile = query.get('user_profile', {})
            profile_hash = temp_sampler._generate_profile_hash(user_profile)
            major = user_profile.get('major', 'unknown')
            gt_count = len(query.get('ground_truth_docs', []))

            profile_hashes.append(profile_hash)
            majors.append(major)
            gt_counts.append(gt_count)

        return {
            'unique_profiles': len(set(profile_hashes)),
            'major_distribution': {major: majors.count(major) for major in set(majors)},
            'gt_count_stats': {
                'mean': sum(gt_counts) / len(gt_counts) if gt_counts else 0,
                'min': min(gt_counts) if gt_counts else 0,
                'max': max(gt_counts) if gt_counts else 0
            }
        }

    original_stats = get_profile_distribution(original_queries)
    sampled_stats = get_profile_distribution(sampled_queries)

    return {
        'original_count': len(original_queries),
        'sampled_count': len(sampled_queries),
        'sampling_ratio': len(sampled_queries) / len(original_queries),
        'original_distribution': original_stats,
        'sampled_distribution': sampled_stats
    }