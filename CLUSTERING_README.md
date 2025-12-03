# 채용공고 클러스터링 파이프라인

## 📋 개요

태그 + 제목 기반으로 1,473개 채용공고를 직무별로 클러스터링하는 파이프라인

## 🔧 설치

```bash
# 필요 라이브러리 설치
pip install -r requirements_clustering.txt

# 또는 개별 설치
pip install sentence-transformers umap-learn hdbscan scikit-learn numpy pandas matplotlib seaborn tqdm
```

## 🚀 실행 방법

### 1. 기본 실행

```bash
cd Experiment
python job_clustering_pipeline.py
```

### 2. 예상 소요 시간

- **데이터 로드**: ~5초
- **임베딩 생성**: ~2-3분 (첫 실행 시 모델 다운로드: +30초)
- **UMAP 차원 축소**: ~30초
- **HDBSCAN 클러스터링**: ~10초
- **TF-IDF 정제**: ~20초
- **시각화 및 저장**: ~10초

**총 예상 시간: 약 3-4분**

## 📊 출력 결과

실행 후 `clustering_results/` 디렉토리에 다음 파일들이 생성됩니다:

```
clustering_results/
├── cluster_results.json           # 전체 결과 (클러스터, 문서, 메타데이터)
├── cluster_keywords.csv           # 클러스터별 키워드 (Excel로 열기)
├── cluster_documents.json         # 클러스터별 문서 목록
├── cluster_statistics.txt         # 통계 요약
├── cluster_visualization.png      # 2D UMAP 산점도
└── cluster_size_distribution.png  # 클러스터 크기 분포
```

## 🎛️ 파라미터

### UMAP (차원 축소)
```python
n_components = 15      # 축소 차원 (5-20 범위)
n_neighbors = 30       # 이웃 수 (지역 구조)
min_dist = 0.0         # 최소 거리 (밀집도)
```

### HDBSCAN (클러스터링)
```python
min_cluster_size = 10  # 클러스터당 최소 문서 수
min_samples = 3        # 코어 포인트 판정 기준
```

### TF-IDF (정제)
```python
top_n_keywords = 10            # 추출 키워드 수
min_docs_per_cluster = 5       # 최소 문서 수
```

## 📝 스크립트 점검 체크리스트

실행 전에 다음 사항들을 확인하세요:

### ✅ 환경 확인
- [ ] Python 3.8 이상 설치 확인
- [ ] 필요 라이브러리 설치 완료
- [ ] 충분한 메모리 (최소 4GB RAM)

### ✅ 파일 확인
- [ ] `structured_chunks/all_chunks_20251120_141253.json` 파일 존재
- [ ] 파일 크기 확인 (~55MB, 2,503개 청크)

### ✅ 실행 환경
- [ ] `Experiment/` 디렉토리에서 실행
- [ ] 쓰기 권한 확인 (clustering_results/ 생성)

### ✅ 예상 결과
- [ ] 약 10-20개 클러스터 생성 예상
- [ ] 1,200-1,400개 문서 클러스터링 예상 (노이즈 제외)
- [ ] 지역 태그 필터링: 약 200개 태그 제거

## 🐛 문제 해결

### 1. 모델 다운로드 실패
```bash
# 수동으로 모델 다운로드
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"
```

### 2. 메모리 부족
```python
# job_clustering_pipeline.py에서 batch_size 줄이기
embeddings = model.encode(texts, batch_size=16)  # 기본 32 → 16
```

### 3. 클러스터 수가 너무 적거나 많을 때

**너무 적을 때** (5개 미만):
```python
# min_cluster_size 줄이기
min_cluster_size = 5  # 기본 10 → 5
```

**너무 많을 때** (50개 이상):
```python
# min_cluster_size 늘리기
min_cluster_size = 15  # 기본 10 → 15
```

## 📈 결과 분석 방법

### 1. cluster_keywords.csv 확인
```bash
# Excel 또는 텍스트 에디터로 열기
open clustering_results/cluster_keywords.csv
```

각 클러스터의 대표 키워드를 확인하여 직무를 파악

### 2. cluster_visualization.png 확인
```bash
open clustering_results/cluster_visualization.png
```

2D 공간에서 클러스터 분포와 밀집도 확인

### 3. cluster_results.json 탐색
```python
import json

with open('clustering_results/cluster_results.json', 'r') as f:
    results = json.load(f)

# 특정 클러스터 확인
cluster_0 = results['clusters']['0']
print(f"크기: {cluster_0['size']}")
print(f"키워드: {cluster_0['keywords']}")
print(f"샘플 제목: {cluster_0['documents'][0]['title']}")
```

## 🎯 다음 단계

클러스터링 완료 후:
1. 클러스터별 대표 문서 선정 (3개)
2. LLM 기반 프로필 쿼리 생성
3. GT 데이터셋 v4 생성
4. 검색 성능 평가

## 📞 문의

문제 발생 시:
1. `cluster_statistics.txt` 확인
2. 콘솔 출력 로그 확인
3. 에러 메시지 복사하여 공유



