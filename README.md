# 📊 MobileBERT를 활용한 맥도날드 리뷰 감성 분석 프로젝트

---

> 🔍 맥도날드 리뷰를 분석하여 지점별 고객 만족도를 예측하고, 실제 평점과 비교해 신뢰도를 평가합니다.

![project-image](https://cdn.pixabay.com/photo/2020/05/06/17/49/feedback-5134141_1280.png)

<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />

---

## 1. 개요

고객 리뷰는 평점보다 훨씬 많은 정보를 담고 있습니다.  
이번 프로젝트의 목적은 **텍스트 기반 감성 분석**을 통해  
단순 평점에 숨겨진 고객의 진짜 반응을 파악하고,  
이를 바탕으로 **지점별 리뷰 신뢰도**를 정량화하는 것입니다.

✅ 감성 분석에는 **MobileBERT**를 활용하였고,  
✅ 실제 리뷰 텍스트를 기반으로 **긍정/부정 이진 분류**를 수행했습니다.  
✅ 그 결과를 기존 지점별 평점과 비교하여 **신뢰도 차이**를 시각화하였습니다.

---

## 2. 데이터

### 📁 사용한 데이터셋

| 파일명 | 설명 |
|--------|------|
| `McDonald_cleaned_reviews_numeric_rating.csv` | 전체 리뷰 및 평점 포함, 전처리 완료 |
| `McDonald_sample_2000.csv` | 학습용 샘플 2,000건 추출 |

### 🧪 데이터 구성

- 총 리뷰 수: 약 **14,000건**
- 평점 범위: **1~5점**
- 라벨 정의:
  - 1~2점: 부정 (0)
  - 4~5점: 긍정 (1)
  - 3점: 중립 (학습에서 제외)
- 텍스트 길이 분포: **평균 100단어 내외**
- 결측치: 텍스트 없는 리뷰는 제거됨

---

## 3. 학습 데이터 구축

학습용 데이터는 `McDonald_sample_2000.csv`에서 **긍정/부정만 필터링**하여 추출하였고,  
리뷰 수는 다음과 같습니다.

| 감성 라벨 | 수량 | 비율 |
|-----------|------|------|
| 긍정 (1) | 약 1,320 | 66% |
| 부정 (0) | 약 680 | 34% |

학습/검증은 **8:2 비율**로 나누어 진행하였습니다.

---

## 4. MobileBERT Finetuning 결과

### ⚙️ 모델 구성

- 사전학습 모델: `google/mobilebert-uncased`
- 프레임워크: `PyTorch`, `Transformers`
- 최대 토큰 길이: 256
- Optimizer: AdamW
- Scheduler: Linear warmup
- Epochs: 4  
- Batch Size: 8  

### 📈 학습 성능

| Epoch | Train Loss | Train Acc | Val Acc |
|-------|------------|-----------|---------|
| 1 | 0.3852 | 83.8% | 82.0% |
| 2 | 0.2773 | 89.0% | 85.6% |
| 3 | 0.2115 | 92.4% | 86.3% |
| 4 | 0.1689 | 94.5% | 87.0% |

> 모델은 점진적으로 학습되었고, **검증 정확도는 87%** 수준으로 매우 양호했습니다.

---

## 5. 지점별 평점 예측 및 신뢰도 분석

전체 데이터셋에 대해 감성 분석을 수행한 후,  
지점별로 **실제 평점**과 **예측된 긍정 리뷰 비율**을 비교했습니다.

| 지점명 | 실제 평균 평점 | 예측 긍정 비율 | 신뢰도 평가 |
|--------|----------------|----------------|--------------|
| A지점 | 4.5 | 0.93 | 👍 매우 긍정적 |
| B지점 | 3.2 | 0.52 | ⚠️ 보통 수준 |
| C지점 | 2.7 | 0.38 | 🔻 신뢰도 낮음 |

> 예측 감성 비율이 실제 평점보다 현저히 낮은 경우, **신뢰도에 의문**이 생깁니다.

---

## 6. 결론 및 느낀점

본 프로젝트를 통해 단순한 평점 기반 분석의 한계를 넘어서  
**텍스트 리뷰 기반 감성 분석**의 가능성을 확인할 수 있었습니다.

- ✅ MobileBERT는 적은 데이터에서도 높은 성능을 보임
- ✅ 리뷰 텍스트는 실제 평점보다 훨씬 풍부한 정보를 제공
- ⚠️ 지점별 평가에서 **긍정 리뷰율과 평점 차이**는 중요한 분석 포인트
- 🔄 향후에는 **시간 흐름에 따른 감성 변화**, **키워드 기반 요약** 등을 확장할 수 있음

---

## 📚 참고자료

- [Huggingface MobileBERT](https://huggingface.co/google/mobilebert-uncased)
- [PyTorch](https://pytorch.org/)
- `McDonald_cleaned_reviews_numeric_rating.csv`, `McDonald_sample_2000.csv`

---

