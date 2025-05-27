# McDonald-s-Store-Reviews
캡스톤프로젝트 기말
# 🍔 맥도날드 리뷰 감성 분석 프로젝트 (Sentiment Analysis on McDonald's Reviews using MobileBERT)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/pytorch-1.13+-EE4C2C.svg" />
  <img src="https://img.shields.io/badge/transformers-MobileBERT-green.svg" />
  <img src="https://img.shields.io/badge/task-Text_Classification-yellow.svg" />
</p>

---

## 📌 프로젝트 개요

맥도날드는 전 세계적으로 사랑받는 글로벌 패스트푸드 브랜드로, 고객 리뷰는 브랜드 이미지와 매출에 직결되는 중요한 지표입니다. 본 프로젝트에서는 사전 수집된 맥도날드 리뷰 데이터를 기반으로 리뷰의 감성을 자동으로 분류하는 모델을 학습하였습니다. 이를 통해 실제 매장 또는 메뉴에 대한 긍·부정 반응을 예측할 수 있으며, 향후 매장 서비스 개선, 마케팅 전략 수립 등 다양한 분야에 활용될 수 있습니다.

---

## 🗃️ 데이터셋 소개

### 📄 사용한 데이터

- `McDonald_cleaned_reviews_numeric_rating.csv`  
  → 전처리 완료된 전체 리뷰 데이터셋 (약 9,000개)
- `McDonald_sample_2000.csv`  
  → 모델 학습을 위한 샘플 데이터셋 (2,000개 샘플)

### ✅ 주요 전처리 사항

- **결측치 제거** (NaN 리뷰 제거)
- **중립 리뷰 제외** (별점 3점 제거)
- **감정 라벨링**
  - 평점 1~2점 → `부정(0)`
  - 평점 4~5점 → `긍정(1)`

### 📊 EDA 요약

| 항목              | 값            |
|------------------|----------------|
| 총 리뷰 수        | 2,000건 (샘플셋 기준) |
| 긍정 리뷰 수      | 1,315건       |
| 부정 리뷰 수      | 685건         |
| 평균 문장 길이     | 약 40~60 토큰 |
| 최대 문장 길이     | 256 토큰 (모델 입력 제한 적용) |

---

## 🧪 모델 및 학습 구성

### ✅ 모델 구조

- **Pretrained Model**: `MobileBERT (google/mobilebert-uncased)`
- **Fine-tuning Task**: 이진 감정 분류 (`긍정/부정`)
- **최대 시퀀스 길이**: 256
- **토크나이저**: `MobileBertTokenizer`

### ⚙️ 학습 환경

- **Batch size**: 8
- **Optimizer**: AdamW (`lr=2e-5`)
- **Scheduler**: Linear with Warmup
- **Epochs**: 4
- **Train:Validation split**: 8:2

### 📁 학습 코드 구성

| 파일명 | 설명 |
|--------|------|
| `1_data_prepare.ipynb` | 데이터 전처리 및 샘플링, 라벨링 |
| `2_model_training.ipynb` | MobileBERT 파인튜닝 및 성능 평가 |
| `3_inference_analysis.ipynb` | 예측 결과 분석 및 시각화 |

---

## 📈 모델 성능 결과

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
|-------|------------|----------------|---------------------|
| 1     | 0.3257     | 0.8713         | 0.8693              |
| 2     | 0.2264     | 0.9225         | 0.8875              |
| 3     | 0.1598     | 0.9481         | 0.9011              |
| 4     | 0.1089     | 0.9662         | 0.9125              |

- **최종 검증 정확도**: `91.25%`
- 성능은 훈련 데이터셋과 검증셋 모두에서 안정적으로 향상되었으며, 과적합 현상은 크게 나타나지 않음

---

## 🔍 예측 결과 분석

모델을 실제 리뷰에 적용해본 결과, 다음과 같은 인사이트를 확인할 수 있었습니다:

- 부정 리뷰 예시:
  > "The fries were cold and the staff was very rude." → **부정(0)** 예측

- 긍정 리뷰 예시:
  > "Excellent service and the burger was fresh!" → **긍정(1)** 예측

추후에는 리뷰 내용을 기반으로 매장별/메뉴별 평가 요약 기능 또는 감정 트렌드 분석 기능으로 확장 가능성이 있습니다.

---

## 🧠 프로젝트 느낀점

이번 프로젝트를 통해 다음을 체험하고 학습했습니다:

- 사전학습 언어 모델(MobileBERT)을 활용한 파인튜닝 실습
- 텍스트 분류 문제에 적절한 데이터 정제 및 라벨링 전략
- 훈련/검증 분리와 성능 평가 기준 이해
- 실제 현업 데이터와 유사한 형태의 리뷰 데이터를 다루는 경험

---

## 📌 향후 개선 방향

- 보다 많은 리뷰 데이터를 이용한 학습으로 모델의 일반화 성능 향상
- 문장 길이 가변 적용 및 attention 시각화 등 모델 해석 기능 추가
- 메뉴/매장별 세분화된 분석과 감정 요약 시스템 개발

---

## 📁 디렉토리 구조

---

## 📚 참고자료

1. [KCI 논문: 영화리뷰와 흥행관계](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001954434)
2. [HuggingFace MobileBERT 모델](https://huggingface.co/google/mobilebert-uncased)
3. [PyTorch 공식 문서](https://pytorch.org/)
4. [Transformers 문서](https://huggingface.co/docs/transformers/index)

---

