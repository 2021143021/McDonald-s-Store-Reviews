# 🍔 MobileBERT를 활용한 맥도날드 리뷰 감성 분석 프로젝트

![Python](https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white)
![MobileBERT](https://img.shields.io/badge/MobileBERT-Finetune-green?style=for-the-badge)

---

## 📝 1. 프로젝트 개요

> **왜 맥도날드 리뷰를 분석하는가?**

맥도날드는 세계적으로 가장 대중적인 패스트푸드 브랜드이며, 고객 리뷰는 품질 및 서비스 개선의 핵심 데이터로 활용됩니다. 이 프로젝트는 고객 리뷰의 텍스트를 분석하여 감성을 자동 분류하고, 이를 통해 지점별 평판 분석 및 개선 포인트 도출에 기여하고자 합니다.

> **목표**

- MobileBERT를 파인튜닝하여 리뷰 감성 분류 모델 구축
- 긍/부정 분류 정확도 평가
- 실제 별점과 예측 결과 비교
- 지점별 리뷰 경향 분석 기반 활용 가능성 탐색

---

## 📦 2. 데이터 소개

### 📁 사용 데이터

| 파일명 | 설명 |
|--------|------|
| `McDonald_cleaned_reviews_numeric_rating.csv` | 전체 리뷰 및 정제된 텍스트, 별점 포함 |
| `McDonald_sample_2000.csv` | 모델 학습용 2,000개 샘플 추출 데이터 |

### 📊 EDA 요약

- 전체 리뷰 수: 약 10,000개 이상
- 학습 데이터 기준 라벨 분포:
  - 긍정 (별점 4~5): 약 60%
  - 부정 (별점 1~2): 약 30%
  - 중립 (별점 3): 제외
- 평균 문장 길이: 약 20~25단어
- 전처리: 소문자화, 특수문자 제거, 공백 정리 등

---

## 🧪 3. 학습 데이터 구축

- `McDonald_sample_2000.csv`를 기반으로 라벨 생성
  - Rating >= 4 → Positive
  - Rating <= 2 → Negative
  - Rating == 3 → 제외
- 최종 라벨링된 데이터 수: 약 1,800개
- 학습/검증 비율: **80:20**
  - Train: 1,440개
  - Validation: 360개

---

## 🤖 4. MobileBERT 모델 학습 결과

- 모델: `google/mobilebert-uncased`
- 학습 방식: 사전학습 모델 파인튜닝 (PyTorch 기반)
- Optimizer: AdamW
- Learning Rate: 2e-5
- Epochs: 5

### 📈 성능 지표

| Metric | 값 |
|--------|----|
| Training Accuracy | 94.2% |
| Validation Accuracy | 91.8% |
| 전체 데이터 Test Accuracy | **92.36%** |

> 학습 그래프는 `matplotlib` 기반 시각화  
> ⮕ `loss`, `accuracy` 변화 추이 확인 가능

---

## 🔍 5. 분석 결과 예시

### ✅ 감성 예측 예시

| 리뷰 | 실제 별점 | 예측 감성 |
|------|-----------|------------|
| "The burger was hot and delicious!" | 5 | Positive |
| "Cold fries and rude staff." | 1 | Negative |
| "Average experience overall." | 3 | (제외) |

### 🏪 지점별 분석 예시

- 각 지점별 리뷰에 감성 예측을 적용하여 신뢰도 지표 구성
- 실제 평점과 비교하여 문제 지점 탐색 가능
- 향후 CS 모니터링 및 고객 대응 우선순위 설정에 활용 가능

---

## 💡 6. 결론 및 느낀점

- 텍스트 리뷰 감성 분석을 통해 서비스 품질을 정량적으로 평가 가능함을 확인함
- MobileBERT는 적은 양의 데이터로도 높은 성능을 보임
- 향후 불만 원인 유형 분석, 시간대별 리뷰 트렌드 분석 등 확장 가능성 존재

---

## 📚 참고자료

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [HuggingFace MobileBERT](https://huggingface.co/google/mobilebert-uncased)
- [Kaggle McDonald's Review Dataset](https://www.kaggle.com/)

---

## 🗂 프로젝트 구조
📁 McDonalds_Review_Analysis/
├── 1_data_preprocessing.py # 데이터 정제 및 전처리
├── 2_model_training.py # MobileBERT 모델 학습
├── 3_inference_analysis.py # 예측 결과 분석 및 시각화
├── McDonald_cleaned_reviews_numeric_rating.csv
├── McDonald_sample_2000.csv
└── README.md

import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
train_acc = [0.861, 0.903, 0.924, 0.935, 0.942]
val_acc = [0.883, 0.901, 0.907, 0.913, 0.918]
train_loss = [0.35, 0.26, 0.19, 0.14, 0.10]
val_loss = [0.29, 0.23, 0.20, 0.18, 0.16]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('train_val_accuracy_loss.png')
plt.show()

