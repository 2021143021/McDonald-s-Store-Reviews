# 🍟 McDonald's 리뷰 기반 지점별 평점 분석

---

> 🔍 맥도날드 리뷰를 분석하여 지점별 고객 만족도를 예측하고, 실제 평점과 비교해 신뢰도를 평가합니다.

## 1. 프로젝트 개요

본 프로젝트는 MobileBERT 사전학습 모델을 활용하여 맥도날드 리뷰 데이터를 감성 분석(긍정/부정 분류)하는 모델을 구축하는 것을 목표로 합니다.  
학습 데이터는 약 2,000개의 리뷰로 구성되며, 평점 4점 이상을 긍정, 1~2점을 부정으로 라벨링 하여 분류 모델을 학습시켰습니다.

---

## 2. 데이터 설명 및 전처리

- 데이터 파일: `McDonald_sample_2000.csv`  
- 컬럼: `review` (텍스트), `rating` (평점)  
- 중립 평점(3점) 및 결측치(`NaN`) 리뷰 제거  
- 평점 기준 라벨 생성:  
  - 긍정(1) : 평점 >= 4  
  - 부정(0) : 평점 <= 2

---

## 3. 모델 및 학습 설정

- 모델: `google/mobilebert-uncased` MobileBERT 기반 분류 모델  
- 최대 토큰 길이: 256 (패딩 포함)  
- 배치 사이즈: 8  
- 옵티마이저: AdamW (learning rate=2e-5)  
- 학습 에폭: 4  
- 학습 데이터 80%, 검증 데이터 20% 분리  
- 스케줄러: 선형 warm-up 스케줄러 사용  
- GPU 환경에서 학습 수행  

---

## 4. 학습 과정 및 시각화

학습 과정에서 기록한 손실과 정확도를 시각화하면 다음과 같습니다.
mkdir images

![Loss and Accuracy over Epochs](./images/loss_accuracy_plot.png)

- **왼쪽 그래프**: 학습 손실(epoch별 평균) 감소 추세  
- **오른쪽 그래프**: 학습 및 검증 정확도 상승 추세

---

## 5. 평가 결과

| Epoch | Train Loss | Train Accuracy | Validation Accuracy |
|-------|------------|----------------|---------------------|
| 1     | 0.5401     | 0.7650         | 0.7400              |
| 2     | 0.3854     | 0.8475         | 0.8300              |
| 3     | 0.2857     | 0.8950         | 0.8650              |
| 4     | 0.2103     | 0.9250         | 0.8850              |

*※ 위 값들은 예시이며, 실제 학습 결과에 맞게 수정하세요.*

---

## 6. 모델 저장 및 활용

- 학습 완료된 모델은 `mobilebert_custom_model_mcdonald_sample2000` 디렉토리에 저장하였습니다.  
- 저장된 모델은 `from_pretrained()` 메서드로 불러와 감성 예측에 바로 사용할 수 있습니다.  
- 추론 예시:

```python
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch

model = MobileBertForSequenceClassification.from_pretrained("mobilebert_custom_model_mcdonald_sample2000")
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

text = "The fries were great but the service was slow."
inputs = tokenizer(text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
print("긍정" if prediction == 1 else "부정")



---

## 2. 학습 결과 시각화용 Python 코드

```python
import matplotlib.pyplot as plt
import numpy as np

# 실제 학습 종료 후 저장한 epoch_result 리스트 예시
# epoch_result = [(train_loss1, train_acc1, val_acc1), (train_loss2, train_acc2, val_acc2), ...]
epoch_result = [
    (0.5401, 0.7650, 0.7400),
    (0.3854, 0.8475, 0.8300),
    (0.2857, 0.8950, 0.8650),
    (0.2103, 0.9250, 0.8850)
]

train_losses = [x[0] for x in epoch_result]
train_accs = [x[1] for x in epoch_result]
val_accs = [x[2] for x in epoch_result]
epochs = range(1, len(epoch_result) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', color='blue', label='Train Loss')
plt.title('Train Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, marker='o', color='green', label='Train Accuracy')
plt.plot(epochs, val_accs, marker='o', color='red', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('./images/loss_accuracy_plot.png')  # 프로젝트 폴더 내 images 폴더에 저장
plt.show()
