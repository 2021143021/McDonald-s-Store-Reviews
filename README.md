## 🍔 MobileBERT를 활용한 맥도날드 리뷰 감정 분석 프로젝트  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>


## 📌 1. 개요

### 1.1 맥도날드와 글로벌 패스트푸드 산업의 성장

1955년 미국 일리노이주에 첫 매장을 연 이후, 맥도날드는 패스트푸드 업계의 선두주자로 자리 잡으며 전 세계적으로 빠르게 성장했다. 특히 20세기 후반부터 글로벌화가 가속화되면서 맥도날드는 빠르고 표준화된 서비스로 전 세계인의 일상생활 속에 깊숙이 자리 잡았다. 오늘날 맥도날드는 전 세계 100여 개 이상의 국가에서 3만 개 이상의 매장을 운영하고 있으며, 전 세계 패스트푸드 시장 점유율의 상당 부분을 차지하고 있다.


![mcd](https://github.com/user-attachments/assets/eab741d3-901f-4710-b21e-844f2688ffb2)


[참고 자료](https://www.mcdonalds.com/us/en-us/about-us.html)

### 1.2 리뷰 플랫폼의 중요성 및 소비자 인식 분석

디지털 시대에 들어서면서 Yelp, Google Reviews, TripAdvisor와 같은 다양한 리뷰 플랫폼이 등장했다. 소비자들은 이러한 플랫폼을 통해 제품 및 서비스에 대한 경험을 공유하며, 이 정보는 다른 소비자의 의사결정에 큰 영향을 미친다. 특히 음식점의 경우 리뷰와 평점이 매장의 매출과 고객 유치에 직접적인 영향을 끼친다는 연구 결과들이 있다.

맥도날드 역시 이러한 리뷰 플랫폼에서 수백만 건의 리뷰가 작성되고 있으며, 이는 맥도날드의 서비스 품질을 평가하는 중요한 지표가 될 수 있다. 리뷰 분석을 통해 소비자들이 맥도날드에 대해 실제로 어떻게 느끼는지 정량적이고 객관적인 자료를 얻을 수 있다.

### 리뷰 플랫폼 예시 이미지


![yelp](https://github.com/user-attachments/assets/15c7c8fd-b303-46d1-a100-1bfbabfba5bc)

![trip](https://github.com/user-attachments/assets/af3c767a-e62b-4f74-ba2c-5e89ff1a2d53)



소비자들은 긍정적 리뷰를 통해 브랜드에 대한 신뢰를 형성하고, 부정적 리뷰를 통해 불만을 표출하거나 개선을 촉구한다. 따라서 기업 입장에서는 이 리뷰를 분석하고 관리하는 것이 필수적이다.

### 1.3 프로젝트의 목적 및 감정 분석 활용방안

#### 📍 프로젝트 배경 및 필요성

패스트푸드 브랜드 중 가장 대표적인 맥도날드는 전 세계 100여 개 이상의 국가에서 3만 개 이상의 매장을 운영하며 매일 수백만 명의 소비자가 방문하는 글로벌 프랜차이즈이다. 맥도날드는 빠르고 간편한 식사를 제공하여 현대인의 바쁜 라이프스타일과 밀접한 관련이 있기 때문에 소비자와 기업의 지속적인 관심 대상이다.

하지만 맥도날드의 높은 인기에도 불구하고 모든 소비자가 항상 긍정적인 경험을 하는 것은 아니다. 소비자들이 실제로 맥도날드를 어떻게 평가하는지 알기 위해서는 그들이 직접 작성한 리뷰 분석이 매우 중요하다. 특히 부정적인 리뷰를 정확히 분석하여 원인을 찾는 작업은 브랜드의 장기적인 발전과 품질 향상에 필수적이다.

---

#### 🎯 프로젝트 목적

이번 프로젝트에서는 전 세계 맥도날드 매장의 소비자 리뷰 데이터를 활용해 리뷰 텍스트가 표현하는 감정을 정확히 분석하고자 한다. 리뷰는 소비자의 실제 경험을 반영하며, 서비스 품질과 소비자 만족도 평가의 핵심 자료가 될 수 있다. 구체적으로는 다음을 목표로 한다.

- 맥도날드 리뷰 데이터를 수집하고 긍정 및 부정 감정으로 명확히 분류
- MobileBERT 모델(최신 NLP 모델)을 이용해 리뷰 텍스트 감정 정확도 분석
- 소비자들이 긍정적 또는 부정적 평가를 내리는 핵심 이유 파악

---

#### 📌 감정 분석 활용방안

리뷰 데이터의 감정 분석을 통해 얻은 결과는 다음과 같은 방면에서 활용될 수 있다.

- **서비스 품질 개선**: 부정적 리뷰에서 반복적으로 언급되는 이슈를 분석해 맥도날드가 실제로 개선할 수 있는 분야를 정확히 찾아낼 수 있다.
- **고객 만족도 제고**: 감정 분석을 통해 소비자의 불만족 요소를 명확히 이해하고 이를 해결함으로써 고객의 만족도를 높일 수 있다.
- **마케팅 전략 개선**: 소비자들이 자주 언급하는 긍정적 요소를 활용하여 더 효과적인 마케팅 전략을 마련할 수 있다.

---

이러한 프로젝트는 맥도날드가 글로벌 프랜차이즈로서 소비자에게 지속적으로 긍정적인 경험을 제공하는 데 필요한 객관적이고 실질적인 자료를 제공할 것으로 기대된다.


## 📊 2. 데이터

### 2.1 원시 데이터

본 프로젝트에서 사용한 데이터는 Kaggle에서 제공하는 맥도날드 리뷰 데이터셋이다.  
[[맥도날드 리뷰 데이터셋 바로가기]](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews/data)

* 데이터 구성

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| rating | 소비자가 매긴 평점 (1점~5점) | 5 |
| review | 소비자가 남긴 리뷰 텍스트 | "정말 맛있어요!" |

![Image](https://github.com/user-attachments/assets/2ab2f5ab-2bcf-44f6-bbff-2910e7ee4153)

총 원본 데이터 수는 **33,396개**이며, 결측치와 중립 리뷰(3점)는 제외하고 최종적으로 **28,570개**의 데이터를 분석에 사용했다.

### 2.2 데이터 필터링 기준

본 프로젝트의 목표가 명확한 긍정 및 부정 리뷰 분석이므로 중립적인 리뷰는 제외했다.

- 긍정 리뷰 (4~5점): **15,935개**
- 부정 리뷰 (1~2점): **12,635개**

> 최종 사용 데이터: 총 **28,570개**

### 2.3 데이터의 리뷰 평점 분포

리뷰 평점 분포를 시각화하여 리뷰 데이터의 특성을 분석했다.

![Image](https://github.com/user-attachments/assets/60c46420-980a-4127-9f85-b5060a6e5064)

- 4점과 5점의 긍정적 평점이 다수였으나, 1점과 2점의 부정적 평점도 무시할 수 없는 비율로 나타났다.

### 2.4 리뷰 문장 길이 분석

리뷰 텍스트의 길이를 분석하여 사용자의 리뷰 작성 특성을 파악했다.

| 항목      | 리뷰 길이(글자 수) |
|-----------|-------------------|
| 최소 길이 | 1자               |
| 최대 길이 | 2,000자 이상      |
| 평균 길이 | 약 255자          |


![Image](https://github.com/user-attachments/assets/334f6480-505e-4c72-bbf1-82024ae77dc1)


- 리뷰의 대부분은 간략하고 짧게 작성되었으며 평균 약 255자였다.
- 긴 리뷰는 상대적으로 적었지만, 일부 사용자는 상세하고 긴 리뷰를 남기기도 했다.

### 2.5 리뷰 감정(긍정/부정) 분포

리뷰 데이터를 감성 분석하여 긍정과 부정의 분포를 분석하였다.  
긍정 리뷰가 전체의 약 55.8%를 차지하며 우세했지만, 부정 리뷰 또한 44.2%로 적지 않은 비중을 보였다.  
이는 맥도날드의 서비스 품질 개선을 위해 부정적인 의견에 주목할 필요가 있음을 시사한다.

![Image](https://github.com/user-attachments/assets/23adb401-e4b3-434b-a93f-7f46750bfc91)

- 👍 긍정 리뷰: 약 55.8%  
- 👎 부정 리뷰: 약 44.2%

> 부정 리뷰에 담긴 핵심 이슈를 심층 분석하여 **소비자 만족도 향상**을 도모하는 것이  
> 본 프로젝트의 주요 목적 중 하나이다.


## 3. 📚 학습 데이터 구축


학습 데이터 비율에 따른 차이점을 확인하기 위해, 두 가지 학습 데이터를 준비하였다.  
첫 번째 학습 데이터는 **긍정 리뷰 1,000건**과 **부정 리뷰 1,000건**을 추출하여 총 **2,000건**으로 구성하였다.  
두 번째 학습 데이터는 전체 리뷰 데이터 비율(긍정 약 55.8%, 부정 약 44.2%)에 따라 **10% 비율(약 2,857건)**을 추출하였다.  
긍정 리뷰 **1,593건**과 부정 리뷰 **1,264건**으로 구성된 총 **2,857건**의 데이터를 사용하였다.

---

* 각 학습 데이터별 리뷰 개수

| 학습 데이터 종류     | 긍정 리뷰 수 | 부정 리뷰 수 |
|----------------------|-------------|-------------|
| 첫 번째 학습 데이터 | 1,000건     | 1,000건     |
| 두 번째 학습 데이터 | 1,593건     | 1,264건     |

---

* 데이터 비율 분포

 ![Image](https://github.com/user-attachments/assets/953f716f-c18b-4cc9-9f50-bf15c70cae03)


- 첫 번째 데이터는 **감정 균형(1:1)**을 맞춰 모델 편향을 최소화하였다.  
- 두 번째 데이터는 **원본 분포(55.8% vs 44.2%)**를 유지하여 현실 반영 성능을 평가하였다.  
- 두 방식을 비교 분석하여 모델의 일반화 성능과 현실 반영 성능을 모두 확인하였다.


## 4. 🤖MobileBERT 학습 결과

### 🛠 개발환경

<p align="center">
  <img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />
</p>

### 📦 사용 패키지

<p align="center">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/transformers-%23E04CFF.svg?&style=for-the-badge&logo=transformers&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />
</p>

---

### 4.1️⃣ 첫 번째 실험 (2,000건 샘플 데이터)

| Epoch | Train Loss     | Train Accuracy | Validation Accuracy |
|:-----:|:---------------|:---------------|:-------------------:|
| 1     | 282,444.8209   | 91.81%         | 91.75%              |
| 2     | 0.3823         | 95.75%         | 93.25%              |
| 3     | 0.1786         | 97.06%         | 93.25%              |
| 4     | 1.0507         | 97.25%         | 93.50%              |

<p align="center">
  <img src="./images/sample_loss_plot.png" width="45%" alt="샘플 데이터 손실 그래프" />
  <img src="./images/sample_acc_plot.png"  width="45%" alt="샘플 데이터 정확도 그래프" />
</p>

---

### 4.2️⃣ 두 번째 실험 (28,570건 전체 데이터)

| Epoch | Train Loss   | Train Accuracy | Validation Accuracy |
|:-----:|:-------------|:---------------|:-------------------:|
| 1     | 18,483.2734  | 95.38%         | 93.65%              |
| 2     | 0.4730       | 96.86%         | 94.33%              |
| 3     | 0.2624       | 97.67%         | 94.96%              |
| 4     | 0.1359       | 97.88%         | 95.03%              |

<p align="center">
  <img src="./images/full_loss_plot.png" width="45%" alt="전체 데이터 손실 그래프" />
  <img src="./images/full_acc_plot.png"  width="45%" alt="전체 데이터 정확도 그래프" />
</p>

---

### 4.3️⃣ 전체 데이터셋 Inference 결과

- **Test Accuracy:** 92.36%  
  (샘플 기반 모델을 전체 28,570건에 적용하여 평가)

```text
Using device: cuda
...
전체 리뷰 데이터에 대한 MobileBERT 정확도: 0.9236
Process finished with exit code 0



## 4. 🤖 MobileBERT Finetuning 결과

### 🛠 개발 환경

<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white" />

### 📦 주요 패키지

<img src="https://img.shields.io/badge/pandas-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/tensorflow-%23FF6F00.svg?&style=for-the-badge&logo=tensorflow&logoColor=white" />

---

### 4.1️⃣ 학습 결과 비교

| **첫 번째 학습 데이터** | Epoch | 0 | 1 | 2 | 3 | **두 번째 학습 데이터** | Epoch | 0 | 1 | 2 | 3 |
|--------------------------|--------|--------|--------|--------|--------|---------------------------|--------|--------|--------|--------|--------|
| Training Loss            | 7.4573e+4 | 0.59 | 0.25 | 0.23 | Training Loss             | 1.3910e+4 | 0.32 | 0.23 | 0.36 |
| Validation Accuracy      | 0.84 | 0.86 | 0.86 | 0.87 | Validation Accuracy       | 0.76 | 0.89 | 0.87 | 0.89 |

📊 시각화 그래프:  
<p align="center">
  <img width="600" src="./images/training_result_graph.png" alt="MobileBERT Training Result">
</p>

- 첫 번째 모델은 초기 Training Loss가 높았으나 빠르게 감소하여 **안정적인 성능**을 보였다.  
- 두 번째 모델은 빠르게 학습되었지만, 후반에 **과적합의 가능성**을 보인다.  
- 결과적으로 첫 번째 모델이 **더 안정적인 학습 곡선**을 보여주었다.

---

### 4.2️⃣ 전체 데이터셋 Inference 결과

- 학습된 모델을 전체 리뷰 데이터(28,570건)에 적용한 결과는 다음과 같다.

```
Using device: cuda
전체 리뷰 데이터에 대한 MobileBERT 정확도: 0.9236
Process finished with exit code 0
```

✅ **최종 Test Accuracy: 92.36%**

📌 전체 리뷰를 기준으로 학습된 모델의 **일반화 성능이 매우 높음**을 확인할 수 있다.






