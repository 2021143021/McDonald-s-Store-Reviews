import torch
import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm

# 1. 디바이스 설정
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device:", device)

# 2. 데이터 불러오기
data_path = "McDonald_cleaned_reviews_numeric_rating.csv"  # 현재 프로젝트 디렉토리에 위치
df = pd.read_csv(data_path, encoding="utf-8")

# 3. 중립 리뷰 제거 (rating=3) 및 결측치 제거
df = df[df['rating'] != 3]
df = df[df['review'].notnull()]

# 4. 레이블 생성: 긍정(1), 부정(0)
df['Sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# 5. 입력 데이터 및 라벨 추출
data_X = df['review'].astype(str).tolist()
labels = df['Sentiment'].values
print(f"총 데이터 수: {len(data_X)}")

# 6. 토크나이저로 토큰화
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased", do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("토큰화 완료")

# 7. TensorDataset 및 DataLoader 구축
batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_data = torch.utils.data.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = torch.utils.data.SequentialSampler(test_data)
test_dataloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("DataLoader 구축 완료")

# 8. 모델 불러오기
model_path = "mobilebert_custom_model_mcdonald_sample2000"  # 저장된 모델 디렉토리명
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 9. 추론
test_pred = []
test_true = []

for batch in tqdm(test_dataloader, desc="예측 중"):
    batch_ids, batch_mask, batch_labels = batch

    batch_ids = batch_ids.to(device)
    batch_mask = batch_mask.to(device)
    batch_labels = batch_labels.to(device)

    with torch.no_grad():
        output = model(batch_ids, attention_mask=batch_mask)
    logits = output.logits
    pred = torch.argmax(logits, dim=1)
    test_pred.extend(pred.cpu().numpy())
    test_true.extend(batch_labels.cpu().numpy())

# 10. 정확도 출력
test_accuracy = np.sum(np.array(test_pred) == np.array(test_true)) / len(test_pred)
print(f"\n전체 리뷰 데이터에 대한 MobileBERT 정확도: {test_accuracy:.4f}")
