import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# GPU 사용 여부 확인
GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print("Using device:", device)

# 경고 메시지 제거
logging.set_verbosity_error()

# 1. 데이터 불러오기
path = "McDonald_sample_2000.csv"  # 샘플 데이터 경로
df = pd.read_csv(path, encoding='utf-8')

# 2. 중립(3점) 제거 및 결측치 제거
df = df[df['rating'] != 3]
df = df[df['review'].notnull()]  # review가 NaN인 경우 제거

# 3. 레이블 생성 (긍정: 1, 부정: 0)
df['Sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# 4. 입력 데이터 준비
data_X = df['review'].astype(str).tolist()  # 문자열로 변환
labels = df['Sentiment'].values

# 5. 토크나이저
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 6. 학습용 및 검증용 데이터셋 분리
train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.2, random_state=2025)
train_mask, validation_mask, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2025)

# 7. DataLoader 설정
batch_size = 8

train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_mask)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_mask)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 8. 모델 초기화 및 옵티마이저 설정
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

# 9. 학습 및 검증
epoch_result = []
for e in range(epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}", leave=True)

    for batch in progress_bar:
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        model.zero_grad()
        output = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    # 학습 데이터 정확도 측정
    model.eval()
    train_pred = []
    train_true = []

    for batch in tqdm(train_dataloader, desc=f"Evaluation Train Epoch {e + 1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)

        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        train_pred.extend(pred.cpu().numpy())
        train_true.extend(batch_labels.cpu().numpy())

    train_accuracy = (np.array(train_pred) == np.array(train_true)).sum() / len(train_pred)

    # 검증 데이터 정확도 측정
    val_pred = []
    val_true = []

    for batch in tqdm(validation_dataloader, desc=f"Validation Epoch {e + 1}", leave=True):
        batch_ids, batch_mask, batch_labels = batch
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            output = model(batch_ids, attention_mask=batch_mask)

        logits = output.logits
        pred = torch.argmax(logits, dim=1)
        val_pred.extend(pred.cpu().numpy())
        val_true.extend(batch_labels.cpu().numpy())

    val_accuracy = (np.array(val_pred) == np.array(val_true)).sum() / len(val_pred)
    epoch_result.append((avg_train_loss, train_accuracy, val_accuracy))

# 결과 출력
for idx, (loss, train_acc, val_acc) in enumerate(epoch_result, start=1):
    print(f"Epoch {idx}: Train loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

# 모델 저장
print("\n모델 저장 중...")
save_path = "mobilebert_custom_model_mcdonald_sample2000"
model.save_pretrained(save_path)
print("모델 저장 완료!")
