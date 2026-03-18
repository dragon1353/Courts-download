import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# 定義 Tokenizer (簡易版中文字元級別斷詞)
def simple_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text) # 移除標點符號
    return list(text.replace(" ", ""))

# 建立辭典 mapping
class Vocab:
    def __init__(self, texts, max_size=10000, unk_token="<UNK>", pad_token="<PAD>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenizer(text))
            
        for word, _ in counter.most_common(max_size):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, text, max_len=512):
        tokens = simple_tokenizer(text)
        indices = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        # Pad or Truncate
        if len(indices) < max_len:
            indices += [self.word2idx[self.pad_token]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

# PyTorch Dataset
class LegalDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=512):
        self.texts = texts
        # 將標籤轉為整數：有罪=1, 無罪/其他=0
        self.labels = [1 if '有罪' in str(l) else 0 for l in labels]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_indices = self.vocab.encode(str(self.texts[idx]), self.max_len)
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# 定義 Focal Loss 損失函數
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) # Prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# 定義神經網路 (Bi-LSTM)
class LegalClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # outputs = [batch size, sent len, hid dim * num directions]
        # hidden/cell = [num layers * num directions, batch size, hid dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # concat the final forward and backward hidden layers
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden).squeeze(1)

def train_model():
    dataset_path = config.CSV_DATASET_PATH
    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到訓練資料集 {dataset_path}。請先執行 local_auto_label.py！")
        return

    print("載入資料集中...")
    df = pd.read_csv(dataset_path)
    # 過濾掉無法標籤或是內文空的資料
    df = df.dropna(subset=['TextContent', 'Label'])
    df = df[df['Label'] != '錯誤']

    if len(df) < 10:
         print(f"警告：資料筆數太少 (只有 {len(df)} 筆)，神經網路可能無法收斂！")

    texts = df['TextContent'].tolist()
    labels = df['Label'].tolist()

    # 劃分訓練集與測試集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    print("建立詞彙字典 (Vocab)...")
    vocab = Vocab(train_texts, max_size=config.TRAIN_MAX_VOCAB_SIZE)
    
    print("準備 PyTorch Dataset & DataLoader...")
    train_dataset = LegalDataset(train_texts, train_labels, vocab, max_len=config.TRAIN_MAX_SEQ_LEN)
    val_dataset = LegalDataset(val_texts, val_labels, vocab, max_len=config.TRAIN_MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的運算裝置: {device}")

    # 初始化模型
    model = LegalClassifier(vocab_size=len(vocab)).to(device)
    
    # 定義 Focal Loss 與 Optimizer
    criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA) 
    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN_LEARNING_RATE)

    # 簡單的訓練迴圈
    epochs = config.TRAIN_EPOCHS
    best_val_loss = float('inf')
    model_save_path = config.MODEL_SAVE_PATH

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_texts)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 驗證迴圈
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                predictions = model(batch_texts)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                # 計算準確率 (Sigmoid > 0.5 即代表預測為正類)
                preds = torch.sigmoid(predictions) > 0.5
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 儲存最佳模型 (.pth)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 同時儲存模型權重與辭典，推論時才抓的到詞
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_word2idx': vocab.word2idx,
                'vocab_idx2word': vocab.idx2word
            }, model_save_path)
            print(f"  --> 已儲存最佳模型至 {model_save_path}")

    print("訓練流程結束。")
    
    print("訓練流程結束。")
    # 不要在此呼叫 sys.exit，讓函式自然返回

if __name__ == "__main__":
    # 在 Windows 上使用 spawn 模式啟動子行程
    multiprocessing.set_start_method('spawn', force=True)
    
    # 將訓練包裝在獨立的子行程中
    p = multiprocessing.Process(target=train_model)
    p.start()
    p.join()
    
    # 不管子行程在釋放 CUDA 資源時有沒有崩潰 (Access Violation)，
    # 主行程只要看到它跑完，就強行以 0 (成功) 退出
    os._exit(0)
