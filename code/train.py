import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn.functional as F
import numpy as np

def pad_sequence(batch):
    # 找出批量中最长的时间步
    max_len = max([x.size(0) for x, _, _, _ in batch])
    
    # 对每个样本进行填充
    padded_batch = []
    for x, binary_label, phq8_score, gender in batch:
        padded_x = F.pad(x, (0,0,0,0,0, max_len - x.size(0)), "constant", 0)
        padded_batch.append((padded_x, binary_label, phq8_score, gender))
        
    
    # 将所有样本堆叠成批量
    spectrograms = torch.stack([x for x, _, _, _ in padded_batch])
    binary_labels = torch.stack([binary_label for _, binary_label, _, _ in padded_batch])
    phq8_scores = torch.stack([phq8_score for _, _, phq8_score, _ in padded_batch])
    genders = torch.stack([gender for _, _, _, gender in padded_batch])
    
    return spectrograms, binary_labels, phq8_scores, genders

class SpectrogramDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels
        self.participant_ids = list(specs.keys())

    def __len__(self):
        return len(self.participant_ids)

    def __getitem__(self, idx):
        participant_id = self.participant_ids[idx]
        spectrograms = self.specs[participant_id]
        binary_label, phq8_score, gender = self.labels[participant_id]
        spectrogram_tensor = torch.tensor(spectrograms, dtype=torch.float32)
        return spectrogram_tensor, torch.tensor(binary_label, dtype=torch.long), torch.tensor(phq8_score, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes, aux_input_dim=1):
        super(MultiHeadAttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(input_dim + aux_input_dim, 128)  # 加入辅助特征维度
        self.fc2_binary = nn.Linear(128, num_classes)
        self.fc2_regression = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, aux_input):
        # x的形状为 (batch_size, num_spectrograms, time_steps, freq_bins)
        batch_size, num_spectrograms, time_steps, freq_bins = x.size()
        x = x.view(batch_size * num_spectrograms, time_steps, freq_bins)  # 展平batch和num_spectrograms
        x = x.transpose(0, 1)  # 转换形状为 (time_steps, batch_size * num_spectrograms, freq_bins)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)  # 转回 (batch_size * num_spectrograms, time_steps, freq_bins)
        attn_output = attn_output.view(batch_size, num_spectrograms, time_steps, freq_bins)
        attn_output = torch.mean(attn_output, dim=2)  # 在time_steps维度上做平均池化
        attn_output = torch.mean(attn_output, dim=1)  # 在num_spectrograms维度上做平均池化
        
        # 加入辅助输入（性别信息）
        aux_input = aux_input.unsqueeze(1).float()
        attn_output = torch.cat((attn_output, aux_input), dim=1)
        
        x = self.fc1(attn_output)
        x = self.dropout(x)
        
        binary_output = self.fc2_binary(x)
        regression_output = self.fc2_regression(x).squeeze(1)
        
        return binary_output, regression_output


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
with open('feature/train_spec.pickle', 'rb') as f:
    train_spec = pickle.load(f)

with open('feature/train_label.pickle', 'rb') as f:
    train_labels = pickle.load(f)
    
# 创建数据集和数据加载器
dataset = SpectrogramDataset(train_spec, train_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_sequence)

# 设置参数
input_dim = train_spec[next(iter(train_spec))][0].shape[1]  # 频谱图的频率维度
num_heads = 4
num_classes = 2

# 初始化模型、损失函数和优化器
model = MultiHeadAttentionModel(input_dim, num_heads, num_classes).to(device)
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for spectrograms, binary_labels, phq8_scores, genders in dataloader:
        spectrograms, binary_labels, phq8_scores, genders = spectrograms.to(device), binary_labels.to(device), phq8_scores.to(device), genders.to(device)
        optimizer.zero_grad()
        binary_outputs, regression_outputs = model(spectrograms, genders)
        loss_classification = criterion_classification(binary_outputs, binary_labels)
        loss_regression = criterion_regression(regression_outputs, phq8_scores)
        loss = loss_classification + loss_regression
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
# 模型评估（示例）
model.eval()
correct_classification = 0
total_classification = 0
total_regression_loss = 0
with torch.no_grad():
    for spectrograms, binary_labels, phq8_scores, genders in dataloader:
        spectrograms, binary_labels, phq8_scores, genders = spectrograms.to(device), binary_labels.to(device), phq8_scores.to(device), genders.to(device)
        binary_outputs, regression_outputs = model(spectrograms, genders)
        _, predicted = torch.max(binary_outputs.data, 1)
        total_classification += binary_labels.size(0)
        correct_classification += (predicted == binary_labels).sum().item()
        total_regression_loss += criterion_regression(regression_outputs, phq8_scores).item()

classification_accuracy = 100 * correct_classification / total_classification
mean_regression_loss = total_regression_loss / len(dataloader)
print(f'Classification Accuracy: {classification_accuracy}%')
print(f'Mean Regression Loss: {mean_regression_loss}')