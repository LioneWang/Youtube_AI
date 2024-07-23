# -*- coding = utf-8 -*-
# @Time : 2024/7/23 11:45
# @Author : 王加炜
# @File : evaluate.py
# @Software : PyCharm
import torch
from transformers import BertTokenizer, BertModel
import json
from torch.utils.data import DataLoader
import torch.nn as nn

class QuestionToVideoModel(nn.Module):
    def __init__(self, num_videos):
        super(QuestionToVideoModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.video_id_classifier = nn.Linear(self.bert.config.hidden_size, num_videos)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        video_id_logits = self.video_id_classifier(pooled_output)
        return video_id_logits

class VideoQADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len=512, video_id_mapping=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.video_id_mapping = video_id_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['text']
        video_id = self.video_id_mapping[item['video_id']]

        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'video_id': torch.tensor(video_id, dtype=torch.long)
        }
def load_video_mapping(file_path):
    with open(file_path, 'r') as file:
        mapping = json.load(file)
    return mapping
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
def create_video_id_mapping(data):
    video_ids = sorted(set(item['video_id'] for item in data))
    video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
    return video_id_to_idx, len(video_ids)

def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        video_ids = data['video_id'].to(device)

        video_id_logits = model(input_ids, attention_mask)
        predictions = torch.argmax(video_id_logits, dim=1)

        total_correct += (predictions == video_ids).sum().item()
        total_samples += video_ids.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = load_data('processed_train_data.json')
    video_id_to_idx, _ = create_video_id_mapping(train_data)

    model = QuestionToVideoModel(len(video_id_to_idx)).to(device)
    model.load_state_dict(torch.load('models_4/model_epoch_20.pth'))

    val_dataset = VideoQADataset(train_data, tokenizer, max_len=512, video_id_mapping=video_id_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    accuracy = evaluate(model, val_loader, device)
    print(f'Accuracy on the validation set: {accuracy:.4f}')

if __name__ == "__main__":
    main()
