import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import json
import os

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

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_video_id_mapping(data):
    video_ids = sorted(set(item['video_id'] for item in data))
    video_id_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}
    return video_id_to_idx, len(video_ids)

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

def train(model, data_loader, loss_fn_video_id, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            video_ids = batch['video_id'].to(device)

            optimizer.zero_grad()
            video_id_logits = model(input_ids, attention_mask)
            
            loss_video_id = loss_fn_video_id(video_id_logits, video_ids)
            loss_video_id.backward()
            optimizer.step()
            
            total_loss += loss_video_id.item()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

def main():
    train_data_path = 'processed_train_data.json'
    train_data = load_data(train_data_path)
    video_id_to_idx, num_videos = create_video_id_mapping(train_data)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = VideoQADataset(train_data, tokenizer, max_len=512, video_id_mapping=video_id_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuestionToVideoModel(num_videos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn_video_id = nn.CrossEntropyLoss()

    # Load the last saved model to continue training
    model.load_state_dict(torch.load('models_4/model_epoch_10.pth'))

    train(model, train_loader, loss_fn_video_id, optimizer, device, epochs=10)  # Continue training for additional epochs

    # Save the new model state
    if not os.path.exists('models_4'):
        os.makedirs('models_4')
    torch.save(model.state_dict(), 'models_4/model_epoch_20.pth')

if __name__ == "__main__":
    main()
