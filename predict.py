import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import json


class QuestionToVideoModel(nn.Module):
    def __init__(self, num_videos):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.video_id_classifier = nn.Linear(self.bert.config.hidden_size, num_videos)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        video_id_logits = self.video_id_classifier(pooled_output)
        return video_id_logits


def load_video_mapping(file_path):
    with open(file_path, 'r') as file:
        mapping = json.load(file)
    return mapping


def predict_question(model, tokenizer, question, device, video_id_to_idx):
    encoding = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        video_id_logits = model(input_ids, attention_mask)

    predicted_video_id_index = torch.argmax(video_id_logits, dim=1).item()
    predicted_video_id = list(video_id_to_idx.keys())[list(video_id_to_idx.values()).index(predicted_video_id_index)]

    return predicted_video_id


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mapping = load_video_mapping('video_id_mapping.json')
    num_videos = len(mapping)  # 获取映射的长度即视频数量
    model = QuestionToVideoModel(num_videos=num_videos).to(device)
    model.load_state_dict(torch.load('models_4/model_epoch_10.pth'))  # 加载模型权重
    question = input("Please enter your question: ")
    video_id = predict_question(model, tokenizer, question, device, mapping)
    print(f"Predicted Video ID: {video_id}")
