import json
import torch
from torch import nn
from transformers import BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

class MultiTaskModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, start_end_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        regressions = self.regressor(pooled_output)

        loss = None
        if labels is not None and start_end_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            regression_loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels) + regression_loss_fct(regressions, start_end_labels)
        return logits, regressions, loss

def encode_labels(data, video_ids):
    le = LabelEncoder()
    le.fit(video_ids)
    for item in data:
        item['video_label'] = le.transform([item['video_id']])[0]
    return data, le

def create_datasets(data):
    texts = [item['text'] for item in data]
    video_labels = [item['video_label'] for item in data]
    start_times = [item['start_time'] for item in data]
    end_times = [item['end_time'] for item in data]

    dataset = Dataset.from_dict({
        'text': texts,
        'video_label': video_labels,
        'start_time': start_times,
        'end_time': end_times
    })
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def train_model(train_dataset, num_labels, model_dir, tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = MultiTaskModel(tokenizer_name, num_labels)

    def compute_metrics(pred):
        labels = pred.label_ids['video_label']
        preds = pred.predictions[0].argmax(-1)
        video_accuracy = accuracy_score(labels, preds)

        start_end_labels = pred.label_ids['start_end_labels']
        start_end_preds = pred.predictions[1]
        mse = mean_squared_error(start_end_labels, start_end_preds)

        return {
            'video_accuracy': video_accuracy,
            'start_end_mse': mse
        }

    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'video_label', 'start_time', 'end_time'])

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(model_dir)

if __name__ == "__main__":
    with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('video_ids.json', 'r', encoding='utf-8') as f:
        video_ids = json.load(f)

    data, label_encoder = encode_labels(data, video_ids)
    train_dataset = create_datasets(data)

    model_dir = './trained_model'
    tokenizer_name = 'bert-base-uncased'
    train_model(train_dataset, num_labels=len(label_encoder.classes_), model_dir=model_dir, tokenizer_name=tokenizer_name)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Training complete. Model and tokenizer saved to 'trained_model'.")
