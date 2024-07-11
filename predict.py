# -*- coding = utf-8 -*-
# @Time : 2024/7/11 10:29
# @Author : 王加炜
# @File : predict.py
# @Software : PyCharm
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from datasets import Dataset
from train import MultiTaskModel

def load_label_encoder(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def find_nearest_time(pred_time, data, video_id):
    times = [item['start_time'] for item in data if item['video_id'] == video_id] + [item['end_time'] for item in data if item['video_id'] == video_id]
    times = np.array(times)
    nearest_time = times[np.abs(times - pred_time).argmin()]
    return nearest_time

def predict_and_evaluate(test_data_path, model_dir, tokenizer_name, label_encoder, data):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = MultiTaskModel.from_pretrained(model_dir, num_labels=len(label_encoder.classes_))

    predictions = []
    for item in test_data:
        question = item["question"]
        inputs = tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        model.eval()
        with torch.no_grad():
            logits, regressions, _ = model(**inputs)
            predicted_label = logits.argmax(-1).item()
            predicted_video_id = label_encoder.inverse_transform([predicted_label])[0]
            predicted_start, predicted_end = regressions[0].cpu().numpy()

        predicted_start = find_nearest_time(predicted_start, data, predicted_video_id)
        predicted_end = find_nearest_time(predicted_end, data, predicted_video_id)

        predictions.append({
            "sample_id": item["sample_id"],
            "actual_video_id": item["video_id"],
            "predicted_video_id": predicted_video_id,
            "actual_answer_start_second": item["answer_start_second"],
            "predicted_answer_start_second": predicted_start,
            "actual_answer_end_second": item["answer_end_second"],
            "predicted_answer_end_second": predicted_end
        })

    return predictions

def plot_comparison(df):
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(df['sample_id'], df['actual_answer_start_second'], label='Actual Start')
    ax[0].plot(df['sample_id'], df['predicted_answer_start_second'], label='Predicted Start')
    ax[0].set_title('Actual vs Predicted Start Time')
    ax[0].legend()

    ax[1].plot(df['sample_id'], df['actual_answer_end_second'], label='Actual End')
    ax[1].plot(df['sample_id'], df['predicted_answer_end_second'], label='Predicted End')
    ax[1].set_title('Actual vs Predicted End Time')
    ax[1].legend()

    ax[2].plot(df['sample_id'], df['actual_video_id'], label='Actual Video ID')
    ax[2].plot(df['sample_id'], df['predicted_video_id'], label='Predicted Video ID')
    ax[2].set_title('Actual vs Predicted Video ID')
    ax[2].legend()

    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.show()

def main():
    test_data_path = 'test.json'
    model_dir = './trained_model'
    tokenizer_name = 'bert-base-uncased'

    with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    label_encoder = load_label_encoder('label_encoder.pkl')
    predictions = predict_and_evaluate(test_data_path, model_dir, tokenizer_name, label_encoder, data)

    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Predictions saved to predictions.json")

    df = pd.DataFrame(predictions)
    plot_comparison(df)

    actual_video_ids = [item['actual_video_id'] for item in predictions]
    predicted_video_ids = [item['predicted_video_id'] for item in predictions]
    video_id_accuracy = accuracy_score(actual_video_ids, predicted_video_ids)
    print(f"Video ID Accuracy: {video_id_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
