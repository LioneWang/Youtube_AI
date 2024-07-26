import json
import os
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 定义文件路径和目录
train_file = 'train.json'
subtitle_dir = 'ckr2'  # 修改为实际字幕文件夹名
local_model_path = 'trained_model'  # 本地模型路径
weight_path = os.path.join(local_model_path, 't5_qa_model.pt')  # 模型权重保存路径

# 从JSON文件加载训练数据
with open(train_file, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# 解析SRT文件
def parse_srt(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 字幕文件 {file_path} 不存在!")
        return []

    with open(file_path, 'r', encoding='utf-8') as file:
        srt_data = file.read()

    pattern = re.compile(r'(\d+\.\d+),(\d+\.\d+)\n(.*?)\n', re.DOTALL)
    matches = pattern.findall(srt_data)

    subtitles = []
    for match in matches:
        start_time = float(match[0])
        end_time = float(match[1])
        text = match[2].strip()
        subtitles.append((start_time, end_time, text))

    return subtitles

def extract_subtitle_text(subtitles, start_time, end_time):
    """根据给定的开始和结束时间从字幕中提取文本"""
    text = []
    for sub in subtitles:
        if sub[0] >= start_time and sub[1] <= end_time:
            text.append(sub[2])
    return ' '.join(text)

def find_most_similar_subtitle(prediction, subtitles):
    """根据预测文本在字幕中找出最相似的部分"""
    subtitle_texts = [sub[2] for sub in subtitles]
    subtitle_texts.append(prediction)  # 包含预测文本

    # 确保所有文本都是字符串
    subtitle_texts = [str(text) for text in subtitle_texts]

    # 使用 TF-IDF 向量化器，并增加 ngram_range
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(subtitle_texts)

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # 找出最相似的字幕段落
    most_similar_indices = np.argsort(cosine_sim[0])[-3:]  # 选择3个最相似的段落
    most_similar_subtitles = [subtitles[idx] for idx in most_similar_indices]

    # 合并这些相似段落
    combined_start_time = min(sub[0] for sub in most_similar_subtitles)
    combined_end_time = max(sub[1] for sub in most_similar_subtitles)
    combined_text = ' '.join(sub[2] for sub in most_similar_subtitles)

    return (combined_start_time, combined_end_time, combined_text)

# 定义用于T5输入格式的Dataset类
class QADataset(Dataset):
    def __init__(self, tokenizer, data, subtitle_dir, max_length=1024):  # 增加 max_length
        self.data = data
        self.tokenizer = tokenizer
        self.subtitle_dir = subtitle_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['question']
        video_id = example['video_id']
        srt_path = os.path.join(self.subtitle_dir, f'{video_id}.srt')

        # 检查字幕文件是否存在
        if not os.path.exists(srt_path):
            print(f"警告: 字幕文件 {srt_path} 不存在!")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'question': question,
                'subtitles': []  # 空的字幕列表
            }

        # 解析SRT文件
        subtitles = parse_srt(srt_path)
        subtitle_text = extract_subtitle_text(subtitles, example['answer_start_second'], example['answer_end_second'])

        # 格式化为T5模型输入
        input_text = f"question: {question} context: {subtitle_text}"

        # 对输入进行编码
        input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': input_ids.squeeze(),
            'question': question,
            'subtitles': subtitles  # 添加字幕信息
        }

# 初始化tokenizer和model
tokenizer = T5Tokenizer.from_pretrained(local_model_path)
model = T5ForConditionalGeneration.from_pretrained(local_model_path)

# 加载训练好的模型权重
model.load_state_dict(torch.load(weight_path))
model.eval()

# 准备数据集和数据加载器
test_dataset = QADataset(tokenizer, train_data, subtitle_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

current_question = None
current_prediction = []
current_subtitles = []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    question = batch['question'][0]
    subtitles = batch['subtitles']

    with torch.no_grad():
        preds = model.generate(input_ids,
                               num_beams=5,  # 使用 beam search 来增加生成的多样性
                               no_repeat_ngram_size=2,  # 防止重复的 n-gram
                               max_length=512,  # 生成文本的最大长度
                               min_length=50,  # 生成文本的最小长度
                               length_penalty=1.0,  # 长度惩罚
                               early_stopping=True)  # 提前停止

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    if current_question is None:
        current_question = question
        current_subtitles = subtitles

    if question == current_question:
        # 追加预测结果
        current_prediction.extend(decoded_preds)
    else:
        # 输出之前的问题的结果
        merged_prediction = " ".join(current_prediction).replace(' .', '.').strip()
        example = next(item for item in train_data if item['question'] == current_question)

        # 找出与预测答案最相似的字幕段落
        most_similar_subtitle = find_most_similar_subtitle(merged_prediction, current_subtitles)

        print(f"问题: {current_question}")
        print(f"真实答案: {example['answer_start']} - {example['answer_end']}")
        print(f"预测答案: {merged_prediction}")
        print(f"最相似的字幕时间: {most_similar_subtitle[0]} - {most_similar_subtitle[1]}")
        print(f"最相似的字幕内容: {most_similar_subtitle[2]}")
        print("-" * 50)

        # 重置当前问题和预测结果
        current_question = question
        current_prediction = decoded_preds
        current_subtitles = subtitles

# 处理最后一个问题
if current_question is not None:
    merged_prediction = " ".join(current_prediction).replace(' .', '.').strip()
    example = next(item for item in train_data if item['question'] == current_question)

    # 找出与预测答案最相似的字幕段落
    most_similar_subtitle = find_most_similar_subtitle(merged_prediction, current_subtitles)

    print(f"问题: {current_question}")
    print(f"真实答案: {example['answer_start']} - {example['answer_end']}")
    print(f"预测答案: {merged_prediction}")
    print(f"最相似的字幕时间: {most_similar_subtitle[0]} - {most_similar_subtitle[1]}")
    print(f"最相似的字幕内容: {most_similar_subtitle[2]}")
    print("-" * 50)
