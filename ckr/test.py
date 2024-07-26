import json
import os
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math

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


def find_top_n_similar_subtitles(prediction, subtitles, top_n=3):
    """根据预测文本在字幕中找出相似度评分最高的N个部分"""
    subtitle_texts = [sub[2] for sub in subtitles]

    if not subtitle_texts:
        print("警告: 字幕文本列表为空，无法计算相似度!")
        return []

    subtitle_texts.append(prediction)  # 包含预测文本

    # 确保所有文本都是字符串
    subtitle_texts = [str(text) for text in subtitle_texts]

    # 使用 TF-IDF 向量化器，并增加 ngram_range
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(subtitle_texts)

    if tfidf_matrix.shape[0] <= 1:
        print("警告: TF-IDF 矩阵样本数量不足!")
        return []

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # 获取相似度评分及其对应的索引
    similarity_scores = cosine_sim[0]
    indexed_scores = list(enumerate(similarity_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)  # 按相似度排序

    # 选择相似度评分最高的N个字幕片段
    top_n_indices = [index for index, score in indexed_scores[:top_n]]
    top_n_subtitles = [subtitles[index] for index in top_n_indices]

    return top_n_subtitles


# 定义用于T5输入格式的Dataset类
class QADataset(Dataset):
    def __init__(self, tokenizer, data, subtitle_dir, max_length=1024):
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
        input_ids = self.tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=self.max_length,
                                          return_tensors='pt')

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


def format_time(seconds):
    """将秒数转换为分钟:秒格式"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def count_sentences(text):
    """计算文本中的句子数量"""
    # 定义句子的分隔符
    sentence_endings = re.compile(r'[,.!?]')
    return len(sentence_endings.findall(text))


def calculate_iou(true_start, true_end, pred_start, pred_end):
    """计算真实值和预测值的 IOU"""
    start_max = max(true_start, pred_start)
    end_min = min(true_end, pred_end)
    intersection = max(0, end_min - start_max)
    union = max(true_end, pred_end) - min(true_start, pred_start)
    return intersection / union if union > 0 else 0


# 用于跟踪 IOU 的总和和计数
iou_total = 0
iou_count = 0


def process_prediction(current_question, current_prediction, current_subtitles):
    if not current_question:
        return

    merged_prediction = " ".join(current_prediction).replace(' .', '.').strip()

    if not merged_prediction:
        print(f"警告: 预测结果为空, 跳过处理问题: {current_question}")
        return

    # 计算预测结果中的句子数量
    num_sentences = count_sentences(merged_prediction)

    example = next(item for item in train_data if item['question'] == current_question)
    two_thirds = (1 / 2) * num_sentences
    result = math.ceil(two_thirds)

    # 找出与预测答案最相似的前三个字幕片段
    top_n_subtitles = find_top_n_similar_subtitles(merged_prediction, current_subtitles, top_n=result or 3)  # 至少选取3个

    if top_n_subtitles:
        # 排序三句字幕片段的时间戳
        top_n_subtitles.sort(key=lambda sub: sub[0])  # 按起始时间排序

        # 合并内容
        merged_content = ''
        try:
            # 确保 sub[2] 是字幕内容字符串
            merged_content = ' '.join(sub[2] for sub in top_n_subtitles if isinstance(sub[2], str))
        except TypeError as e:
            print(f"Error: The elements in top_n_subtitles are not as expected: {e}")
            print("top_n_subtitles:", top_n_subtitles)
            merged_content = 'Error processing subtitles.'

        # 确定最终的起始和终止时间
        start_time = float(top_n_subtitles[0][0])
        end_time = float(top_n_subtitles[-1][1])

        # 提取时间并格式化
        true_start = example['answer_start_second']
        true_end = example['answer_end_second']
        formatted_true_start = format_time(true_start)
        formatted_true_end = format_time(true_end)
        formatted_pred_start = format_time(start_time)
        formatted_pred_end = format_time(end_time)

        # 计算 IOU
        iou = calculate_iou(true_start, true_end, start_time, end_time)

        # 更新 IOU 总和和计数
        global iou_total, iou_count
        iou_total += float(iou)
        iou_count += 1

        # 计算当前平均 IOU
        average_iou = iou_total / iou_count if iou_count > 0 else 0

        print(f"问题: {current_question}")
        print(f"真实答案: {formatted_true_start} - {formatted_true_end}")
        print(f"预测答案: {merged_prediction}")
        print(f"预测结果中的句子数量: {num_sentences} {result}")
        print(f"最相似的字幕时间: {formatted_pred_start} - {formatted_pred_end}")
        print(f"IOU: {float(iou):.2f}")
        print(f"当前平均 IOU: {average_iou:.2f}")
        print("-" * 50)


# 处理每个批次的数据
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    question = batch['question'][0]
    subtitles = batch['subtitles']

    with torch.no_grad():
        preds = model.generate(input_ids,
                               num_beams=5,  # 使用 beam search 来增加生成的多样性
                               no_repeat_ngram_size=2,  # 防止重复的 n-gram
                               max_length=512,  # 生成文本的最大长度
                               min_length=100,  # 生成文本的最小长度
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
        # 处理之前的问题的预测结果
        process_prediction(current_question, current_prediction, current_subtitles)

        # 重置当前问题和预测结果
        current_question = question
        current_prediction = decoded_preds
        current_subtitles = subtitles

# 处理最后一个问题
process_prediction(current_question, current_prediction, current_subtitles)
