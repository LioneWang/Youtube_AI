import re
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('trained_model')

# 加载预训练的BERT模型权重
model_path = '/mnt/bert/trained_model/bert_qa_model.pt'
model = BertForQuestionAnswering.from_pretrained('trained_model')
model.load_state_dict(torch.load(model_path))

# 解析SRT文件
def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_data = file.read()

    pattern = re.compile(r'(\d+\.\d+),(\d+\.\d+)\s+(.*?)\n', re.DOTALL)
    matches = pattern.findall(srt_data)

    subtitles = []
    for match in matches:
        start_time = float(match[0])
        end_time = float(match[1])
        text = match[2].strip()
        subtitles.append((start_time, end_time, text))

    return subtitles

# 加载SRT文件
srt_path = 'lbPbM8018CE.srt'
subtitles = parse_srt(srt_path)

# 将字幕文本拼接成一个字符串
text = " ".join([sub[2] for sub in subtitles])

# 提问的问题
question = "How to perform epley maneuver for vertigo?"

# 将问题和上下文进行tokenize
inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
input_ids = inputs['input_ids'].tolist()[0]

# 模型预测
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits

# 获取答案开始和结束位置
answer_start = torch.argmax(answer_start_scores).item()
answer_end = torch.argmax(answer_end_scores).item() + 1

# 确保答案的索引在有效范围内
if answer_start >= len(subtitles):
    answer_start = len(subtitles) - 1
if answer_end >= len(subtitles):
    answer_end = len(subtitles) - 1

# 精确计算答案时间段
answer_start_second = subtitles[answer_start][0]
answer_end_second = subtitles[answer_end][1]

# 计算相关得分
relevant_score = (answer_start_scores[0][answer_start] + answer_end_scores[0][answer_end - 1]).item()

# 输出结果
result = {
    "video_id": "lbPbM8018CE",
    "relevant_score": relevant_score,
    "answer_start_second": answer_start_second,
    "answer_end_second": answer_end_second
}

print(result)
