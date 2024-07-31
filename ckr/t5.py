import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 初始化T5 tokenizer和model
tokenizer = T5Tokenizer.from_pretrained('D:/vs_code/t5/t-5')
model_path = 'trained_model/t5_qa_model.pt'
model = T5ForConditionalGeneration.from_pretrained('D:/vs_code/t5/t-5')
model.load_state_dict(torch.load(model_path))


# 函数：解析自定义SRT文件
def parse_custom_srt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            srt_data = file.read()
    except Exception as e:
        print(f"Error reading SRT file {file_path}: {e}")
        return []

    # 匹配自定义格式
    pattern = re.compile(r'\{\s*(\d+\.\d+),\s*(\d+\.\d+)\s*\}\s*(.*?)(?=\{|\Z)', re.DOTALL)
    matches = pattern.findall(srt_data)

    if not matches:
        print(f"No matches found in custom SRT file {file_path}")
        return []

    subtitles = []
    for match in matches:
        start_time = float(match[0])
        end_time = float(match[1])
        text = match[2].strip()
        subtitles.append((start_time, end_time, text))

    if not subtitles:
        print(f"No subtitles extracted from custom SRT file {file_path}")

    return subtitles


def split_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s]


# 加载JSON文件
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

max_length = 512
stride = 256

for sample in train_data:
    video_id = sample['video_id']
    srt_path = os.path.join('ckr2', f'{video_id}.srt')

    if not os.path.exists(srt_path):
        print(f"SRT file does not exist at path: {srt_path}")
        continue

    subtitles = parse_custom_srt(srt_path)
    if not subtitles:
        print(f"No subtitles found in SRT file for video_id: {video_id}")
        continue

    subtitle_texts = [sub[2] for sub in subtitles]
    if not subtitle_texts:
        print(f"No subtitle texts extracted for video_id: {video_id}")
        continue

    text = " ".join(subtitle_texts)

    question = sample['question']

    answer_start_minute, answer_start_second = map(int, sample['answer_start'].split(':'))
    answer_end_minute, answer_end_second = map(int, sample['answer_end'].split(':'))

    answer_start_seconds = answer_start_minute * 60 + answer_start_second
    answer_end_seconds = answer_end_minute * 60 + answer_end_second

    input_text = f"question: {question} context: {text}"

    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            min_length=50,
            num_beams=8,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True
        )

    answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Answer: {answer_text}")

    sentences = split_sentences(answer_text)
    print(f"Split Sentences: {sentences}")

    if not sentences:
        print("No sentences generated to compare.")
        continue

    vectorizer = TfidfVectorizer().fit(subtitle_texts + sentences)
    tfidf_matrix = vectorizer.transform(subtitle_texts + sentences)

    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        print("TF-IDF matrix is empty.")
        continue

    subtitle_vectors = tfidf_matrix[:len(subtitle_texts)]
    sentence_vectors = tfidf_matrix[len(subtitle_texts):]

    best_match = None
    best_match_score = 0.0

    for i, sentence_vector in enumerate(sentence_vectors):
        if np.count_nonzero(sentence_vector) == 0:
            continue

        similarities = cosine_similarity(sentence_vector, subtitle_vectors)

        if similarities.size == 0:
            continue

        max_similarity = np.max(similarities)

        if max_similarity > best_match_score:
            best_match_score = max_similarity
            best_match = (sentences[i], subtitle_texts[np.argmax(similarities)], np.argmax(similarities))

    if best_match:
        best_sentence, subtitle_text, best_subtitle_index = best_match
        predicted_start_second, predicted_end_second = subtitles[best_subtitle_index][0:2]
    else:
        best_sentence, predicted_start_second, predicted_end_second = "", 0, 0

    intersection = max(0, min(predicted_end_second, answer_end_seconds) - max(predicted_start_second,
                                                                              answer_start_seconds))
    union = max(predicted_end_second, answer_end_seconds) - min(predicted_start_second, answer_start_seconds)
    iou = intersection / union if union != 0 else 0

    result = {
        "video_id": video_id,
        "question": question,
        "iou_score": iou,
        "predicted_start_second": predicted_start_second,
        "predicted_end_second": predicted_end_second,
        "answer_start_seconds": answer_start_seconds,
        "answer_end_seconds": answer_end_seconds,
        "best_sentence": best_sentence
    }

    # 打印结果
    for key, value in result.items():
        print(f"{key}: {value}")
