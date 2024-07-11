import os
import re
import json
import numpy as np

def parse_srt_file(srt_file_path):
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    segments = re.findall(r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', content, re.DOTALL)
    texts = []
    for start_time, end_time, text in segments:
        texts.append((start_time, end_time, text.replace('\n', ' ')))
    return texts

def time_to_seconds(time_str):
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def merge_segments(segments, max_length=512):
    merged_segments = []
    current_text = ""
    current_start = segments[0][0]
    current_end = segments[0][1]

    for start_time, end_time, text in segments:
        if len(current_text) + len(text) <= max_length:
            current_text += " " + text
            current_end = end_time
        else:
            merged_segments.append((current_start, current_end, current_text.strip()))
            current_text = text
            current_start = start_time
            current_end = end_time

    merged_segments.append((current_start, current_end, current_text.strip()))
    return merged_segments

def preprocess_srt_data(srt_dir):
    data = []
    video_ids = []
    for srt_file in os.listdir(srt_dir):
        if srt_file.endswith('.srt'):
            video_id = srt_file.split('.')[0]
            srt_file_path = os.path.join(srt_dir, srt_file)
            segments = parse_srt_file(srt_file_path)
            merged_segments = merge_segments(segments)
            for start_time, end_time, text in merged_segments:
                data.append({
                    'video_id': video_id,
                    'start_time': time_to_seconds(start_time),
                    'end_time': time_to_seconds(end_time),
                    'text': text
                })
            video_ids.append(video_id)
    return data, video_ids

if __name__ == "__main__":
    srt_dir = './processed'  # Update this path to your SRT files directory
    data, video_ids = preprocess_srt_data(srt_dir)

    with open('preprocessed_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open('video_ids.json', 'w', encoding='utf-8') as f:
        json.dump(video_ids, f, ensure_ascii=False, indent=4)

    print("Preprocessing complete. Data saved to 'preprocessed_data.json' and 'video_ids.json'.")
