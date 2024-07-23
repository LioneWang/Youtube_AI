import json
import re
from collections import defaultdict


def update_timestamps(sentences, start_time, end_time):
    sentence_lengths = [len(sentence) for sentence in sentences]
    total_length = sum(sentence_lengths)
    timestamps = []
    last_start = start_time

    for length in sentence_lengths:
        segment_duration = (end_time - last_start) * (length / total_length)
        new_end_time = last_start + segment_duration
        timestamps.append((last_start, new_end_time))
        last_start = new_end_time

    return timestamps


def process_transcripts(data):
    processed_data = []
    errors = []
    video_processed = set()  # Track processed video IDs

    for entry in data:
        sample_id = entry["sample_id"]
        video_id = entry["video_id"]

        if video_id in video_processed:
            continue  # Skip this entry if the video has been processed

        video_processed.add(video_id)  # Mark this video as processed

        transcript = entry["transcript"]
        current_text = ''
        current_start = None

        for segment in transcript:
            start_time = segment['timestamp'][0]
            end_time = segment['timestamp'][1]
            text = segment['text']

            if start_time is None or end_time is None:
                errors.append({
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "text": text,
                    "reason": "Timestamp is empty"
                })
                continue  # Skip processing this segment
            if current_start is None:
                current_start = start_time  # Initialize current_start if it's None
            current_text += ' ' + text
            sentences = re.split(r'(?<=[.!?]) +', current_text.strip())

            # Update timestamps for each sentence if we reach a full stop
            if len(sentences) > 1:
                timestamps = update_timestamps(sentences[:-1], current_start, end_time)
                for sentence, (start, end) in zip(sentences[:-1], timestamps):
                    if not sentence.strip():
                        continue  # Ignore empty sentences
                    processed_data.append({
                        "sample_id": sample_id,
                        "video_id": video_id,
                        "start_time": start,
                        "end_time": end,
                        "text": sentence.strip()
                    })
                current_text = sentences[-1]  # Start accumulating text again from the last partial sentence
                current_start = timestamps[-1][1] if timestamps else current_start
            else:
                current_start = start_time  # Update start time if no sentence end is found

    return processed_data, errors


def main():
    with open('train_srt.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data, errors = process_transcripts(data)

    with open('processed_train_data.json', 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, indent=4)

    with open('train_errors.json', 'w', encoding='utf-8') as file:
        json.dump(errors, file, indent=4)

    print("Processing complete. Data and errors saved.")


if __name__ == "__main__":
    main()
