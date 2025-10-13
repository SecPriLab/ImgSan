import io
import os
import re
import csv
import time
import json
import logging
import base64
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "PrivacyVQA2K"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
OUTPUT_DIR = PROJECT_ROOT / "ImgSan" / "sanitized_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gpt-4o"

def encode_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

client_openai = OpenAI(
    api_key="",
    base_url=""
)

def get_answer_from_model(client, question, answer, base64_image, model):
    while True:
        try:
            messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant. Keep your answer to 10 words or less."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            completion = client.chat.completions.create(
                model = model,
                temperature = 0,
                messages = messages
            )
            output_1 = completion.choices[0].message.content
            answer_parts = answer.split(" ")
            if len(answer_parts) <= 1:
                message_content_1 = output_1.lower()
                answer = answer.lower()
                if answer in message_content_1:
                    return True, output_1
                else:
                    return False, output_1
            messages.append({"role": "assistant", "content": f"My answer is '{completion.choices[0].message.content}'."})
            messages.append({
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": f"The correct answer is '{answer}'. Do you think your answer is close to the correct answer? If it is close, please reply with 'Yes'; if it is not, please reply with 'No'."
                    }
                    ]
                })
            completion = client.chat.completions.create(
                model = model,
                temperature = 0,
                messages=messages
                )
            output_2 = completion.choices[0].message.content
            message_content_2 = output_2.lower()

            if 'yes' in message_content_2:
                return True, [output_1, output_2]
            else:
                return False, [output_1, output_2]

        except Exception as e:
            error_message = str(e)
            print("----------------------------")
            return None, error_message

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)


client = client_openai
all_utility = []
for item in tqdm(data, desc="Evaluating Utility Metrics"):
    time.sleep(1)
    question = item["qas_en"][0][0]
    answer = item["qas_en"][0][1]
    processed_path = os.path.join(OUTPUT_DIR, item['image'])
    if not os.path.exists(processed_path):
        continue
    base64_image = encode_image(processed_path)
    result, info = get_answer_from_model(client, question, answer, base64_image, MODEL)
    if result == True:
        all_utility.append(1)
    else:
        all_utility.append(0)

avg_acc = np.mean(all_utility) if all_utility else 0

average_metrics = {
    "average_accuracy": avg_acc,
}

output_summary_path = PROJECT_ROOT / "ImgSan" / "evaluation" / "utility_results.json"
output_summary_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_summary_path, "w", encoding="utf-8") as f:
    json.dump(average_metrics, f, indent=4)