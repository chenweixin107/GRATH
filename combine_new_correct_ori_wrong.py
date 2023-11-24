import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/common/weixinchen/huggingface/cache/'

from dataclasses import dataclass, field
from typing import Dict, Optional
import argparse
import torch
from datasets import Dataset, load_dataset
import random
import json

import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--new_data_path", type=str, default=None)
parser.add_argument("--ori_data_path", type=str, default=None)
args = parser.parse_args()

new_dataset = load_dataset("json", data_files={'train': args.new_data_path})['train']
ori_dataset = load_dataset("json", data_files={'train': args.ori_data_path})['train']
dataset = []

for new_question, new_correct, new_incorrect in zip(new_dataset["question"], new_dataset["correct"], new_dataset["incorrect"]):
    try:
        question_idx = ori_dataset["question"].index(new_question)
    except ValueError:
        continue
    ori_incorrect = ori_dataset["incorrect"][question_idx]
    ori_correct = ori_dataset["correct"][question_idx]
    print(f"question:\n{new_question}")
    print(f"new_correct:\n{new_correct}")
    print(f"ori_correct:\n{ori_correct}")
    print(f"new_incorrect\n{new_incorrect}")
    print(f"ori_incorrect:\n{ori_incorrect}")
    print("\n")
    dataset.append({
        "question": new_question,
        "correct": new_correct,
        "incorrect": ori_incorrect,
    })

print(f"There are {len(dataset)} samples.")
json_file = args.new_data_path.split(".json")[0] + "_combined.json"
with open(json_file, "w") as file:
    for item in dataset:
        json.dump(item, file)
        file.write('\n')
