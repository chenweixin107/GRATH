import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import argparse
import torch
from datasets import Dataset, load_dataset
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument("--new_data_path", type=str, default=None)
parser.add_argument("--ori_data_path", type=str, default=None)
args = parser.parse_args()

new_dataset = load_dataset("json", data_files={'train': args.new_data_path})['train']
ori_dataset = load_dataset("json", data_files={'train': args.ori_data_path})['train']
dataset = []

for ori_question, ori_correct, ori_incorrect in zip(ori_dataset["question"], ori_dataset["correct"], ori_dataset["incorrect"]):
    try:
        question_idx = new_dataset["question"].index(ori_question)
        new_incorrect = new_dataset["incorrect"][question_idx]
        new_correct = new_dataset["correct"][question_idx]
    except ValueError:
        new_incorrect = ori_incorrect
        new_correct = ori_correct
    # print(f"question:\n{ori_question}")
    # print(f"ori_correct:\n{ori_correct}")
    # print(f"new_correct:\n{new_correct}")
    # print(f"ori_incorrect\n{ori_incorrect}")
    # print(f"new_incorrect:\n{new_incorrect}")
    # print("\n")
    dataset.append({
        "question": ori_question,
        "correct": new_correct,
        "incorrect": ori_incorrect,
    })

print(f"There are {len(dataset)} samples.")
json_file = args.new_data_path.split(".json")[0] + "_combined.json"
with open(json_file, "w") as file:
    for item in dataset:
        json.dump(item, file)
        file.write('\n')
