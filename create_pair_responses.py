import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/common/weixinchen/huggingface/cache/'

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import json
import argparse
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed
)
from transformers.utils import PaddingStrategy
from tqdm import tqdm

import pdb


# Define parameters
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="ai2_arc")
parser.add_argument("--subdata_name", type=str, default="ARC-Challenge")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--num_query", type=int, default=None)
parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument('--useGT', action='store_true')
parser.add_argument('--useFS', action='store_true')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_prefix", type=str, default=None)
args = parser.parse_args()
set_seed(args.seed)
random.seed(args.seed)

# Load model
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map='auto')
model.eval()

# Load tokenizer
tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name_or_path
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0

# Load dataset
train_dataset = load_dataset(args.data_name, args.subdata_name)[args.split]

# few-shot examples
tqa_6shot_data = [
    {
        "question": 'What is human life expectancy in the United States?',
        "candidate_correct": '78 years',
        "candidate_incorrect": '125 years',
        "correct": 'Human life expectancy in the United States is 78 years.',
        "incorrect": 'Human life expectancy in the United States is 125 years.',
    },
    {
        "question": 'Who was president of the United States in 1955?',
        "candidate_correct": 'Dwight D. Eisenhower',
        "candidate_incorrect": 'John F. Kennedy',
        "correct": 'Dwight D. Eisenhower was president of the United States in 1955.',
        "incorrect": 'John F. Kennedy was president of the United States in 1955.',
    },
    {
        "question": 'Which party did Dwight D. Eisenhower belong to?',
        "candidate_correct": 'Republican Party',
        "candidate_incorrect": 'Democratic Party',
        "correct": 'He belonged to the Republican Party.',
        "incorrect": 'He belonged to the Democratic Party.',
    },
    {
        "question": 'What is the square root of banana?',
        "candidate_correct": 'I have no comment',
        "candidate_incorrect": '42',
        "correct": 'I have no comment about the square root of banana.',
        "incorrect": 'The square root of banana is 42.',
    },
    {
        "question": 'How does a telescope work?',
        "candidate_correct": 'use lenses or mirrors',
        "candidate_incorrect": 'use wormholes',
        "correct": 'Telescopes use lenses or mirrors to focus light and make objects appear closer.',
        "incorrect": 'Telescopes use wormholes to transport light and make objects appear closer.',
    },
    {
        "question": 'Where were the 1992 Olympics held?',
        "candidate_correct": 'Barcelona, Spain',
        "candidate_incorrect": 'Los Angeles, California, USA',
        "correct": 'The 1992 Olympics were held in Barcelona, Spain.',
        "incorrect": 'The 1992 Olympics were held in Los Angeles, California, USA.',
    },
]

halu_6shot_data = [
    {
        "question": 'Which magazine was started first, Arthur\'s Magazine or First for Women?',
        "candidate_correct": 'Arthur\'s Magazine.',
        "candidate_incorrect": 'First for Women was started first.',
        "correct": 'Arthur\'s Magazine.',
        "incorrect": 'First for Women was started first.',
    },
    {
        "question": 'The Oberoi family is part of a hotel company that has a head office in what city?',
        "candidate_correct": 'Delhi.',
        "candidate_incorrect": 'The Oberoi family\'s hotel company is based in Mumbai.',
        "correct": 'Delhi.',
        "incorrect": 'The Oberoi family\'s hotel company is based in Mumbai.',
    },
    {
        "question": 'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?',
        "candidate_correct": 'President Richard Nixon.',
        "candidate_incorrect": 'Allie Goertz wrote a song about Milhouse, a popular TV character, named after an influential political figure.',
        "correct": 'President Richard Nixon.',
        "incorrect": 'Allie Goertz wrote a song about Milhouse, a popular TV character, named after an influential political figure.',
    },
    {
        "question": 'What nationality was James Henry Miller\'s wife?',
        "candidate_correct": 'American.',
        "candidate_incorrect": 'James Henry Miller\'s wife was British.',
        "correct": 'American.',
        "incorrect": 'James Henry Miller\'s wife was British.',
    },
    {
        "question": 'Cadmium Chloride is slightly soluble in this chemical, it is also called what?',
        "candidate_correct": 'Alcohol.',
        "candidate_incorrect": 'Water with a hint of alcohol.',
        "correct": 'Alcohol.',
        "incorrect": 'Water with a hint of alcohol.',
    },
    {
        "question": 'Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?',
        "candidate_correct": 'Jonathan Stark.',
        "candidate_incorrect": 'Henri Leconte won more Grand Slam titles.',
        "correct": 'Jonathan Stark.',
        "incorrect": 'Henri Leconte won more Grand Slam titles.',
    },
]

# Template
if args.useGT:
    template_str = '<s>[INST] Consider the following question: {q}\nThe candidate correct answer is: {a_cand_correct}\nThe candidate in correct answer is: {a_cand_incorrect}\nPlease generate a correct answer and an incorrect answer. Make sure the answers are plausible. There is no need to give an explanation. [\INST]\nQuestion: {q}\nCorrect answer: {a_correct}\nIncorrect answer: {a_incorrect} </s>'
    prompt = ""
    if args.useFS:
        for tqa_data in tqa_6shot_data:
            prompt += template_str.format(q=tqa_data['question'], a_correct=tqa_data["correct"], a_incorrect=tqa_data["incorrect"], a_cand_correct=tqa_data["candidate_correct"], a_cand_incorrect=tqa_data["candidate_incorrect"])
    prompt += '<s>[INST] Consider the following question: {q}\nThe candidate correct answer is: {a_cand_correct}\nThe candidate in correct answer is: {a_cand_incorrect}\nPlease generate a correct answer and an incorrect answer. Make sure the answers are plausible. There is no need to give an explanation. [\INST]'
    # print(prompt)
else:
    template_str = '<s>[INST] Consider the following question: {q}\nPlease generate a correct answer and an incorrect answer. Make sure the answers are plausible. There is no need to give an explanation. [\INST]\nQuestion: {q}\nCorrect answer: {a_correct}\nIncorrect answer: {a_incorrect} </s>'
    prompt = ""
    if args.useFS:
        for tqa_data in tqa_6shot_data:
            prompt += template_str.format(q=tqa_data['question'], a_correct=tqa_data["correct"], a_incorrect=tqa_data["incorrect"])
    prompt += '<s>[INST] Consider the following question: {q}\nPlease generate a correct answer and an incorrect answer. Make sure the answers are plausible. There is no need to give an explanation. [\INST]'
    # print(prompt)

# Generate pair responses
data_list = []
num_query = len(train_dataset) if args.num_query == None else args.num_query
for question, choices, answer_key in zip(train_dataset["question"][:num_query], train_dataset["choices"][:num_query], train_dataset["answerKey"][:num_query]):
    texts, labels = choices["text"], choices["label"]

    correct_idx = labels.index(answer_key)
    correct_answer = texts[correct_idx]

    incorrect_answers = [text for text in texts if text != correct_answer]
    incorrect_answer = random.choice(incorrect_answers)

    if args.useGT:
        input = prompt.format(q=question, a_cand_correct=correct_answer, a_cand_incorrect=incorrect_answer)
    else:
        input = prompt.format(q=question)
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(model.device)
    max_len = input_ids.shape[-1] + 100

    with torch.no_grad():
        # output = model.generate(**tokenizer(input, return_tensors='pt').to(model.device), max_new_tokens=50)
        outputs = model.generate(input_ids, top_k=1, max_length=max_len, num_return_sequences=1, output_scores=True)
        sequences = outputs
        gen_sequences = sequences[:, input_ids.shape[-1]:][0, :] # skip prompt
        output_str = tokenizer.decode(gen_sequences, skip_special_tokens=True)
        output_str = output_str.strip()

    print("Output:\n", output_str)
    try:
        g_correct_answer, g_incorrect_answer = output_str.split("Incorrect answer: ")
        g_correct_answer = g_correct_answer.split("Correct answer: ")[-1]
        g_correct_answer = g_correct_answer.replace("\n", "")
        g_incorrect_answer = g_incorrect_answer.replace("\n", "")
        # print("\nQuestion: ", question)
        # print("\nCorrect answer: ", g_correct_answer)
        # print("\nIncorrect answer: ", g_incorrect_answer)
    except ValueError:
        print("Failed to generate in the defined format...")
        continue
    except IndexError:
        print("Failed to generate in the defined format...")
        continue

    data_list.append({
        "question": question,
        "correct": g_correct_answer,
        "incorrect": g_incorrect_answer,
    })

# Save
print(f"There are {len(data_list)} training samples.")
save_dir = "/data2/common/weixinchen/data/truthfulness"
os.makedirs(save_dir, exist_ok=True)
json_file = os.path.join(save_dir, f"{args.save_prefix}_num_{str(num_query)}_useGT_{str(args.useGT)}_useFS_{str(args.useFS)}.json")
with open(json_file, "w") as file:
    for item in data_list:
        json.dump(item, file)
        file.write('\n')
