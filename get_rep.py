import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/common/weixinchen/huggingface/cache/'

import tqdm
import argparse
import torch
import torch.nn.functional as F
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from datasets import load_dataset

import sys
sys.path.append('../representation-engineering')
from repe import repe_pipeline_registry
repe_pipeline_registry()
from repe_eval.tasks import task_dataset

import pdb

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--model_type", type=str, default="llama2")
parser.add_argument("--data_name", type=str, default=None)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--save_name", type=str, default=None)
args = parser.parse_args()

# Load dataset
if args.data_path == None:
    train_dataset = load_dataset(args.data_name, split="train")
else:
    train_dataset = load_dataset("json", data_files={"train": args.data_path})["train"]
# base_train_dataset = load_dataset("json", data_files={"train": "/data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json"})["train"]

# Load model
print("model_name_or_path: ", args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto")
model.eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1

# Load truthfulness direction
rep_kwargs = {
    "task": "tqa",
    "direction_method": "pca",
    "rep_token": -1,
    "max_length": 2048,
    "ntrain": 0,
    "n_components": 1,
    "n_difference": 1,
    "batch_size": 8,
    "seed": 2,
}
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
chosen_layer = -1

# rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
# tqa_data = task_dataset(rep_kwargs["task"])(ntrain=rep_kwargs["ntrain"], seed=rep_kwargs["seed"])
# direction_finder_kwargs = {"n_components": rep_kwargs["n_components"]}
#
# rep_reader = rep_pipeline.get_directions(
#     tqa_data['train']['data'],
#     rep_token=rep_kwargs["rep_token"],
#     hidden_layers=hidden_layers,
#     n_difference=rep_kwargs["n_difference"],
#     train_labels=tqa_data['train']['labels'],
#     direction_method=rep_kwargs["direction_method"],
#     direction_finder_kwargs=direction_finder_kwargs,
#     batch_size=rep_kwargs["batch_size"],
#     max_length=rep_kwargs["max_length"],
#     padding="longest",
# )
#
# # np.save(os.path.join("/home/weixinchen/AdaptiveDPO/reading_vec", f"{rep_kwargs['task']}_reading_vec.npy"), rep_reader.directions)
# # np.save(os.path.join("/home/weixinchen/AdaptiveDPO/reading_vec", f"{rep_kwargs['task']}_reading_sign.npy"), rep_reader.direction_signs)
#
# truth_direction = rep_reader.directions[chosen_layer] * rep_reader.direction_signs[chosen_layer]
# truth_direction = truth_direction.squeeze() # (4096,)

# Project data to representation space
# if args.model_type == "llama2":
#     user_tag = "[INST]"
#     assistant_tag = "[/INST]"
# elif args.model_type == "zephyr":
#     user_tag = "<|user|>\n"
#     assistant_tag = "</s>\n<|assistant|>"
# else:
#     raise ValueError(f"There is no {model_type} model type.")
# template_str = '{user_tag} {scenario} {assistant_tag}'
prompt = "Consider the amount of truthfulness in the following answer:"
suffix = "The amount of truthfulness in the answer is "
template_str = '{prompt}\n{scenario}\n{suffix}'

save_list = []
distance_list = []

def project_onto_direction(a, b):
    # unit vector
    u = b / np.linalg.norm(b)

    # projection: uu^Ta
    uu_T = np.outer(u, u)
    result = np.dot(uu_T, a)

    return result

for question, correct_answer, incorrect_answer in zip(train_dataset["question"], train_dataset["correct"], train_dataset["incorrect"]):
    # correct_qa = template_str.format(user_tag=user_tag, assistant_tag=assistant_tag, scenario=question) + correct_answer
    correct_qa = correct_answer
    # correct_qa = template_str.format(prompt=prompt, suffix=suffix, scenario=f"Question: {question}\nAnswer: {correct_answer}")
    correct_tokens = tokenizer(correct_qa, truncation=True, return_tensors="pt")
    with torch.no_grad():
        correct_output = model(**correct_tokens, output_hidden_states=True)
    hidden_states = correct_output['hidden_states'][chosen_layer]
    correct_vec = hidden_states[:, rep_kwargs["rep_token"], :].squeeze() # torch.Size([4096])
    correct_vec = correct_vec.cpu().to(dtype=torch.float32).numpy()


    # question_idx = base_train_dataset["question"].index(question)
    # base_correct = base_train_dataset["correct"][question_idx]
    # base_incorrect = base_train_dataset["incorrect"][question_idx]
    # assert base_incorrect == incorrect_answer, "base_incorrect should be the same as incorrect_answer"
    # base_correct_qa = template_str.format(prompt=prompt, suffix=suffix, scenario=f"Question: {question}\nAnswer: {base_correct}")
    # base_correct_tokens = tokenizer(base_correct_qa, truncation=True, return_tensors="pt")
    # with torch.no_grad():
    #     base_correct_output = model(**base_correct_tokens, output_hidden_states=True)
    # hidden_states = base_correct_output['hidden_states'][chosen_layer]
    # base_correct_vec = hidden_states[:, rep_kwargs["rep_token"], :].squeeze()  # torch.Size([4096])
    # base_correct_vec = base_correct_vec.cpu().to(dtype=torch.float32).numpy()


    # incorrect_qa = template_str.format(user_tag=user_tag, assistant_tag=assistant_tag, scenario=question) + incorrect_answer
    incorrect_qa = incorrect_answer
    # incorrect_qa = template_str.format(prompt=prompt, suffix=suffix, scenario=f"Question: {question}\nAnswer: {incorrect_answer}")
    incorrect_tokens = tokenizer(incorrect_qa, truncation=True, return_tensors="pt")
    with torch.no_grad():
        incorrect_output = model(**incorrect_tokens, output_hidden_states=True)
    hidden_states = incorrect_output['hidden_states'][chosen_layer]
    incorrect_vec = hidden_states[:, rep_kwargs["rep_token"], :].squeeze() # torch.Size([4096])
    incorrect_vec = incorrect_vec.cpu().to(dtype=torch.float32).numpy()


    # calculate distance
    distance_vec = correct_vec - incorrect_vec
    distance = np.linalg.norm(distance_vec)
    # proj_distance_vec = project_onto_direction(distance_vec, truth_direction)
    # distance = np.linalg.norm(proj_distance_vec)
    distance_list.append(distance)
    print(distance)

    # distance_vec = base_correct_vec - incorrect_vec
    # proj_distance_vec = project_onto_direction(distance_vec, truth_direction)
    # distance2 = np.linalg.norm(proj_distance_vec)
    # distance_list.append(distance2)
    # print(distance2)


    # save
    save_list.append({
        "question": question,
        "correct": correct_answer,
        "incorrect": incorrect_answer,
        "distance": distance.item(),
    })

    # # calculate cosine similarity
    # correct_vec = correct_vec.to(torch.float64)
    # incorrect_vec = incorrect_vec.to(torch.float64)
    # norm_correct = torch.linalg.vector_norm(correct_vec)
    # norm_incorrect = torch.linalg.vector_norm(incorrect_vec)
    # correct_vec, incorrect_vec = correct_vec / norm_correct, incorrect_vec / norm_incorrect
    # similarity = torch.dot(correct_vec, incorrect_vec)
    # similarity_list.append(similarity.item())
    # print(similarity.item())

# calculate
print(f"The mean of distance is {np.mean(distance_list)}")
print(f"The std of distance is {np.std(distance_list)}")

# save
save_path = os.path.join("/data2/common/weixinchen/AdaptiveDPO/results/dist", f"{args.save_name}.json")
with open(save_path, 'w') as json_file:
    json.dump(save_list, json_file, indent=4)


