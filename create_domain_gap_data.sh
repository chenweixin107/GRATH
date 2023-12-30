#!/bin/bash


## 700 model generations given arc-c (w/ ood few-shot, w/o gt)
#echo "Running 700 model generations given arc-c (w/ ood few-shot, w/o gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_ood" --useFS --fs_type="out" --fs_num=6
#
## 700 model generations given arc-c (w/ ood few-shot, w/ gt)
#echo "Running 700 model generations given arc-c (w/ ood few-shot, w/ gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_ood" --useFS --useGT --fs_type="out" --fs_num=6

## 700 model generations given arc-c (w/ in-domain few-shot, w/o gt)
#CUDA_VISIBLE_DEVICES="2" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_indomain" --useFS --fs_type="in" --fs_num=6
#
## 700 model generations given arc-c (w/ in-domain few-shot, w/ gt)
#CUDA_VISIBLE_DEVICES="2" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_indomain" --useFS --useGT --fs_type="in" --fs_num=6
#
#
## 700 model generations given arc-c (w/o few-shot, w/o gt)
#CUDA_VISIBLE_DEVICES="2" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_none" --fs_type="none" --fs_num=0
#
## 700 model generations given arc-c (w/o few-shot, w/ gt)
#CUDA_VISIBLE_DEVICES="2" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_none" --useGT --fs_type="none" --fs_num=0


## 700 model generations given arc-c (w/ ood few-shot, w/o gt)
#echo "Running 700 model generations given arc-c (w/ ood few-shot, w/o gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_ood" --useFS --fs_type="out" --fs_num=4
#
## 700 model generations given arc-c (w/ ood few-shot, w/ gt)
#echo "Running 700 model generations given arc-c (w/ ood few-shot, w/ gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_ood" --useFS --useGT --fs_type="out" --fs_num=4
#
## 700 model generations given arc-c (w/ in-domain few-shot, w/o gt)
#echo "Running 700 model generations given arc-c (w/ in-domain few-shot, w/o gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_indomain" --useFS --fs_type="in" --fs_num=4
#
## 700 model generations given arc-c (w/ in-domain few-shot, w/ gt)
#echo "Running 700 model generations given arc-c (w/ in-domain few-shot, w/ gt)"
#CUDA_VISIBLE_DEVICES="7" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_arcc_few_indomain" --useFS --useGT --fs_type="in" --fs_num=4


# 700 model generations given tqa (w/ in-domain few-shot, w/o gt)
echo "Running 700 model generations given arc-c (w/ in-domain few-shot, w/o gt)"
CUDA_VISIBLE_DEVICES="6" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_tqa_few_indomain" --useFS --fs_type="in" --fs_num=6 --data_name="truthful_qa" --subdata_name="multiple_choice" --split="validation"

# 700 model generations given tqa (w/ in-domain few-shot, w/ gt)
echo "Running 700 model generations given arc-c (w/ in-domain few-shot, w/ gt)"
CUDA_VISIBLE_DEVICES="6" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_tqa_few_indomain" --useFS --useGT --fs_type="in" --fs_num=6 --data_name="truthful_qa" --subdata_name="multiple_choice" --split="validation"

# 700 model generations given tqa (w/ ood few-shot, w/o gt)
echo "Running 700 model generations given arc-c (w/ ood few-shot, w/o gt)"
CUDA_VISIBLE_DEVICES="6" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_tqa_few_ood" --useFS --fs_type="out" --fs_num=6 --data_name="truthful_qa" --subdata_name="multiple_choice" --split="validation"

# 700 model generations given tqa (w/ ood few-shot, w/ gt)
echo "Running 700 model generations given arc-c (w/ ood few-shot, w/ gt)"
CUDA_VISIBLE_DEVICES="6" python create_pair_responses_domain_gap.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="prompt_tqa_few_ood" --useFS --useGT --fs_type="out" --fs_num=6 --data_name="truthful_qa" --subdata_name="multiple_choice" --split="validation"

