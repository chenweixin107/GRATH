#!/bin/bash

#echo "prompt_tqa_annotated:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_annotated.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_annotated" --seed=0
#
#echo "prompt_arcc_annotated:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_arcc_annotated.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_annotated" --seed=0
#
#echo "prompt_arcc_few_none_num_1119_useGT_False_useFS_False_fs_type_none_fs_num_0:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_none_num_1119_useGT_False_useFS_False_fs_type_none_fs_num_0.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_none_num_1119_useGT_False_useFS_False_fs_type_none_fs_num_0" --seed=0
#
#echo "prompt_arcc_few_none_num_1119_useGT_True_useFS_False_fs_type_none_fs_num_0:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_none_num_1119_useGT_True_useFS_False_fs_type_none_fs_num_0.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_none_num_1119_useGT_True_useFS_False_fs_type_none_fs_num_0" --seed=0

#echo "prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_4:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_4.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_4" --seed=0
#
#echo "prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_4:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_4.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_4" --seed=0
#
#echo "prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4" --seed=0
#
#echo "prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_4:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_4.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_4" --seed=0

#echo "prompt_tqa_shakespeare_p_1.0:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_1.0.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_1.0" --seed=0
#
#echo "prompt_tqa_shakespeare_p_0.8:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.8.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_0.8" --seed=0
#
#echo "prompt_tqa_shakespeare_p_0.6:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_0.6" --seed=0
#
#echo "prompt_tqa_shakespeare_p_0.4:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.4.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_0.4" --seed=0
#
#echo "prompt_tqa_shakespeare_p_0.0:"
#CUDA_VISIBLE_DEVICES="1" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.0.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_0.0" --seed=0

echo "prompt_tqa_shakespeare_p_0.2:"
CUDA_VISIBLE_DEVICES="5" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.2.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_shakespeare_p_0.2" --seed=0

echo "prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="5" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6" --seed=0

echo "prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="5" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6" --seed=0

echo "prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="5" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6" --seed=0

echo "prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="5" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6" --seed=0

echo "prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6" --seed=0

echo "prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6" --seed=0

echo "prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_6" --seed=0 # n/a

echo "prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_domain_gap.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6" --seed=0

