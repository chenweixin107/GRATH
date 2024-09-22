#!/bin/bash

#echo "prompt_tqa_annotated:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_annotated.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_annotated"

#echo "prompt_arcc_annotated:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_arcc_annotated.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_arcc_annotated"
#
#echo "prompt_tqa_shakespeare_p_1.0:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_1.0.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_shakespeare_p_1.0"
#
#echo "prompt_tqa_shakespeare_p_0.8:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.8.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_shakespeare_p_0.8"
#
#echo "prompt_tqa_shakespeare_p_0.6:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_shakespeare_p_0.6"
#
#echo "prompt_tqa_shakespeare_p_0.4:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.4.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_shakespeare_p_0.4"

#echo "prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_arcc_few_indomain_num_1119_useGT_False_useFS_True_fs_type_in_fs_num_6"
#
#echo "prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_arcc_few_indomain_num_1119_useGT_True_useFS_True_fs_type_in_fs_num_6"
#
#echo "prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_arcc_few_ood_num_1119_useGT_False_useFS_True_fs_type_out_fs_num_4"
#
#echo "prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6:"
#CUDA_VISIBLE_DEVICES="6" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_arcc_few_ood_num_1119_useGT_True_useFS_True_fs_type_out_fs_num_6"


echo "prompt_tqa_shakespeare_p_0.2:"
CUDA_VISIBLE_DEVICES="4" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/prompt_tqa_shakespeare_p_0.2.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_shakespeare_p_0.2"

echo "prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="4" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_few_indomain_num_817_useGT_False_useFS_True_fs_type_in_fs_num_6"

echo "prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6:"
CUDA_VISIBLE_DEVICES="4" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_few_indomain_num_817_useGT_True_useFS_True_fs_type_in_fs_num_6"

echo "prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="4" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_few_ood_num_817_useGT_False_useFS_True_fs_type_out_fs_num_6"

echo "prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6:"
CUDA_VISIBLE_DEVICES="4" torchrun --nnodes 1 --nproc_per_node 1 --master_port=29512 sft.py --data_path="/data2/common/weixinchen/data/truthfulness/llama2/prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6.json" --model_id="meta-llama/Llama-2-7b-chat-hf" --no_gradient_checkpointing --learning_rate=1e-5 --max_steps=1000 --eval_freq=2000 --save_freq=1000 --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/domain_gap/sft/prompt_tqa_few_ood_num_817_useGT_True_useFS_True_fs_type_out_fs_num_6"
