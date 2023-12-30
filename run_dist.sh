#!/bin/bash

#python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --save_prefix iter0_arcc --useFS
#accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True --seed 0

#CUDA_VISIBLE_DEVICES="5" python create_pair_responses.py --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --save_prefix iter1_arcc --useFS
#CUDA_VISIBLE_DEVICES="5" python combine_new_correct_ori_wrong.py --new_data_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True.json --ori_data_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json
#CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_2_arcc_num_1119_useGT_False_useFS_True --seed 0


run_python_command() {
    iteration=$1
    echo "Running iteration ${iteration}:"
    CUDA_VISIBLE_DEVICES="2" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/start_base_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0
#"/data2/common/weixinchen/data/truthfulness/llama2/iter1_arcc_num_1119_useGT_False_useFS_True_seed_1_combined.json"
}

# Loop from 1 to 10 with a stride of 1
for iteration in {1..2..1}
do
  run_python_command "$iteration"
done


#echo "Running seed 0, iteration 1:"
#CUDA_VISIBLE_DEVICES="2" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/start_base_iter_2_arcc_num_1119_useGT_False_useFS_True" --seed=0
#
#echo "Running seed 0, iteration 2:"
#CUDA_VISIBLE_DEVICES="2" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter2_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/start_base_iter_3_arcc_num_1119_useGT_False_useFS_True" --seed=0
#
#echo "Running seed 1, iteration 1:"
#CUDA_VISIBLE_DEVICES="2" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/iter1_arcc_num_1119_useGT_False_useFS_True_seed_1_combined.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/start_base_iter_2_arcc_num_1119_useGT_False_useFS_True" --seed=1
#
#echo "Running seed 1, iteration 2:"
#CUDA_VISIBLE_DEVICES="2" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/iter2_arcc_num_1119_useGT_False_useFS_True_seed_1_combined.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/start_base_iter_3_arcc_num_1119_useGT_False_useFS_True" --seed=1


#CUDA_VISIBLE_DEVICES="0" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json" --save_name "iter0_arcc_num_1119_useGT_False_useFS_True"
#CUDA_VISIBLE_DEVICES="1" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json" --save_name "iter1_arcc_num_1119_useGT_False_useFS_True_combined"
#CUDA_VISIBLE_DEVICES="0" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter2_arcc_num_1119_useGT_False_useFS_True_combined.json" --save_name "iter2_arcc_num_1119_useGT_False_useFS_True_combined"
#CUDA_VISIBLE_DEVICES="1" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter3_arcc_num_1119_useGT_False_useFS_True_combined.json" --save_name "iter3_arcc_num_1119_useGT_False_useFS_True_combined"
#CUDA_VISIBLE_DEVICES="0" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter4_arcc_num_1119_useGT_False_useFS_True_combined.json" --save_name "iter4_arcc_num_1119_useGT_False_useFS_True_combined"
#CUDA_VISIBLE_DEVICES="1" python get_rep.py --data_path "/data2/common/weixinchen/data/truthfulness/iter5_arcc_num_1119_useGT_False_useFS_True_combined.json" --save_name "iter5_arcc_num_1119_useGT_False_useFS_True_combined"

