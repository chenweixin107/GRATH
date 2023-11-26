#!/bin/bash

#python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --save_prefix iter0_arcc --useFS
#accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True --seed 0

#CUDA_VISIBLE_DEVICES="5" python create_pair_responses.py --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --save_prefix iter1_arcc --useFS
#CUDA_VISIBLE_DEVICES="5" python combine_new_correct_ori_wrong.py --new_data_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True.json --ori_data_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json
#CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_2_arcc_num_1119_useGT_False_useFS_True --seed 0




#directory_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_0_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000"
#mkdir -p "$directory_path"
#CUDA_VISIBLE_DEVICES="5" python transform_base_model.py --base_model_name="meta-llama/Llama-2-7b-chat-hf" --output_name="/data2/common/weixinchen/AdaptiveDPO/models/iter_0_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000"

run_python_command() {
    iteration=$1
    echo "Running iteration ${iteration}:"

    echo "Creating data..."
    if [ -e "/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" ]; then
      echo "Path exists"
    else
      echo "Path does not exist"
      CUDA_VISIBLE_DEVICES="5" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000" --save_prefix="iter${iteration}_arcc" --useFS
    fi

    echo "Combining data..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to combine data"
    else
      CUDA_VISIBLE_DEVICES="5" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/iter${iteration-1}_arcc_num_1119_useGT_False_useFS_True.json"
    fi

    echo "Conducting DPO..."
    if [ "${iteration}" -eq 0 ]; then
      CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration+1}_arcc_num_1119_useGT_False_useFS_True" --seed 0
    else
      CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration+1}_arcc_num_1119_useGT_False_useFS_True" --seed 0
    fi
}

# Loop from 1 to 10 with a stride of 1
for iteration in {0..9..1}
do
  run_python_command "$iteration"
done
