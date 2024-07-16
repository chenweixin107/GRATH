#!/bin/bash

run_python_command() {
    iteration=$1
    echo "Running iteration ${iteration}:"

    echo "Creating data..."
    if [ "${iteration}" -eq 2 ]; then
      echo "Pass"
    else
      if [ "${iteration}" -eq 0 ]; then
        CUDA_VISIBLE_DEVICES="6" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --save_prefix="iter${iteration}_arcc" --useFS
      else
        CUDA_VISIBLE_DEVICES="6" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000" --save_prefix="iter${iteration}_arcc" --useFS
      fi
    fi

    echo "Combining data..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to combine data"
    else
      CUDA_VISIBLE_DEVICES="6" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json"
    fi

    echo "Conducting DPO..."
    if [ "${iteration}" -eq 0 ]; then
      CUDA_VISIBLE_DEVICES="6" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0
    else
      CUDA_VISIBLE_DEVICES="6" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0
    fi
    
}

for iteration in {2..9..1}
do
  run_python_command "$iteration"
done

