#!/bin/bash

#python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --save_prefix iter0_arcc --useFS
#accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True --seed 0

#CUDA_VISIBLE_DEVICES="5" python create_pair_responses.py --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --save_prefix iter1_arcc --useFS
#CUDA_VISIBLE_DEVICES="5" python combine_new_correct_ori_wrong.py --new_data_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True.json --ori_data_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json
#CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_2_arcc_num_1119_useGT_False_useFS_True --seed 0

## ref_model = base_model
#run_python_command() {
#    iteration=$1
#    echo "Running iteration ${iteration}:"
#
#    echo "Creating data..."
#    if [ "${iteration}" -eq 0 ]; then
##      CUDA_VISIBLE_DEVICES="3" python create_pair_responses.py --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="iter${iteration}_arcc" --useFS
#      echo "No need to create data"
#    else
#      CUDA_VISIBLE_DEVICES="3" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-500" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="iter${iteration}_arcc" --useFS
#    fi
#
#    echo "Combining data..."
#    if [ "${iteration}" -eq 0 ]; then
#      echo "No need to combine data"
#    else
#      CUDA_VISIBLE_DEVICES="3" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/zephyr/iter0_arcc_num_1119_useGT_False_useFS_True.json"
#    fi
#
#    echo "Conducting DPO..."
#    if [ "${iteration}" -eq 0 ]; then
#      CUDA_VISIBLE_DEVICES="3" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0 --max_steps=500 --save_steps=500
#    else
#      CUDA_VISIBLE_DEVICES="3" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-500" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0 --max_steps=500 --save_steps=500
#    fi
#
#}
#
## Loop from 1 to 10 with a stride of 1
#for iteration in {0..3..1}
#do
#  run_python_command "$iteration"
#done




# ref_model = pretrained_model
run_python_command() {
    iteration=$1
    seed=$2
    echo "Running iteration ${iteration} with seed ${seed}:"

    echo "Creating data..."
    if [ "${iteration}" -eq 0 ] || [ "${iteration}" -eq 1 ]; then
      echo "No need to create data"
#      CUDA_VISIBLE_DEVICES="3" python create_pair_responses.py --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
    else
      CUDA_VISIBLE_DEVICES="3" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
    fi


    echo "Combining data..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to combine data"
    else
      CUDA_VISIBLE_DEVICES="3" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/zephyr/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/zephyr/iter0_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json"
    fi

    echo "Conducting DPO..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to train model"
#      CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/zephyr/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
    elif [ "${iteration}" -eq 1 ]; then
      CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/zephyr/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
    else
      CUDA_VISIBLE_DEVICES="3" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/zephyr/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/zephyr/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
    fi
}

# Loop from 1 to 10 with a stride of 1
for iteration in {0..9..1}
do
  seed=0
  run_python_command "$iteration" "$seed"
done
