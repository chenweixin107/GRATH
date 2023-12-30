#!/bin/bash

#python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-chat-hf --save_prefix iter0_arcc --useFS
#accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True --seed 0

#CUDA_VISIBLE_DEVICES="5" python create_pair_responses.py --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --save_prefix iter1_arcc --useFS
#CUDA_VISIBLE_DEVICES="5" python combine_new_correct_ori_wrong.py --new_data_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True.json --ori_data_path /data2/common/weixinchen/data/truthfulness/iter0_arcc_num_1119_useGT_False_useFS_True.json
#CUDA_VISIBLE_DEVICES="5" accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/iter1_arcc_num_1119_useGT_False_useFS_True_combined.json --model_name_or_path /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-1000 --output_dir /data2/common/weixinchen/AdaptiveDPO/iter_2_arcc_num_1119_useGT_False_useFS_True --seed 0


## ref_model = base_model
## change seed
#run_python_command() {
#    iteration=$1
#    seed=$2
#    echo "Running iteration ${iteration} with seed ${seed}:"
#
#    echo "Creating data..."
#    if [ "${iteration}" -eq 0 ]; then
#      CUDA_VISIBLE_DEVICES="9" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="iter${iteration}_arcc" --useFS --seed=${seed}
#    else
#      CUDA_VISIBLE_DEVICES="9" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="iter${iteration}_arcc" --useFS --seed=${seed}
#    fi
#
#
#    echo "Combining data..."
#    if [ "${iteration}" -eq 0 ]; then
#      echo "No need to combine data"
#    else
#      CUDA_VISIBLE_DEVICES="9" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/llama2/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/llama2/iter0_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json"
#    fi
#
#    echo "Conducting DPO..."
#    if [ "${iteration}" -eq 0 ]; then
#      CUDA_VISIBLE_DEVICES="9" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
#    else
#      CUDA_VISIBLE_DEVICES="9" accelerate launch dpo.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
#    fi
#}
#
## Loop from 1 to 10 with a stride of 1
#for iteration in {0..3..1}
#do
#  seed=1
#  run_python_command "$iteration" "$seed"
#done



## ref_model = pretrained_model
#run_python_command() {
#    iteration=$1
#    seed=$2
#    echo "Running iteration ${iteration} with seed ${seed}:"
#
#    echo "Creating data..."
#    if [ "${iteration}" -eq 0 ] || [ "${iteration}" -eq 1 ]; then
#      echo "No need to create data"
##      CUDA_VISIBLE_DEVICES="9" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
#    else
#      CUDA_VISIBLE_DEVICES="9" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
#    fi
#
#
#    echo "Combining data..."
#    if [ "${iteration}" -eq 0 ] || [ "${iteration}" -eq 1 ]; then
#      echo "No need to combine data"
#    else
#      CUDA_VISIBLE_DEVICES="9" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/llama2/iter0_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json"
#    fi
#
#    echo "Conducting DPO..."
#    if [ "${iteration}" -eq 0 ]; then
#      echo "No need to train model"
##      CUDA_VISIBLE_DEVICES="9" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
#    elif [ "${iteration}" -eq 1 ]; then
#      CUDA_VISIBLE_DEVICES="9" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
#    else
#      CUDA_VISIBLE_DEVICES="9" accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
#    fi
#}
#
## Loop from 1 to 10 with a stride of 1
#for iteration in {4..9..1}
#do
#  seed=0
#  run_python_command "$iteration" "$seed"
#done


# ref_model = base_model
# seed=0, model scale=13B
run_python_command() {
    iteration=$1
    seed=$2
    echo "Running iteration ${iteration} with seed ${seed}:"

    echo "Creating data..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to create data"
#      CUDA_VISIBLE_DEVICES="1,5,7,9" python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-13b-chat-hf" --model_type="llama2_13b" --tokenizer_name="meta-llama/Llama-2-13b-chat-hf" --save_prefix="iter${iteration}_arcc" --useFS --seed=${seed}
    else
      CUDA_VISIBLE_DEVICES="8,9" python create_pair_responses.py --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2_13b/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2_13b" --tokenizer_name="meta-llama/Llama-2-13b-chat-hf" --save_prefix="iter${iteration}_arcc" --useFS --seed=${seed}
    fi


    echo "Combining data..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to combine data"
    else
      CUDA_VISIBLE_DEVICES="8,9" python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/weixinchen/data/truthfulness/llama2_13b/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --ori_data_path="/data2/common/weixinchen/data/truthfulness/llama2_13b/iter0_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json"
    fi

    echo "Conducting DPO..."
    if [ "${iteration}" -eq 0 ]; then
      echo "No need to train model"
#      CUDA_VISIBLE_DEVICES="8,9" accelerate launch --multi_gpu --num_machines 1  --num_processes 2  dpo_llama2_13b.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2_13b/iter0_arcc_num_1119_useGT_False_useFS_True_seed_0.json" --model_name_or_path="meta-llama/Llama-2-13b-chat-hf" --model_type="llama2_13b" --tokenizer_name="meta-llama/Llama-2-13b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2_13b/iter_1_arcc_num_1119_useGT_False_useFS_True" --seed=0
    else
      CUDA_VISIBLE_DEVICES="8,9" accelerate launch --multi_gpu --num_machines 1  --num_processes 2 dpo_llama2_13b.py --dataset_path="/data2/common/weixinchen/data/truthfulness/llama2_13b/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/weixinchen/AdaptiveDPO/models/llama2_13b/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2_13b" --tokenizer_name="meta-llama/Llama-2-13b-chat-hf" --output_dir="/data2/common/weixinchen/AdaptiveDPO/models/llama2_13b/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
    fi
}

# Loop from 1 to 10 with a stride of 1
for iteration in {0..1..1}
do
  seed=0
  run_python_command "$iteration" "$seed"
done
