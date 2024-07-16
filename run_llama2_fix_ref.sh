#!/bin/bash

# DPO's reference model is set to the pretrained base_model (recommended, which could avoid overfitting problem)
run_python_command() {
   iteration=$1
   seed=$2
   echo "Running iteration ${iteration} with seed ${seed}:"

   echo "Creating data..."
   if [ "${iteration}" -eq 0 ]; then
     python create_pair_responses.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
   else
     python create_pair_responses.py --model_name_or_path="/data2/common/username/GRATH/models/llama2/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --save_prefix="fix_ref_iter${iteration}_arcc" --useFS --seed=${seed}
   fi

   echo "Combining data..."
   if [ "${iteration}" -eq 0 ]; then
     echo "No need to combine data"
   else
     python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/username/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --ori_data_path="/data2/common/username/data/truthfulness/llama2/iter0_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json"
   fi

   echo "Conducting DPO..."
   if [ "${iteration}" -eq 0 ]; then
     accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/username/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}.json" --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/username/GRATH/models/llama2/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
   else
     accelerate launch dpo_fix_ref.py --dataset_path="/data2/common/username/data/truthfulness/llama2/fix_ref_iter${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}_combined.json" --model_name_or_path="/data2/common/username/GRATH/models/llama2/fix_ref_iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_${seed}/checkpoint-1000" --model_type="llama2" --tokenizer_name="meta-llama/Llama-2-7b-chat-hf" --output_dir="/data2/common/username/GRATH/models/llama2/fix_ref_iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=${seed}
   fi
}

# Loop from 1 to 10 with a stride of 1
for iteration in {0..9..1}
do
 seed=0
 run_python_command "$iteration" "$seed"
done
