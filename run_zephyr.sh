#!/bin/bash

# DPO's reference model is set to the current base model
run_python_command() {
   iteration=$1
   echo "Running iteration ${iteration}:"

   echo "Creating data..."
   if [ "${iteration}" -eq 0 ]; then
     python create_pair_responses.py --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="iter${iteration}_arcc" --useFS
   else
     python create_pair_responses.py --model_name_or_path="/data2/common/username/GRATH/models/zephyr/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-500" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --save_prefix="iter${iteration}_arcc" --useFS
   fi

   echo "Combining data..."
   if [ "${iteration}" -eq 0 ]; then
     echo "No need to combine data"
   else
     python combine_new_correct_ori_wrong.py --new_data_path="/data2/common/username/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --ori_data_path="/data2/common/username/data/truthfulness/zephyr/iter0_arcc_num_1119_useGT_False_useFS_True.json"
   fi

   echo "Conducting DPO..."
   if [ "${iteration}" -eq 0 ]; then
     accelerate launch dpo.py --dataset_path="/data2/common/username/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True.json" --model_name_or_path="HuggingFaceH4/zephyr-7b-beta" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/username/GRATH/models/zephyr/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0 --max_steps=500 --save_steps=500
   else
     accelerate launch dpo.py --dataset_path="/data2/common/username/data/truthfulness/zephyr/iter${iteration}_arcc_num_1119_useGT_False_useFS_True_combined.json" --model_name_or_path="/data2/common/username/GRATH/models/zephyr/iter_${iteration}_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-500" --model_type="zephyr" --tokenizer_name="HuggingFaceH4/zephyr-7b-beta" --output_dir="/data2/common/username/GRATH/models/zephyr/iter_$((iteration + 1))_arcc_num_1119_useGT_False_useFS_True" --seed=0 --max_steps=500 --save_steps=500
   fi

}

# Loop from 1 to 10 with a stride of 1
for iteration in {0..9..1}
do
 run_python_command "$iteration"
done
