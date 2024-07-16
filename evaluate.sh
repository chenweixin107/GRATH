#!/bin/bash

cd /home/weixinchen/lm-evaluation-harness

# truthfulqa_mc
run_python_command() {
   checkpoint=$1
   echo "Using model from: /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}"
   python main.py \
   --model hf-causal \
   --model_args pretrained="/data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}" \
   --tasks truthfulqa_mc \
   --device cuda:7 \
   --num_fewshot 0 \
   --batch_size 16 \
   --output_path "/data2/common/weixinchen/AdaptiveDPO/results/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/truthfulqa_mc_cp_${checkpoint}.json"
}

# Loop from 1 to 10 with a stride of 1
for checkpoint in {100..2000..100}
do
 run_python_command "$checkpoint"
done


## arc_challenge
#run_python_command() {
#    checkpoint=$1
#    echo "Using model from: /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}"
#    python main.py \
#    --model hf-causal \
#    --model_args pretrained="/data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}" \
#    --tasks arc_challenge \
#    --device cuda:7 \
#    --num_fewshot 25 \
#    --batch_size 16 \
#    --output_path "/data2/common/weixinchen/AdaptiveDPO/results/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/arc_challenge_cp_${checkpoint}.json"
#}
#
## Loop from 1 to 10 with a stride of 1
#for checkpoint in {100..2000..100}
#do
#  run_python_command "$checkpoint"
#done
#
#
# hellaswag
run_python_command() {
    checkpoint=$1
    echo "Using model from: /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}"
    python main.py \
    --model hf-causal \
    --model_args pretrained="/data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}" \
    --tasks hellaswag \
    --device cuda:8 \
    --num_fewshot 10 \
    --batch_size 16 \
    --output_path "/data2/common/weixinchen/AdaptiveDPO/results/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/hellaswag_cp_${checkpoint}.json"
}

# Loop from 1 to 10 with a stride of 1
for checkpoint in {1300..2000..100}
do
  run_python_command "$checkpoint"
done


# mmlu
run_python_command() {
    checkpoint=$1
    echo "Using model from: /data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}"
    python main.py \
    --model hf-causal \
    --model_args pretrained="/data2/common/weixinchen/AdaptiveDPO/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/checkpoint-${checkpoint}" \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda:8 \
    --num_fewshot 5 \
    --batch_size 16 \
    --output_path "/data2/common/weixinchen/AdaptiveDPO/results/iter_1_arcc_num_1119_useGT_False_useFS_True_seed_0/mmlu_cp_${checkpoint}.json"
}

# Loop from 1 to 10 with a stride of 1
for checkpoint in {100..2000..100}
do
  run_python_command "$checkpoint"
done
