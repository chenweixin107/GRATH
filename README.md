# AdaptiveDPO
## Create data
```
python create_pair_responses.py --model_name_or_path $model_name_or_path$ --useFS # preferred
python create_pair_responses.py --model_name_or_path $model_name_or_path$
python create_pair_responses.py --model_name_or_path $model_name_or_path$ --useGT --useFS
```
Data is saved at: /data2/common/weixinchen/data/truthfulness/{model_name_split}_{args.data_name}_{args.subdata_name}_{args.split}_useGT_{str(args.useGT)}_useFS_{str(args.useFS)}.json

## Combine data
```
python combine_new_correct_ori_wrong.py --new_data_path $new_data_path$ --ori_data_path $ori_data_path$
```
Data is saved at: /data2/common/weixinchen/data/truthfulness/{new_data_path}_combined.json

## DPO
```
accelerate launch dpo.py --dataset_path $dataset_path$ --model_name_or_path $model_name_or_path$ --output_dir $output_dir$ # preferred
accelerate launch dpo.py --dataset_name $dataset_name$ --model_name_or_path $model_name_or_path$ --output_dir $output_dir$
```

## Evaluation
Use https://github.com/EleutherAI/lm-evaluation-harness
```
# arc_challenge, 25-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=/data2/common/weixinchen/RLKF/sft_dpo_models/no_sft_adaptive_dpo/dpo_generated_correct_arc_generated_incorrect_arc_nogt/checkpoint-1000 \
    --tasks arc_challenge \
    --device cuda:3 \
    --num_fewshot 25 \
    --batch_size 16 \
    --output_path /data2/common/weixinchen/RLKF/results/arc_challenge/dpo2.json

# hellaswag, 10-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=/data2/common/weixinchen/RLKF/sft_dpo_models/no_sft_adaptive_dpo/dpo_generated_correct_arc_generated_incorrect_arc_nogt/checkpoint-1000 \
    --tasks hellaswag \
    --device cuda:1 \
    --num_fewshot 10 \
    --batch_size 16 \
    --output_path /data2/common/weixinchen/RLKF/results/hellaswag/dpo2.json

# mmlu, 5-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=/data2/common/weixinchen/RLKF/sft_dpo_models/no_sft_adaptive_dpo/dpo_generated_correct_arc_generated_incorrect_arc_nogt/checkpoint-1000 \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda:8 \
    --num_fewshot 5 \
    --batch_size 16 \
    --output_path /data2/common/weixinchen/RLKF/results/mmlu/dpo2.json

# truthfulqa_mc, 0-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=/data2/common/weixinchen/RLKF/sft_dpo_models/no_sft_adaptive_dpo/dpo_generated_correct_arc_generated_incorrect_arc_nogt/checkpoint-1000 \
    --tasks truthfulqa_mc \
    --device cuda:5 \
    --num_fewshot 0 \
    --batch_size 16 \
    --output_path /data2/common/weixinchen/RLKF/results/truthfulqa_mc/dpo2.json
```
