#!/bin/bash

cd /home/username/lm-evaluation-harness

MODEL_NAME_OR_PATH=""

# arc_challenge, 25-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_NAME_OR_PATH \
    --tasks arc_challenge \
    --device cuda:0 \
    --num_fewshot 25 \
    --batch_size 16 \
    --output_path /data2/common/username/GRATH/results/arc_challenge.json

# hellaswag, 10-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_NAME_OR_PATH \
    --tasks hellaswag \
    --device cuda:0 \
    --num_fewshot 10 \
    --batch_size 16 \
    --output_path /data2/common/username/GRATH/results/hellaswag.json

# mmlu, 5-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_NAME_OR_PATH \
    --tasks hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda:0 \
    --num_fewshot 5 \
    --batch_size 16 \
    --output_path /data2/common/username/GRATH/results/mmlu.json

# truthfulqa_mc, 0-shot
python main.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_NAME_OR_PATH \
    --tasks truthfulqa_mc \
    --device cuda:0 \
    --num_fewshot 0 \
    --batch_size 16 \
    --output_path /data2/common/username/GRATH/results/truthfulqa_mc.json
