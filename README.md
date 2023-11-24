# AdaptiveDPO
## Create data
```
python create_pair_responses.py --model_name $model_name$ --useFS # preferred
python create_pair_responses.py --model_name $model_name$
python create_pair_responses.py --model_name $model_name$ --useGT --useFS
```

## Combine data
```
python combine_new_correct_ori_wrong.py --new_data_path $new_data_path$ --ori_data_path $ori_data_path$
```

## DPO
```
accelerate launch dpo_llama2.py --model_name_or_path $model_name_or_path$ --output_dir $output_dir$ --dataset_path $dataset_path$
```

## Evaluation
Use https://github.com/EleutherAI/lm-evaluation-harness
