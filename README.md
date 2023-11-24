# AdaptiveDPO
## Create data
```
python create_pair_responses.py --model_name $MODEL_NAME$ --useFS # preferred
python create_pair_responses.py --model_name $MODEL_NAME$
python create_pair_responses.py --model_name $MODEL_NAME$ --useGT --useFS
```

## Combine data
```
python combine_new_correct_ori_wrong.py --new_data_path $new_data_path$ --ori_data_path $ori_data_path$
```

## DPO


## Evaluation

