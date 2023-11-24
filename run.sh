# DPO*1
python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-hf --useFS
accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/{model_name_split}{args.data_name}{args.subdata_name}_{args.split}useGT{str(args.useGT)}useFS{str(args.useFS)}.json --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir iter_xxx_num_xxx_useFT_xxx_useFS_xxx

# DPO*2
python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-hf --useFS
python combine_new_correct_ori_wrong.py --new_data_path $new_data_path$ --ori_data_path $ori_data_path$
accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/{model_name_split}{args.data_name}{args.subdata_name}_{args.split}useGT{str(args.useGT)}useFS{str(args.useFS)}.json --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir iter_xxx_num_xxx_useFT_xxx_useFS_xxx

# DPO*3
python create_pair_responses.py --model_name_or_path meta-llama/Llama-2-7b-hf --useFS
python combine_new_correct_ori_wrong.py --new_data_path $new_data_path$ --ori_data_path $ori_data_path$
accelerate launch dpo.py --dataset_path /data2/common/weixinchen/data/truthfulness/{model_name_split}{args.data_name}{args.subdata_name}_{args.split}useGT{str(args.useGT)}useFS{str(args.useFS)}.json --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir iter_xxx_num_xxx_useFT_xxx_useFS_xxx
