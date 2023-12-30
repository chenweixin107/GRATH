import argparse
import os
os.environ['TRANSFORMERS_CACHE'] = '/data2/common/weixinchen/huggingface/cache/'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


"""
Fine-Tune Llama2-7b-chat on (adversarial prompts, gt labels)
"""
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--data_name", type=str, default="ai2_arc")
    parser.add_argument("--data_path", type=str, default=None)
    # parser.add_argument("--subset", type=str, default="data/finetune")
    # parser.add_argument("--split", type=str, default="train")
    # parser.add_argument("--size_valid_set", type=int, default=4000)
    # parser.add_argument("--streaming", action="store_true")
    # parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    # prompt = "For the given sentence, label the sentiment of the sentence as positive or negative. The answer should be exactly 'positive' or 'negative'."
    # if example['question'].startswith("sentence:"):
    #     text = f"{prompt}\n{example['question'].replace('sentence:', 'Sentence:')}\n\nAnswer: {example['response_j']}"
    # else:
    #     text = f"{prompt}\nSentence: {example['question']}\n\nAnswer: {example['response_j']}"
    # text = f"{example['message']}\nAnswer: {example['response_j']}"

    # text = f"{example['message']} {example['response_j']}"

    if "choices" in example.keys(): # arc
        question, choices, answer_key = example["question"], example["choices"], example["answerKey"]
        texts, labels = choices["text"], choices["label"]
        correct_idx = labels.index(answer_key)
        correct_answer = texts[correct_idx]
    elif "correct" in example.keys(): # generated_arc_hasgt / generated_arc_nogt
        question, correct_answer = example["question"], example["correct"]
    elif "best_answer" in example.keys(): # tqa
        question, correct_answer = example["question"], example["best_answer"]

    # text = f"<s>[INST] {question} [/INST] {correct_answer} </s>"
    text = f"[INST] {question} [/INST] {correct_answer}"

    return text


def create_datasets(tokenizer, args):
    # dataset = load_dataset("json", data_files={"train": f"/data2/common/weixinchen/data/truthfulness/{args.data_name}_train.json",
    #                                           "test": f"/data2/common/weixinchen/data/truthfulness/{args.data_name}_test.json"})

    # # training dataset
    # if args.data_name == "arc":
    #     dataset1 = load_dataset("ai2_arc", 'ARC-Easy')
    #     dataset2 = load_dataset("ai2_arc", 'ARC-Challenge')
    #     train_dataset = concatenate_datasets([dataset1['train'], dataset2['train']])
    # elif args.data_name == "generated_arc_nogt":
    #     dataset1 = load_dataset("json", data_files={'train': "/data2/common/weixinchen/data/truthfulness/Llama-2-7b-chat-hf_ai2_arc_ARC-Easy_train_useGT_False.json"})
    #     dataset2 = load_dataset("json", data_files={'train': "/data2/common/weixinchen/data/truthfulness/Llama-2-7b-chat-hf_ai2_arc_ARC-Challenge_train_useGT_False.json"})
    #     train_dataset = concatenate_datasets([dataset1['train'], dataset2['train']])
    # elif args.data_name == "generated_arc_hasgt":
    #     dataset1 = load_dataset("json", data_files={'train': "/data2/common/weixinchen/data/truthfulness/Llama-2-7b-chat-hf_ai2_arc_ARC-Easy_train_useGT_True.json"})
    #     dataset2 = load_dataset("json", data_files={'train': "/data2/common/weixinchen/data/truthfulness/Llama-2-7b-chat-hf_ai2_arc_ARC-Challenge_train_useGT_True.json"})
    #     train_dataset = concatenate_datasets([dataset1['train'], dataset2['train']])
    # elif args.data_name == "dpo_generated_arc_nogt":
    #     train_dataset = load_dataset("json", data_files={'train': "/data2/common/weixinchen/data/truthfulness/no_sft_dpo_generated_arcc_model_ai2_arc_ARC-Challenge_train_useGT_False.json"})['train']
    # elif args.data_name == "tqa":
    #     train_dataset = load_dataset("truthful_qa", 'generation')['validation']
    # else:
    #     raise ValueError(f"There is no {args.data_name} dataset.")

    if args.data_path == None:
        if "arc" in args.data_name:
            if args.data_name == "ai2_arc/ARC-Challenge":
                train_dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
            elif args.data_name == "ai2_arc/ARC-Easy":
                train_dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
            else:
                raise ValueError(f"There is no {args.data_name} dataset.")
        else:
            train_dataset = load_dataset(args.data_name, split=split)
    else:
        print("Loading data from path...")
        train_dataset = load_dataset("json", data_files={"train": args.data_path})["train"]
    train_dataset = train_dataset.select(range(700))

    # validation dataset
    val_dataset = load_dataset("truthful_qa", 'generation')['validation']

    train_data = train_dataset
    valid_data = val_dataset
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")

    # model_name = args.model_id.split("/")[-1]
    # args.output_dir = os.path.join(args.output_dir, f"{model_name}_{args.data_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="llama-7b-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    # assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
