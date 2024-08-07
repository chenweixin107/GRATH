# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM
from trl import DPOTrainer
import random


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    dataset_path: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the tokenizer name"})
    seed: Optional[int] = field(default=0, metadata={"help": "random seed"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    model_type: Optional[str] = field(
        default="llama2",
        metadata={"help": "the type of model"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_dataset(
    model_type: str = None,
    data_name: str = None,
    data_path: str = None,
    split: str = "train",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    if data_path == None:
        if "arc" in data_name:
            if data_name == "ai2_arc/ARC-Challenge":
                dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
            elif data_name == "ai2_arc/ARC-Easy":
                dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
            else:
                raise ValueError(f"There is no {data_name} dataset.")
        else:
            dataset = load_dataset(data_name, split=split)
    else:
        dataset = load_dataset("json", data_files={"train": data_path})["train"]

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    if model_type == "llama2":
        user_tag = "[INST]"
        assistant_tag = "[/INST]"
    elif model_type == "zephyr":
        user_tag = "<|user|>\n"
        assistant_tag = "</s>\n<|assistant|>"
    else:
        raise ValueError(f"There is no {model_type} model type.")

    template_str = '{user_tag} {scenario} {assistant_tag}'

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        question_list, chosen_list, rejected_list = [], [], []
        if data_path == None:
            if "arc" in data_name:
                for question, choices, answer_key in zip(samples["question"], samples["choices"], samples["answerKey"]):
                    texts, labels = choices["text"], choices["label"]
                    correct_idx = labels.index(answer_key)
                    correct_answer = texts[correct_idx]
                    incorrect_answers = [text for text in texts if text != correct_answer]
                    incorrect_answer = random.choice(incorrect_answers)
                    question_list.append(template_str.format(scenario=question, user_tag=user_tag, assistant_tag=assistant_tag))
                    chosen_list.append(correct_answer)
                    rejected_list.append(incorrect_answer)
            elif "truthful_qa" in data_name:
                for question, mc1_targets in zip(samples["question"], samples["mc1_targets"]):
                    texts, labels = mc1_targets["choices"], mc1_targets["labels"]
                    correct_idx = labels.index(1)
                    correct_answer = texts[correct_idx]
                    incorrect_answers = [text for text in texts if text != correct_answer]
                    incorrect_answer = random.choice(incorrect_answers)
                    question_list.append(template_str.format(scenario=question, user_tag=user_tag, assistant_tag=assistant_tag))
                    chosen_list.append(correct_answer)
                    rejected_list.append(incorrect_answer)
            else:
                raise ValueError(f"There is no {data_name} dataset.")
        else:
            for question, correct, incorrect in zip(samples["question"], samples["correct"], samples["incorrect"]):
                question_list.append(template_str.format(scenario=question, user_tag=user_tag, assistant_tag=assistant_tag))
                chosen_list.append(correct)
                rejected_list.append(incorrect)

        return {
            "prompt": question_list,
            "chosen": chosen_list,
            "rejected": rejected_list,
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    random.seed(script_args.seed)

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    pretrained_model_name = None
    if script_args.model_type == "llama2":
        pretrained_model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif script_args.model_type == "zephyr":
        pretrained_model_name = "HuggingFaceH4/zephyr-7b-beta"
    model_ref = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_dataset(model_type=script_args.model_type, data_name=script_args.dataset_name, data_path=script_args.dataset_path, split="train", sanity_check=script_args.sanity_check)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = get_dataset(model_type=script_args.model_type, data_name=script_args.dataset_name, data_path=script_args.dataset_path, split="validation", sanity_check=True)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 4. initialize training arguments:
    script_args.output_dir = script_args.output_dir + f"_seed_{str(script_args.seed)}"
    os.makedirs(script_args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
