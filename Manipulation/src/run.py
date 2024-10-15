
import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import torch

import datasets
import numpy as np
from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from collator import DataCollator

from trainer_lens import Trainer, skip_instructions

# off wandb
os.environ['WANDB_DISABLED'] = "True"
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    push_end_layer: Optional[int] = field(
        default=32,
        metadata={"help": "frequency to forget."}
    )
    push_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "weight of push loss."}
    )
    pull_weight: Optional[str] = field(
        default="",
        metadata={"help": "weight of pull loss."}
    )
    retain_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "weight of retain loss."}
    )
    space_tag: Optional[str] = field(
        default="",
        metadata={"help": "tag of saved space."}
    )
    lang_list: Optional[str] = field(
        default="",
        metadata={"help": "weight of retain loss."}
    )
    layer_wise: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    model_id: Optional[str] = field(
        default=None,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False

    training_args.model_name_or_path = model_args.model_name_or_path
    training_args.push_end_layer = data_args.push_end_layer
    
    training_args.pull_weight = data_args.pull_weight
    training_args.push_weight = data_args.push_weight
    training_args.retain_weight = data_args.retain_weight

    training_args.space_tag = data_args.space_tag
    training_args.lang_list = data_args.lang_list.split(',')
    training_args.layer_wise = data_args.layer_wise
    training_args.model_id = data_args.model_id
    
    pull_weight = {}
    value = data_args.pull_weight.split(',')
    for lang, v in zip(training_args.lang_list, value):
        pull_weight.update({lang: float(v)})
    training_args.pull_weight = pull_weight

    print(training_args.lang_list)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    if 'phi3.5' in training_args.model_id:
        print("Using Phi3.5 Dataset")
        raw_datasets = load_dataset(
            os.path.join(CURRENT_DIR, "dataset_phi3.5.py"),
            data_dir=data_args.data_dir,
            trust_remote_code=True
        )
    elif 'llama3' in training_args.model_id:
        print("Using LLaMA3 Dataset")
        raw_datasets = load_dataset(
            os.path.join(CURRENT_DIR, "dataset_llama3.py"),
            data_dir=data_args.data_dir,
            trust_remote_code=True
        )
    else:
        raise ValueError("Unsupport model")
    
    raw_datasets.cleanup_cache_files()
    print(raw_datasets)
    
    if 'llama' in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            use_fast = model_args.use_fast_tokenizer,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.resize_token_embeddings(len(tokenizer))

    # Calculate n_params    
    total_params, params = 0, 0
    for n, p in model.named_parameters():
        print(n)
        if "model.embed_tokens" in n:
            p.requires_grad = False
        if "model.norm.weight" in n:
            p.requires_grad = False
        if "lm_head.weight" in n:
            p.requires_grad = False
        if "model.layers." in n:
            layer_idx = int(n.split('.')[2])
            if layer_idx < training_args.push_end_layer and layer_idx >= 0:
                p.requires_grad = False
        if p.requires_grad:
            print(n)
            total_params += p.numel()
        params += p.numel()
    print("@@@")
    print(params)
    print(
        "Total number of Tuning parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )


    if "train" not in raw_datasets:
        raise ValueError("requires a train dataset")
    train_dataset = raw_datasets["train"]

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file,
        lang_list=training_args.lang_list
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    train_result = trainer.train()

    save_path = training_args.output_dir + "/saved_weights"
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
