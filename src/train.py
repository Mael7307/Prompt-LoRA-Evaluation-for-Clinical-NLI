#!/usr/bin/env python3

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType


def setup_logging() -> None:
    """Configure root logger for INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for conversational causal LM"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="HF dataset identifier with train/dev splits"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to save adapters and logs"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Base causal LM model name or path"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max sequence length for prompt+response"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100
    )
    parser.add_argument(
        "--save_steps", type=int, default=500
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="Override total training steps if > 0"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Set random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def preprocess_batch(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize conversational examples: split on markers,
    combine prompt+response, mask prompt loss tokens.
    Expects examples['text'] with user/assistant markers.
    """
    instruct_marker = "<|start_header_id|>user<|end_header_id|>\n\n"
    response_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    input_ids, attention_mask, labels = [], [], []

    for text in examples['text']:
        prompt, response = text.split(response_marker, 1)
        prompt = prompt.replace(instruct_marker, "").strip()
        response = response.strip()

        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        ).input_ids
        enc_response = tokenizer(
            response,
            truncation=True,
            max_length=max_length - len(enc_prompt),
            add_special_tokens=False
        ).input_ids + [tokenizer.eos_token_id]

        ids = enc_prompt + enc_response
        mask = [1] * len(ids)
        label_ids = [-100] * len(enc_prompt) + enc_response

        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(label_ids)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def main() -> None:
    setup_logging()
    args = parse_args()
    set_random_seed(args.seed)

    logging.info("Loading dataset %s", args.dataset_name)
    raw_train = load_dataset(args.dataset_name, split="train")
    raw_eval = load_dataset(args.dataset_name, split="dev")

    logging.info("Loading tokenizer and model %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    logging.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    logging.info("Preprocessing training data")
    train_ds = raw_train.map(
        lambda ex: preprocess_batch(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text']
    )
    logging.info("Preprocessing evaluation data")
    eval_ds = raw_eval.map(
        lambda ex: preprocess_batch(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text']
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=20,
        gradient_accumulation_steps=16,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        logging_dir=str(args.output_dir / "tensorboard_logs"),
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logging.info("Starting training")
    trainer.train()

    adapter_dir = args.output_dir / "lora_adapters"
    model.save_pretrained(adapter_dir)
    logging.info("Saved LoRA adapters to %s", adapter_dir)


if __name__ == '__main__':
    main()
