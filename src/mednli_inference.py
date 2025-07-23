#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set, Iterator

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)

# Attempt to import PEFT for LoRA support
try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# Valid labels for MedNLI
LABELS = ["entailment", "contradiction", "neutral"]
EXTRACT_RE = re.compile(
    r"\b(output[:\-]?\s*)?(entailment|contradiction|neutral|entailed|contradicted)\b",
    re.IGNORECASE,
)


def setup_logging() -> None:
    """Configure root logger for standardized output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Infer MedNLI labels with optional LoRA finetuning."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to local Arrow dataset directory"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Base HF model identifier or path"
    )
    parser.add_argument(
        "--lora_path", type=Path,
        help="Path to LoRA adapter directory (optional)",
        default=None,
    )
    parser.add_argument(
        "--prompt_template", type=Path, required=True,
        help="Text file with prompt template (use {sentence_1}, {sentence_2})"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to save predictions and metrics"
    )
    parser.add_argument(
        "--prompt_name", type=str, default="MedNLI",
        help="Identifier for output filenames"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda",
        help="Compute device"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for generation"
    )
    return parser.parse_args()


def load_model(
    base_model_id: str,
    lora_path: Path,
    device: str
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Load tokenizer and (optionally merged) model to device."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
    if lora_path:
        if not _PEFT_AVAILABLE:
            logging.error("LoRA support requires `peft` package")
            sys.exit(1)
        lora_model = PeftModel.from_pretrained(model, str(lora_path))
        model = lora_model.merge_and_unload()

    return tokenizer, model.to(device)


def build_generator(
    tokenizer,
    model,
    device: str,
    batch_size: int,
    min_length: int = 5
) -> pipeline:
    """Construct HF text-generation pipeline with no-short-EOS processor."""
    eos_id = tokenizer.eos_token_id
    logits_proc = LogitsProcessorList([
        MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_id)
    ])
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        batch_size=batch_size,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=logits_proc,
        max_new_tokens=1_024
    )


def extract_label(text: str) -> str:
    """Extract the last matching label or return 'none'."""
    matches = EXTRACT_RE.findall(text)
    if not matches:
        return "none"
    token = matches[-1][1].lower()
    # normalize
    return {
        'entailed': 'entailment',
        'contradicted': 'contradiction'
    }.get(token, token)


def load_done_keys(path: Path) -> Set[Tuple[str, str]]:
    """Read existing predictions to skip duplicates."""
    keys = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                keys.add((rec['sentence1'], rec['sentence2']))
            except Exception:
                continue
    return keys


def batch_iterator(
    dataset: Dataset,
    batch_size: int,
    done_keys: Set[Tuple[str, str]]
) -> Iterator[List[Tuple[str, str, str]]]:
    """Yield batches of (s1, s2, gold) for inference, skipping done."""
    batch = []
    for ex in dataset:
        s1, s2 = ex['sentence1'].strip(), ex['sentence2'].strip()
        gold = ex['gold_label'].lower().strip()
        if (s1, s2) in done_keys:
            continue
        batch.append((s1, s2, gold))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_record(
    out_fh,
    s1: str,
    s2: str,
    gold: str,
    generated: str,
    pred: str
) -> None:
    """Append a single JSON record to file handle."""
    rec = {
        'sentence1': s1,
        'sentence2': s2,
        'generated': generated,
        'predicted_label': pred,
        'gold_label': gold,
    }
    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_and_save_metrics(
    y_true: List[str],
    y_pred: List[str],
    out_dir: Path,
    prompt_name: str
) -> None:
    """Compute precision, recall, F1, accuracy, and save LaTeX row."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    total = len(y_true)
    correct = sum(gt == pd for gt, pd in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0
    precision = precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)

    row = f"{f1:.3f} & {recall:.3f} & {precision:.3f} & {accuracy:.3f} \\\""
    metrics_file = out_dir / f"{prompt_name}_MedNLI_metrics.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(row + "\n", encoding='utf-8')
    logging.info(f"Saved metrics to {metrics_file}")


def main() -> None:
    setup_logging()
    args = parse_args()

    # Prepare outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_file = args.output_dir / f"{args.prompt_name}_MedNLI_predictions.jsonl"
    done_keys = load_done_keys(preds_file)

    # Load resources
    ds = load_from_disk(str(args.dataset))
    tokenizer, model = load_model(args.model, args.lora_path, args.device)
    gen = build_generator(tokenizer, model, args.device, args.batch_size)
    prompt_tpl = args.prompt_template.read_text(encoding='utf-8').strip()

    # Inference loop
    y_true, y_pred = [], []
    processed = 0

    with preds_file.open('a', encoding='utf-8') as out_fh:
        for batch in batch_iterator(ds, args.batch_size, done_keys):
            prompts = [prompt_tpl.format(sentence_1=s1, sentence_2=s2) for s1, s2, _ in batch]
            results = gen(prompts)
            for (s1, s2, gold), res in zip(batch, results):
                gen_text = res['generated_text']
                pred_label = extract_label(gen_text)
                y_true.append(gold)
                y_pred.append(pred_label)
                write_record(out_fh, s1, s2, gold, gen_text, pred_label)
                processed += 1
            logging.info(f"Processed {processed}/{len(ds)} examples")

    # Final metrics
    compute_and_save_metrics(y_true, y_pred, args.output_dir, args.prompt_name)


if __name__ == '__main__':
    main()