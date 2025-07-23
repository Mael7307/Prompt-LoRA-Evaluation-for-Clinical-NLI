#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set, Iterator, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)

# Attempt PEFT for LoRA support
try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# Labels
LABELS = ["entailment", "contradiction"]
EXTRACT_PATTERN = re.compile(
    r"\boutput\s*[:\-]?\s*(entailment|contradiction)\b",
    re.IGNORECASE
)
FALLBACK_PATTERN = re.compile(
    r"\b(entailed|entailment|contradicted|contradiction)\b",
    re.IGNORECASE
)


def setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Infer NLI4CT labels with (optional) LoRA adapters."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or local Arrow directory (split 'test').")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model identifier or path.")
    parser.add_argument("--lora_path", type=Path, default=None,
                        help="LoRA adapter directory (optional).")
    parser.add_argument("--prompt_template", type=Path, required=True,
                        help="Prompt template file with {premise} and {statement} placeholders.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save JSONL predictions.")
    parser.add_argument("--prompt_name", type=str, required=True,
                        help="Identifier for this run (used in filenames).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Compute device.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation.")
    return parser.parse_args()


def load_model(
    model_id: str,
    lora_path: Path,
    device: str
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Load tokenizer and model, merge LoRA if provided."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if lora_path:
        if not _PEFT_AVAILABLE:
            logging.error("PEFT not available: install peft to use --lora_path")
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
    """Construct HF text-generation pipeline with min-length processor."""
    eos_token = tokenizer.eos_token_id
    proc = LogitsProcessorList([MinLengthLogitsProcessor(min_length=min_length, eos_token_id=eos_token)])
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        batch_size=batch_size,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=proc,
        max_new_tokens=1024
    )


def extract_label(text: str) -> str:
    """Extract label from generated text or return 'none'."""
    m = EXTRACT_PATTERN.search(text)
    if m:
        return m.group(1).lower()
    # fallback to last occurrence
    matches = FALLBACK_PATTERN.findall(text)
    if matches:
        token = matches[-1].lower()
        return 'entailment' if 'entail' in token else 'contradiction'
    return 'none'


def load_completed_keys(path: Path) -> Set[Tuple[str, str]]:
    """Read existing JSONL to skip completed examples."""
    keys: Set[Tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                keys.add((rec['premise'], rec['hypothesis']))
            except Exception:
                continue
    return keys


def batch_iterator(
    dataset,
    batch_size: int,
    done_keys: Set[Tuple[str, str]]
) -> Iterator[List[Tuple[str, str, str]]]:
    """Yield batches of (premise, hypothesis, gold) skipping done."""
    batch: List[Tuple[str, str, str]] = []
    for ex in dataset:
        premise = ex.get('premise_text') or ex.get('Primary_premise', '')
        hypothesis = ex.get('hypothesis') or ex.get('Statement', '')
        gold = ex.get('gold_label', ex.get('Label', '')).lower()
        key = (premise, hypothesis)
        if key in done_keys:
            continue
        batch.append((premise, hypothesis, gold))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_record(
    fh,
    premise: str,
    hypothesis: str,
    generated: str,
    pred: str,
    gold: str
) -> None:
    """Write single JSON record to file handle."""
    rec: Dict[str, str] = {
        'premise': premise,
        'hypothesis': hypothesis,
        'generated': generated,
        'predicted_label': pred,
        'gold_label': gold
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_and_log_accuracy(
    true_labels: List[str],
    pred_labels: List[str]
) -> float:
    """Compute and log running accuracy."""
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    total = len(true_labels)
    acc = correct / total if total else 0.0
    logging.info(f"Current accuracy: {acc:.4f} over {total} samples")
    return acc


def main() -> None:
    setup_logging()
    args = parse_args()

    # Prepare output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"NLI4CT_{args.prompt_name}_predictions.jsonl"
    done_keys = load_completed_keys(out_file)

    # Load dataset
    ds = load_dataset(args.dataset, split="test")

    # Load model & tokenizer
    tokenizer, model = load_model(args.model, args.lora_path, args.device)
    generator = build_generator(tokenizer, model, args.device, args.batch_size)

    # Read prompt template
    prompt_tpl = args.prompt_template.read_text().strip()

    # Inference
    y_true: List[str] = []
    y_pred: List[str] = []
    processed = 0

    with out_file.open('a', encoding='utf-8') as fh:
        for batch in batch_iterator(ds, args.batch_size, done_keys):
            inputs = [prompt_tpl.format(premise=p, statement=h) for p, h, _ in batch]
            outputs = generator(inputs)
            for (p, h, gold), out in zip(batch, outputs):
                text = out.get('generated_text', '')
                pred = extract_label(text)
                write_record(fh, p, h, text, pred, gold)
                y_true.append(gold)
                y_pred.append(pred)
                processed += 1
            if y_true:
                compute_and_log_accuracy(y_true, y_pred)

    logging.info("Inference complete.")

if __name__ == '__main__':
    main()
