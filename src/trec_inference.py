#!/usr/bin/env python3

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple, Iterator, Set, Dict

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from sklearn.metrics import precision_score, recall_score, f1_score

# Optional LoRA support
try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

# Allowed labels and regex patterns
LABELS = ["eligible", "excluded"]
EXTRACT_RE = re.compile(r"\boutput\s*[:\-]?\s*(eligible|excluded)\b", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b(eligible|excluded)\b", re.IGNORECASE)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer TREC eligibility with optional LoRA adapters and CoT prompts."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path for load_from_disk of TREC dataset split")
    parser.add_argument("--model", type=str, required=True,
                        help="Base Hugging Face causal LM model or path")
    parser.add_argument("--lora_path", type=Path, default=None,
                        help="Directory of LoRA adapters to merge (optional)")
    parser.add_argument("--prompt_template", type=Path, required=True,
                        help="Text file with prompt template using {topic_text} and {xml_text}")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save predictions and metrics")
    parser.add_argument("--prompt_name", type=str, default="run",
                        help="Identifier for this run (used in file names)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    return parser.parse_args()


def load_existing_predictions(path: Path) -> Set[Tuple[str, str]]:
    """Read existing JSONL to skip already-processed examples."""
    keys: Set[Tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                keys.add((rec.get("topic_text", ""), rec.get("xml_text", "")))
            except json.JSONDecodeError:
                continue
    return keys


def load_model(
    model_name: str,
    lora_path: Path,
    device: str
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Load tokenizer and optionally merge LoRA adapters into the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )
    if lora_path:
        if not _PEFT_AVAILABLE:
            logging.error("LoRA requires peft: `pip install peft`")
            sys.exit(1)
        lora_model = PeftModel.from_pretrained(model, str(lora_path))
        model = lora_model.merge_and_unload()

    return tokenizer, model.to(device)


def build_generator(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    min_length: int = 5
) -> pipeline:
    """Construct a HF text-generation pipeline with a MinLengthLogitsProcessor."""
    processor = LogitsProcessorList([
        MinLengthLogitsProcessor(min_length=min_length, eos_token_id=tokenizer.eos_token_id)
    ])
    device_id = 0 if device == "cuda" else -1
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        batch_size=batch_size,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=processor,
        max_new_tokens=1024,
    )


def extract_label(text: str) -> str:
    """Extract the last matching label or return 'none'."""
    m = EXTRACT_RE.search(text)
    if m:
        return m.group(1).lower()
    matches = FALLBACK_RE.findall(text)
    return matches[-1].lower() if matches else "none"


def batch_iterator(
    dataset,
    batch_size: int,
    done_keys: Set[Tuple[str, str]]
) -> Iterator[List[Tuple[str, str, str]]]:
    """Yield batches of (topic_text, xml_text, gold_label), skipping done keys."""
    batch: List[Tuple[str, str, str]] = []
    for ex in dataset:
        topic = ex.get("topic_text", "").strip()
        xml = ex.get("xml_text", "").strip()
        gold = ex.get("label", "").lower().strip()
        key = (topic, xml)
        if key in done_keys:
            continue
        batch.append((topic, xml, gold))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_record(
    fh,
    topic: str,
    xml: str,
    generated: str,
    pred: str,
    gold: str
) -> None:
    """Append a single JSON record to the output file."""
    rec: Dict[str, str] = {
        "topic_text": topic,
        "xml_text": xml,
        "generated": generated,
        "predicted_label": pred,
        "gold_label": gold,
    }
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy."""
    precision = precision_score(
        y_true, y_pred, pos_label=LABELS[0], average="binary", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label=LABELS[0], average="binary", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def write_metrics(
    metrics: Dict[str, float],
    none_count: int,
    total: int,
    output_dir: Path,
    prompt_name: str
) -> None:
    """Write LaTeX-friendly metrics to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{prompt_name}_TREC_metrics.txt"
    row = (
        f"{metrics['f1']:.3f} & {metrics['recall']:.3f}"
        f" & {metrics['precision']:.3f} & {metrics['accuracy']:.3f} \\\""
    )
    summary = f"None predictions: {none_count}/{total}"
    with txt_path.open("w", encoding="utf-8") as tf:
        tf.write(row + "\n" + summary + "\n")
    logging.info(f"Written metrics to {txt_path}")


def main() -> None:
    setup_logging()
    args = parse_args()

    # Load dataset and prepare outputs
    dataset = load_from_disk(args.dataset)
    total = len(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_file = args.output_dir / f"{args.prompt_name}_TREC_predictions.jsonl"
    done_keys = load_existing_predictions(preds_file)

    # Load model and tokenizer
    tokenizer, model = load_model(args.model, args.lora_path, args.device)
    generator = build_generator(tokenizer, model, args.device, args.batch_size)
    prompt_tpl = args.prompt_template.read_text(encoding="utf-8").strip()

    y_true: List[str] = []
    y_pred: List[str] = []
    none_count = 0
    processed = 0

    # Inference loop
    with preds_file.open("a", encoding="utf-8") as fh:
        for batch in batch_iterator(dataset, args.batch_size, done_keys):
            prompts = [prompt_tpl.format(topic_text=t, xml_text=x) for t, x, _ in batch]
            outputs = generator(prompts)
            for (t, x, gold), out in zip(batch, outputs):
                text = out.get("generated_text", "")
                pred_label = extract_label(text)
                y_true.append(gold)
                y_pred.append(pred_label)
                if pred_label == "none":
                    none_count += 1
                write_record(fh, t, x, text, pred_label, gold)
                processed += 1
                logging.info(f"Processed {processed}/{total}")

    # Compute and write final metrics
    metrics = compute_metrics(y_true, y_pred)
    write_metrics(metrics, none_count, processed, args.output_dir, args.prompt_name)


if __name__ == "__main__":
    main()
