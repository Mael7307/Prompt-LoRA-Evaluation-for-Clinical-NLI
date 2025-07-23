#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Allowed labels
LABELS = ["eligible", "excluded"]


def setup_logging() -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TREC predictions and write metric summaries."
    )
    parser.add_argument(
        "--results_file",
        type=Path,
        required=True,
        help="Path to the predictions JSONL file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save metrics outputs."
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="metrics",
        help="Identifier for this run, used in output filenames."
    )
    return parser.parse_args()


def load_predictions(path: Path) -> List[Dict]:
    """Load JSONL predictions from a file."""
    records = []
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON on line {idx}: {e}")
    return records


def filter_valid(records: List[Dict]) -> List[Dict]:
    """Keep only entries with predicted_label in LABELS."""
    return [r for r in records if r.get('predicted_label') in LABELS]


def compute_confusion(y_true: List[str], y_pred: List[str]) -> Tuple[int, int, int, int]:
    """Compute TP, TN, FP, FN for binary labels."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == LABELS[0])
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == LABELS[1])
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == LABELS[0])
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == LABELS[1])
    return tp, tn, fp, fn


def write_outputs(
    metrics: Dict[str, float],
    confusion: Tuple[int, int, int, int],
    valid_fraction: float,
    none_count: int,
    total_count: int,
    output_dir: Path,
    prompt_name: str
) -> None:
    """Write metrics and confusion matrix to JSON and text outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / f"trec_{prompt_name}_metrics.json"
    summary = {
        **metrics,
        "tp": confusion[0],
        "tn": confusion[1],
        "fp": confusion[2],
        "fn": confusion[3],
        "valid_fraction": valid_fraction,
        "none_count": none_count,
        "total_count": total_count
    }
    with json_path.open('w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)
    logging.info(f"Written JSON metrics to {json_path}")

    # LaTeX-friendly text
    tex_lines = []
    # Summary row: F1 & Recall & Precision & Accuracy & ValidFrac
    tex_lines.append(
        f"{metrics['f1']:.3f} & {metrics['recall']:.3f}"
        f" & {metrics['precision']:.3f} & {metrics['accuracy']:.3f}"
        f" & {valid_fraction:.3f} \\"  
    )
    # Confusion
    tp, tn, fp, fn = confusion
    tex_lines.append(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\\")
    tex_lines.append(f"None predictions: {none_count} out of {total_count}\\")

    txt_path = output_dir / f"trec_{prompt_name}_metrics.txt"
    with txt_path.open('w', encoding='utf-8') as tf:
        tf.write("\n".join(tex_lines) + "\n")
    logging.info(f"Written LaTeX text metrics to {txt_path}")


def main() -> None:
    setup_logging()
    args = parse_args()

    records = load_predictions(args.results_file)
    total = len(records)
    valid_records = filter_valid(records)
    valid_count = len(valid_records)
    none_count = total - valid_count
    valid_fraction = valid_count / total if total > 0 else 0.0

    logging.info(f"Total records: {total}, valid: {valid_count}, filtered: {none_count}")

    # Extract true/pred lists
    y_true = [r['gold_label'] for r in valid_records]
    y_pred = [r['predicted_label'] for r in valid_records]

    # Compute metrics
    precision = precision_score(y_true, y_pred, pos_label=LABELS[0], zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=LABELS[0], zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    confusion = compute_confusion(y_true, y_pred)

    write_outputs(
        metrics,
        confusion,
        valid_fraction,
        none_count,
        total,
        args.output_dir,
        args.prompt_name
    )


if __name__ == '__main__':
    main()
