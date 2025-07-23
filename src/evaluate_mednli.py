#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

from sklearn.metrics import precision_score, recall_score, f1_score

# Allowed labels for evaluation
LABELS = ["entailment", "contradiction", "neutral"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedNLI predictions and write summary metrics."
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


def load_predictions(results_path: Path) -> List[Dict]:
    """Load JSONL predictions from file."""
    preds = []
    with results_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return preds


def filter_valid(preds: List[Dict]) -> List[Dict]:
    """Keep only entries with a predicted_label in known LABELS."""
    valid = [p for p in preds if p.get("predicted_label") in LABELS]
    return valid


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute classification metrics."""
    precision = precision_score(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def write_outputs(
    metrics: Dict[str, float],
    valid_fraction: float,
    none_count: int,
    total_count: int,
    output_dir: Path,
    prompt_name: str
) -> None:
    """Write metrics to JSON and plain-text (LaTeX row) files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_dir / f"mednli_{prompt_name}_metrics.json"
    summary = {
        **metrics,
        "valid_fraction": valid_fraction,
        "none_count": none_count,
        "total_count": total_count
    }
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    logging.info(f"Written JSON metrics to {json_path}")

    # LaTeX-style row (F1 & recall & precision & accuracy & valid_fraction)
    tex_row = (
        f"{metrics['f1']:.3f} & {metrics['recall']:.3f}"
        f" & {metrics['precision']:.3f} & {metrics['accuracy']:.3f}"
        f" & {valid_fraction:.3f} \\"  
    )
    txt_path = output_dir / f"mednli_{prompt_name}_metrics.txt"
    with txt_path.open("w", encoding="utf-8") as tf:
        tf.write(tex_row + "\n")
    logging.info(f"Written LaTeX metrics row to {txt_path}")


def main() -> None:
    setup_logging()
    args = parse_args()

    preds = load_predictions(args.results_file)
    total = len(preds)
    valid_preds = filter_valid(preds)
    valid_count = len(valid_preds)
    valid_fraction = valid_count / total if total > 0 else 0.0
    none_count = total - valid_count

    logging.info(f"Loaded {total} entries, {valid_count} valid, {none_count} filtered out.")

    y_true = [p["gold_label"] for p in valid_preds]
    y_pred = [p["predicted_label"] for p in valid_preds]

    metrics = compute_metrics(y_true, y_pred)
    logging.info(
        f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}"
    )

    write_outputs(
        metrics,
        valid_fraction,
        none_count,
        total,
        args.output_dir,
        args.prompt_name
    )


if __name__ == "__main__":
    main()
