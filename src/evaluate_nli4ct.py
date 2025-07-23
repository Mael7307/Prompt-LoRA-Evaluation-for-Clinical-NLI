#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score

# fixed inference labels
default_labels = ["entailment", "contradiction"]
# task classes
task_classes = [
    "Numerical_Comp", "Numerical_Op", "Paraphrase",
    "Clinical", "Common sense", "Existence"
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NLI4CT predictions with per-class breakdown."
    )
    parser.add_argument(
        "--results_file", type=Path, required=True,
        help="Path to JSONL file of predictions."
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to save metric outputs."
    )
    parser.add_argument(
        "--prompt_template", type=str, required=True,
        help="Name of prompt template (used in filenames)."
    )
    parser.add_argument(
        "--model_type", type=str, required=True,
        help="Identifier for model type (Base/LoRa)."
    )
    return parser.parse_args()


def load_predictions(path: Path) -> List[Dict]:
    preds: List[Dict] = []
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Line {idx}: invalid JSON, skipping ({e})")
    return preds


def enrich_with_labels(preds: List[Dict]) -> None:
    """
    Load NLI4CT test split, map hypothesis to selected_labels,
    and attach to each prediction entry in-place.
    """
    ds = load_dataset("Mael7307/NLI4CT", split="test")
    mapping = {example['hypothesis']: example.get('selected_labels', [])
               for example in ds}
    for entry in preds:
        entry['selected_labels'] = mapping.get(entry.get('hypothesis', ''), [])


def aggregate_by_class(
    preds: List[Dict]
) -> Dict[str, Tuple[List[str], List[str], int, int]]:
    """
    For each task class, collect true/pred arrays and counts.
    Returns dict: class -> (y_true, y_pred, total, valid_count).
    """
    agg: Dict[str, Tuple[List[str], List[str], int, int]] = {}
    # initialize
    for cls in task_classes:
        agg[cls] = ([], [], 0, 0)

    for entry in preds:
        gold = entry.get('gold_label')
        pred = entry.get('predicted_label')
        is_valid = pred in default_labels
        labels = entry.get('selected_labels', [])
        for cls in labels:
            if cls in agg:
                y_true, y_pred, total, valid = agg[cls]
                total += 1
                if is_valid:
                    valid += 1
                    y_true.append(gold)
                    y_pred.append(pred)
                agg[cls] = (y_true, y_pred, total, valid)
    return agg


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    average: str = 'binary',
    pos_label: str = 'entailment'
) -> Dict[str, float]:
    if not y_true:
        return {k: 0.0 for k in ('precision', 'recall', 'f1', 'accuracy')}
    precision = precision_score(
        y_true, y_pred, labels=labels,
        average=average, pos_label=pos_label, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=labels,
        average=average, pos_label=pos_label, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, labels=labels,
        average='macro', zero_division=0
    )
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def write_outputs(
    overall: Dict[str, float],
    valid_frac: float,
    none_count: int,
    class_metrics: Dict[str, Dict[str, float]],
    out_dir: Path,
    meta: argparse.Namespace
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    summary = {
        **overall,
        'valid_fraction': valid_frac,
        'none_count': none_count
    }
    json_path = out_dir / f"nli4ct_{meta.prompt_template}_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logging.info(f"Wrote JSON summary: {json_path}")

    # LaTeX table
    lines = []
    lines.append("% Overall metrics: F1 & Recall & Precision & Accuracy & ValidFrac")
    lines.append(
        f"{overall['f1']:.3f} & {overall['recall']:.3f}"
        f" & {overall['precision']:.3f} & {overall['accuracy']:.3f} & {valid_frac:.3f} \\\""
    )
    lines.append('')
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule Class & F1 & Recall & Precision & Accuracy & ValidFrac \\")
    lines.append(r"\midrule")
    for cls, metrics in class_metrics.items():
        lines.append(
            f"{cls} & {metrics['f1']:.3f} & {metrics['recall']:.3f}"
            f" & {metrics['precision']:.3f} & {metrics['accuracy']:.3f}"
            f" & {metrics['valid_fraction']:.3f} \\\""
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = out_dir / f"nli4ct_{meta.prompt_template}_metrics.tex"
    tex_path.write_text("\n".join(lines), encoding='utf-8')
    logging.info(f"Wrote LaTeX table: {tex_path}")


def main() -> None:
    setup_logging()
    args = parse_args()

    preds = load_predictions(args.results_file)
    total = len(preds)
    valid_preds = [p for p in preds if p.get('predicted_label') in default_labels]
    valid = len(valid_preds)
    none_count = total - valid
    valid_frac = valid / total if total else 0.0

    logging.info(f"Total entries: {total}, valid: {valid}, none: {none_count}")

    enrich_with_labels(preds)
    class_data = aggregate_by_class(preds)

    # compute per-class metrics
    class_metrics: Dict[str, Dict[str, float]] = {}
    for cls, (yt, yp, total_cls, valid_cls) in class_data.items():
        base = compute_metrics(yt, yp, default_labels)
        base['valid_fraction'] = valid_cls / total_cls if total_cls else 0.0
        class_metrics[cls] = base

    # overall metrics on valid only
    y_true = [p['gold_label'] for p in valid_preds]
    y_pred = [p['predicted_label'] for p in valid_preds]
    overall = compute_metrics(y_true, y_pred, default_labels)

    write_outputs(overall, valid_frac, none_count, class_metrics, args.output_dir, args)


if __name__ == '__main__':
    main()
