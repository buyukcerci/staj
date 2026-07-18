"""Compare YOLO and Ollama evaluation metrics against manual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from compare_results_with_labels import (
    build_details,
    default_output_paths,
    evaluate_categories,
    evaluate_counts,
    evaluate_latency,
    evaluate_valid_json,
    index_by_image_id,
    read_csv,
    write_csv,
)
from label_config import CSV_PATH, PROJECT_ROOT


BENCHMARKS = {
    "yolo": PROJECT_ROOT / "results" / "cv_yolo_ppe",
    "ollama": PROJECT_ROOT / "results" / "ollama_raw_vlm",
}
DEFAULT_METRICS_OUTPUT = PROJECT_ROOT / "results" / "combined_evaluation_metrics.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "combined_evaluation_summary.json"


def latest_results_csv(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.glob("*/results.csv")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No results.csv files found under {results_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def flatten_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = {
        "benchmark": summary["benchmark"],
        "run_id": summary["run_id"],
        "results_path": summary["results_path"],
        "label_count": summary["label_count"],
        "result_count": summary["result_count"],
        "matched_count": summary["matched_count"],
    }
    rows: List[Dict[str, Any]] = []

    for field, metrics in summary["count_metrics"].items():
        rows.append(
            {
                **base,
                "field": field,
                "metric_type": "count",
                "n": metrics["n"],
                "mae": metrics["mae"],
                "accuracy": metrics["exact_accuracy"],
                "precision": "",
                "recall": "",
                "f1": "",
                "tp": "",
                "fp": "",
                "tn": "",
                "fn": "",
            }
        )

    for field, metrics in summary["categorical_metrics"].items():
        rows.append(
            {
                **base,
                "field": field,
                "metric_type": "categorical",
                "n": metrics["n"],
                "mae": "",
                "accuracy": metrics["accuracy"],
                "precision": metrics["positive_precision"],
                "recall": metrics["positive_recall"],
                "f1": metrics["positive_f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            }
        )

    if summary["valid_json_rate"] is not None:
        rows.append(
            {
                **base,
                "field": "valid_json",
                "metric_type": "quality",
                "n": summary["result_count"],
                "mae": "",
                "accuracy": summary["valid_json_rate"],
                "precision": "",
                "recall": "",
                "f1": "",
                "tp": "",
                "fp": "",
                "tn": "",
                "fn": "",
            }
        )

    if summary["avg_latency_ms"] is not None:
        rows.append(
            {
                **base,
                "field": "latency_ms",
                "metric_type": "performance",
                "n": summary["result_count"],
                "mae": summary["avg_latency_ms"],
                "accuracy": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "tp": "",
                "fp": "",
                "tn": "",
                "fn": "",
            }
        )

    return rows


def compare_one(
    benchmark: str,
    labels_path: Path,
    labels: Dict[str, Dict[str, str]],
    results_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results = index_by_image_id(read_csv(results_path), f"{benchmark} results")

    matched_ids = sorted(set(labels) & set(results))
    missing_results = sorted(set(labels) - set(results))
    extra_results = sorted(set(results) - set(labels))

    summary = {
        "benchmark": benchmark,
        "labels_path": str(labels_path),
        "results_path": str(results_path),
        "run_id": results_path.parent.name,
        "label_count": len(labels),
        "result_count": len(results),
        "matched_count": len(matched_ids),
        "missing_result_count": len(missing_results),
        "extra_result_count": len(extra_results),
        "missing_results": missing_results,
        "extra_results": extra_results,
        "count_metrics": evaluate_counts(labels, results, matched_ids),
        "categorical_metrics": evaluate_categories(labels, results, matched_ids),
        "valid_json_rate": evaluate_valid_json(results),
        "avg_latency_ms": evaluate_latency(results),
    }
    details = build_details(labels, results, matched_ids)
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO and Ollama metrics against manual labels.")
    parser.add_argument("--labels", default=str(CSV_PATH), help="Path to manual_labels.csv.")
    parser.add_argument(
        "--yolo-results",
        default=None,
        help="Path to YOLO results.csv. Defaults to the newest cv_yolo_ppe run.",
    )
    parser.add_argument(
        "--ollama-results",
        default=None,
        help="Path to Ollama results.csv. Defaults to the newest ollama_raw_vlm run.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(DEFAULT_METRICS_OUTPUT),
        help="Combined metrics CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Combined summary JSON output path.",
    )
    parser.add_argument(
        "--write-individual",
        action="store_true",
        help="Also write evaluation_summary.json and evaluation_details.csv beside each results.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return 1

    requested_results = {
        "yolo": Path(args.yolo_results).resolve() if args.yolo_results else latest_results_csv(BENCHMARKS["yolo"]),
        "ollama": Path(args.ollama_results).resolve() if args.ollama_results else latest_results_csv(BENCHMARKS["ollama"]),
    }
    for benchmark, results_path in requested_results.items():
        if not results_path.exists():
            print(f"{benchmark} results file not found: {results_path}")
            return 1

    labels = index_by_image_id(read_csv(labels_path), "labels")
    summaries: List[Dict[str, Any]] = []
    combined_metrics: List[Dict[str, Any]] = []

    for benchmark, results_path in requested_results.items():
        summary, details = compare_one(benchmark, labels_path, labels, results_path)
        summaries.append(summary)
        combined_metrics.extend(flatten_summary(summary))

        if args.write_individual:
            summary_path, details_path = default_output_paths(results_path)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            write_csv(details_path, details)

        print(
            f"{benchmark}: matched {summary['matched_count']}/{summary['label_count']} labels "
            f"from {results_path}"
        )

    metrics_output = Path(args.metrics_output).resolve()
    summary_output = Path(args.summary_output).resolve()
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_output, combined_metrics)
    summary_output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"Combined metrics: {metrics_output}")
    print(f"Combined summary: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
