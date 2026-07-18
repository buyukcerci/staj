"""Compare benchmark result CSV files against manual labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from label_config import CSV_PATH


COUNT_FIELDS = ("worker_count", "forklift_count")
CATEGORICAL_FIELDS = ("helmet_violation", "forklift_person_risk", "overall_risk")
POSITIVE_VALUES = {
    "helmet_violation": "true",
    "forklift_person_risk": "true",
    "overall_risk": "warning",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_bool(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1", "y"}:
        return "true"
    if normalized in {"false", "no", "0", "n"}:
        return "false"
    return "unknown"


def normalize_risk(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"safe", "warning"}:
        return normalized
    if normalized in {"risky", "danger", "dangerous"}:
        return "warning"
    return "unknown"


def normalize_category(field: str, value: Any) -> str:
    if field == "overall_risk":
        return normalize_risk(value)
    return normalize_bool(value)


def parse_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def f1_from_counts(tp: int, fp: int, fn: int) -> Optional[float]:
    denominator = (2 * tp) + fp + fn
    if denominator == 0:
        return None
    return (2 * tp) / denominator


def precision_from_counts(tp: int, fp: int) -> Optional[float]:
    denominator = tp + fp
    if denominator == 0:
        return None
    return tp / denominator


def recall_from_counts(tp: int, fn: int) -> Optional[float]:
    denominator = tp + fn
    if denominator == 0:
        return None
    return tp / denominator


def pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100.0, 2)


def evaluate_counts(
    labels: Dict[str, Dict[str, str]],
    results: Dict[str, Dict[str, str]],
    matched_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for field in COUNT_FIELDS:
        diffs: List[int] = []
        exact = 0
        for image_id in matched_ids:
            expected = parse_int(labels[image_id].get(field))
            predicted = parse_int(results[image_id].get(field))
            if expected is None or predicted is None:
                continue
            diff = abs(predicted - expected)
            diffs.append(diff)
            if diff == 0:
                exact += 1

        if diffs:
            metrics[field] = {
                "n": len(diffs),
                "mae": round(mean(diffs), 4),
                "exact_accuracy": pct(exact / len(diffs)),
            }
        else:
            metrics[field] = {"n": 0, "mae": None, "exact_accuracy": None}
    return metrics


def evaluate_categories(
    labels: Dict[str, Dict[str, str]],
    results: Dict[str, Dict[str, str]],
    matched_ids: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    result_fields = set(next(iter(results.values())).keys()) if results else set()
    for field in CATEGORICAL_FIELDS:
        if field not in result_fields:
            continue

        positive = POSITIVE_VALUES[field]
        tp = fp = tn = fn = exact = n = 0
        for image_id in matched_ids:
            expected = normalize_category(field, labels[image_id].get(field))
            predicted = normalize_category(field, results[image_id].get(field))
            if expected == "unknown" or predicted == "unknown":
                continue

            n += 1
            if expected == predicted:
                exact += 1

            expected_positive = expected == positive
            predicted_positive = predicted == positive
            if expected_positive and predicted_positive:
                tp += 1
            elif not expected_positive and predicted_positive:
                fp += 1
            elif expected_positive and not predicted_positive:
                fn += 1
            else:
                tn += 1

        precision = precision_from_counts(tp, fp)
        recall = recall_from_counts(tp, fn)
        f1 = f1_from_counts(tp, fp, fn)
        metrics[field] = {
            "n": n,
            "accuracy": pct(exact / n) if n else None,
            "positive_label": positive,
            "positive_precision": pct(precision),
            "positive_recall": pct(recall),
            "positive_f1": pct(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    return metrics


def evaluate_valid_json(results: Dict[str, Dict[str, str]]) -> Optional[float]:
    values = [normalize_bool(row.get("valid_json")) for row in results.values() if row.get("valid_json") != ""]
    if not values:
        return None
    return pct(sum(value == "true" for value in values) / len(values))


def evaluate_latency(results: Dict[str, Dict[str, str]]) -> Optional[float]:
    values = [
        value
        for value in (parse_float(row.get("latency_ms")) for row in results.values())
        if value is not None
    ]
    if not values:
        return None
    return round(mean(values), 2)


def build_details(
    labels: Dict[str, Dict[str, str]],
    results: Dict[str, Dict[str, str]],
    matched_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    result_fields = set(next(iter(results.values())).keys()) if results else set()
    fields = list(COUNT_FIELDS) + [field for field in CATEGORICAL_FIELDS if field in result_fields]

    for image_id in matched_ids:
        row: Dict[str, Any] = {"image_id": image_id}
        for field in fields:
            expected = labels[image_id].get(field, "")
            predicted = results[image_id].get(field, "")
            if field in COUNT_FIELDS:
                expected_norm = parse_int(expected)
                predicted_norm = parse_int(predicted)
            else:
                expected_norm = normalize_category(field, expected)
                predicted_norm = normalize_category(field, predicted)
            row[f"label_{field}"] = expected
            row[f"pred_{field}"] = predicted
            row[f"{field}_match"] = str(expected_norm == predicted_norm).lower()
        rows.append(row)
    return rows


def index_by_image_id(rows: List[Dict[str, str]], source_name: str) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        if image_id in indexed:
            raise ValueError(f"Duplicate image_id in {source_name}: {image_id}")
        indexed[image_id] = row
    return indexed


def default_output_paths(results_path: Path) -> Tuple[Path, Path]:
    return (
        results_path.with_name("evaluation_summary.json"),
        results_path.with_name("evaluation_details.csv"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results.csv against manual label CSV."
    )
    parser.add_argument("--results", required=True, help="Path to a benchmark results.csv file.")
    parser.add_argument("--labels", default=str(CSV_PATH), help="Path to manual_labels.csv.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--details-output", default=None, help="Optional per-image CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    results_path = Path(args.results).resolve()

    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return 1
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return 1

    labels = index_by_image_id(read_csv(labels_path), "labels")
    results = index_by_image_id(read_csv(results_path), "results")

    matched_ids = sorted(set(labels) & set(results))
    missing_results = sorted(set(labels) - set(results))
    extra_results = sorted(set(results) - set(labels))

    summary_path, details_path = default_output_paths(results_path)
    if args.summary_output:
        summary_path = Path(args.summary_output).resolve()
    if args.details_output:
        details_path = Path(args.details_output).resolve()

    summary = {
        "labels_path": str(labels_path),
        "results_path": str(results_path),
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

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(details_path, build_details(labels, results, matched_ids))

    print(f"Matched images: {len(matched_ids)}/{len(labels)} labels")
    for field, metrics in summary["count_metrics"].items():
        print(
            f"{field}: n={metrics['n']} mae={metrics['mae']} "
            f"exact_acc={metrics['exact_accuracy']}"
        )
    for field, metrics in summary["categorical_metrics"].items():
        print(
            f"{field}: n={metrics['n']} acc={metrics['accuracy']} "
            f"precision({metrics['positive_label']})={metrics['positive_precision']} "
            f"recall({metrics['positive_label']})={metrics['positive_recall']} "
            f"f1({metrics['positive_label']})={metrics['positive_f1']}"
        )
    if summary["valid_json_rate"] is not None:
        print(f"valid_json_rate={summary['valid_json_rate']}")
    if summary["avg_latency_ms"] is not None:
        print(f"avg_latency_ms={summary['avg_latency_ms']}")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
