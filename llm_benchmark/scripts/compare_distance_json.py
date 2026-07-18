"""Compare distance estimation results CSV against distance_labels.json ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from label_config import PROJECT_ROOT


DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "distance_labels.json"

DETAIL_FIELDS = (
    "image_id",
    "ground_truth_cm",
    "predicted_cm",
    "absolute_error_cm",
    "relative_error_pct",
    "within_5cm",
    "within_10cm",
    "within_20pct",
)


def load_labels(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels: Dict[str, float] = {}
    for image_key, annotations in raw.items():
        image_id = image_key.replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
        for ann in annotations:
            if ann.get("type") == "line":
                dist_str = ann.get("distances", {}).get("1-2", "")
                if dist_str:
                    labels[image_id] = float(dist_str)
    return labels


def load_csv_by_image_id(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {str(row.get("image_id", "")).strip(): dict(row) for row in csv.DictReader(f) if row.get("image_id")}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalize_bool(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return "true"
    if normalized in {"false", "0", "no", "n"}:
        return "false"
    return "unknown"


def pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 100.0, 2)


def evaluate_distance(
    labels: Dict[str, float],
    results: Dict[str, Dict[str, str]],
    matched_ids: List[str],
) -> Dict[str, Any]:
    pred_cms: List[float] = []
    gt_cms: List[float] = []
    missing_preds: List[str] = []

    for image_id in matched_ids:
        row = results[image_id]
        gt_cm = labels[image_id]
        pred_cm = parse_float(row.get("estimated_distance_cm"))
        if pred_cm is None:
            pred_m = parse_float(row.get("estimated_distance_m"))
            if pred_m is not None:
                pred_cm = pred_m * 100.0

        if pred_cm is None:
            missing_preds.append(image_id)
            continue

        pred_cms.append(pred_cm)
        gt_cms.append(gt_cm)

    if not pred_cms:
        return {
            "n": 0,
            "mae_cm": None,
            "mape_pct": None,
            "within_5cm_rate": None,
            "within_10cm_rate": None,
            "within_20pct_rate": None,
            "missing_prediction_count": len(missing_preds),
            "missing_predictions": missing_preds,
        }

    abs_errors = [abs(p - g) for p, g in zip(pred_cms, gt_cms)]
    rel_errors = [(abs(p - g) / g * 100.0) if g > 0 else 0.0 for p, g in zip(pred_cms, gt_cms)]

    metrics = {
        "n": len(pred_cms),
        "mae_cm": round(mean(abs_errors), 2),
        "mape_pct": round(mean(rel_errors), 2),
        "within_5cm_rate": pct(mean(1.0 if e <= 5.0 else 0.0 for e in abs_errors)),
        "within_10cm_rate": pct(mean(1.0 if e <= 10.0 else 0.0 for e in abs_errors)),
        "within_20pct_rate": pct(mean(1.0 if re <= 20.0 else 0.0 for re in rel_errors)),
        "missing_prediction_count": len(missing_preds),
        "missing_predictions": missing_preds,
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
    labels: Dict[str, float],
    results: Dict[str, Dict[str, str]],
    matched_ids: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_id in matched_ids:
        row = results[image_id]
        gt_cm = labels[image_id]
        pred_cm = parse_float(row.get("estimated_distance_cm"))
        if pred_cm is None:
            pred_m = parse_float(row.get("estimated_distance_m"))
            if pred_m is not None:
                pred_cm = pred_m * 100.0

        entry: Dict[str, Any] = {"image_id": image_id}

        if pred_cm is None:
            entry.update({
                "ground_truth_cm": f"{gt_cm:.1f}",
                "predicted_cm": "",
                "absolute_error_cm": "",
                "relative_error_pct": "",
                "within_5cm": "",
                "within_10cm": "",
                "within_20pct": "",
            })
        else:
            abs_error = abs(pred_cm - gt_cm)
            rel_error_pct = (abs_error / gt_cm * 100.0) if gt_cm > 0 else 0.0
            entry.update({
                "ground_truth_cm": f"{gt_cm:.1f}",
                "predicted_cm": f"{pred_cm:.1f}",
                "absolute_error_cm": f"{abs_error:.1f}",
                "relative_error_pct": f"{rel_error_pct:.2f}",
                "within_5cm": str(abs_error <= 5.0).lower(),
                "within_10cm": str(abs_error <= 10.0).lower(),
                "within_20pct": str(rel_error_pct <= 20.0).lower(),
            })
        rows.append(entry)
    return rows


def default_output_paths(results_path: Path) -> Tuple[Path, Path]:
    return (
        results_path.with_name("evaluation_summary.json"),
        results_path.with_name("evaluation_details.csv"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare distance results against distance_labels.json.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Path to distance_labels.json.")
    parser.add_argument("--results", required=True, help="Path to results.csv from a distance run.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--details-output", default=None, help="Optional per-image CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    results_path = Path(args.results).resolve()

    if not labels_path.exists():
        print(f"Labels not found: {labels_path}")
        return 1
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return 1

    labels = load_labels(labels_path)
    results = load_csv_by_image_id(results_path)

    matched_ids = sorted(set(labels) & set(results))
    missing_results = sorted(set(labels) - set(results))
    extra_results = sorted(set(results) - set(labels))

    summary_path, details_path = default_output_paths(results_path)
    if args.summary_output:
        summary_path = Path(args.summary_output).resolve()
    if args.details_output:
        details_path = Path(args.details_output).resolve()

    distance_metrics = evaluate_distance(labels, results, matched_ids)

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
        "distance_metrics": distance_metrics,
        "valid_json_rate": evaluate_valid_json(results),
        "avg_latency_ms": evaluate_latency(results),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(details_path, build_details(labels, results, matched_ids))

    print(f"Coverage: {len(matched_ids)}/{len(labels)} images matched ({summary['missing_result_count']} missing, {summary['extra_result_count']} extra)")
    dm = summary["distance_metrics"]
    print("── Distance Error ───────────────────────────────")
    if dm["n"]:
        print(f"  Samples:               {dm['n']}")
        print(f"  MAE:                   {dm['mae_cm']} cm")
        print(f"  MAPE:                  {dm['mape_pct']} %")
        print(f"  Within 5 cm:           {dm['within_5cm_rate']} %")
        print(f"  Within 10 cm:          {dm['within_10cm_rate']} %")
        print(f"  Within 20%:            {dm['within_20pct_rate']} %")
    else:
        print("  No valid predictions to evaluate.")
    print("── Quality ──────────────────────────────────────")
    if summary["valid_json_rate"] is not None:
        print(f"  Valid JSON rate:       {summary['valid_json_rate']} %")
    if summary["avg_latency_ms"] is not None:
        print(f"  Avg latency:           {summary['avg_latency_ms']} ms")
    print("── Files ────────────────────────────────────────")
    print(f"  Summary:  {summary_path}")
    print(f"  Details:  {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
