"""Compare distance-estimation results against distance labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from label_config import PROJECT_ROOT


DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "distance_labels.csv"

DETAIL_FIELDS = (
    "image_id",
    "target_pair",
    "ground_truth_distance_m",
    "predicted_distance_m",
    "absolute_error_m",
    "relative_error_pct",
    "within_0_5m",
    "within_1_0m",
    "within_20pct",
    "ground_truth_risk",
    "predicted_risk",
    "risk_match",
    "distance_source",
    "notes",
)


def load_csv_by_image_id(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {str(row.get("image_id", "")).strip(): row for row in rows if row.get("image_id")}


def parse_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_distance_m(row: Optional[Dict[str, str]], meter_field: str, centimeter_field: str) -> Optional[float]:
    if not row:
        return None
    meters = parse_float(row.get(meter_field))
    if meters is not None:
        return meters
    centimeters = parse_float(row.get(centimeter_field))
    if centimeters is not None:
        return centimeters / 100.0
    return None


def risk_from_distance(distance_m: float) -> str:
    if distance_m < 1.0:
        return "danger"
    if distance_m < 2.0:
        return "warning"
    return "safe"


def bool_string(value: bool) -> str:
    return str(value).lower()


def compare(labels_path: Path, results_path: Path) -> List[Dict[str, Any]]:
    labels = load_csv_by_image_id(labels_path)
    results = load_csv_by_image_id(results_path)
    details: List[Dict[str, Any]] = []

    for image_id, label in labels.items():
        result = results.get(image_id)
        ground_truth = parse_distance_m(label, "ground_truth_distance_m", "ground_truth_distance_cm")
        predicted = parse_distance_m(result, "estimated_distance_m", "estimated_distance_cm")
        ground_truth_risk = label.get("risk_label", "").strip().lower()
        predicted_risk = str(result.get("distance_risk", "")).strip().lower() if result else ""

        row: Dict[str, Any] = {
            "image_id": image_id,
            "target_pair": label.get("target_pair", ""),
            "ground_truth_distance_m": "" if ground_truth is None else f"{ground_truth:.3f}".rstrip("0").rstrip("."),
            "predicted_distance_m": "" if predicted is None else predicted,
            "absolute_error_m": "",
            "relative_error_pct": "",
            "within_0_5m": "",
            "within_1_0m": "",
            "within_20pct": "",
            "ground_truth_risk": ground_truth_risk,
            "predicted_risk": predicted_risk,
            "risk_match": "",
            "distance_source": label.get("distance_source", ""),
            "notes": label.get("notes", ""),
        }

        if ground_truth is None or predicted is None:
            details.append(row)
            continue

        absolute_error = abs(predicted - ground_truth)
        relative_error_pct = (absolute_error / ground_truth * 100.0) if ground_truth > 0 else 0.0

        row.update(
            {
                "predicted_distance_m": f"{predicted:.3f}".rstrip("0").rstrip("."),
                "absolute_error_m": f"{absolute_error:.3f}",
                "relative_error_pct": f"{relative_error_pct:.2f}",
                "within_0_5m": bool_string(absolute_error <= 0.5),
                "within_1_0m": bool_string(absolute_error <= 1.0),
                "within_20pct": bool_string(relative_error_pct <= 20.0),
                "ground_truth_risk": ground_truth_risk,
                "predicted_risk": predicted_risk,
                "risk_match": bool_string(ground_truth_risk == predicted_risk)
                if ground_truth_risk and predicted_risk
                else "",
            }
        )
        details.append(row)

    return details


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in details if row.get("absolute_error_m") != ""]
    errors = [float(row["absolute_error_m"]) for row in valid_rows]
    rel_errors = [float(row["relative_error_pct"]) for row in valid_rows]

    def rate(field: str) -> float:
        if not valid_rows:
            return 0.0
        return mean(1.0 if row.get(field) == "true" else 0.0 for row in valid_rows)

    risk_rows = [
        row
        for row in valid_rows
        if str(row.get("ground_truth_risk", "")).strip()
        and str(row.get("predicted_risk", "")).strip()
    ]
    risk_accuracy = (
        round(mean(1.0 if row.get("risk_match") == "true" else 0.0 for row in risk_rows), 3)
        if risk_rows
        else None
    )

    return {
        "label_count": len(details),
        "matched_prediction_count": len(valid_rows),
        "missing_prediction_count": len(details) - len(valid_rows),
        "mae_m": round(mean(errors), 3) if errors else None,
        "mape_pct": round(mean(rel_errors), 2) if rel_errors else None,
        "within_0_5m_accuracy": round(rate("within_0_5m"), 3),
        "within_1_0m_accuracy": round(rate("within_1_0m"), 3),
        "within_20pct_accuracy": round(rate("within_20pct"), 3),
        "risk_accuracy": risk_accuracy,
    }


def write_details(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DETAIL_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distance-estimation results.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Distance labels CSV.")
    parser.add_argument("--results", required=True, help="VLM/CV distance results CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation outputs.")
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

    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    details = compare(labels_path, results_path)
    summary = summarize(details)

    details_path = output_dir / "distance_evaluation_details.csv"
    summary_path = output_dir / "distance_evaluation_summary.json"
    write_details(details_path, details)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
