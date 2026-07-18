"""Run YOLO object detection + PPE detection as the local CV benchmark baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from label_config import IMAGE_EXTENSIONS, IMAGES_DIR, PROJECT_ROOT

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - dependency boundary
    raise SystemExit(
        "Ultralytics is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


ANZULIFT_ROOT = PROJECT_ROOT.parent / "AnzuLift"
DEFAULT_OBJECT_MODEL = ANZULIFT_ROOT / "models" / "detection" / "Model A.pt"
DEFAULT_PPE_MODEL = ANZULIFT_ROOT / "models" / "ppe" / "hansung-ppe.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "cv_yolo_ppe"

WORKER_ALIASES = ("person", "worker", "human")
FORKLIFT_ALIASES = ("forklift", "lift_truck", "lifttruck")
HELMET_POSITIVE_ALIASES = ("helmet", "hardhat", "hard_hat", "safety_helmet")
HELMET_NEGATIVE_ALIASES = ("nohelmet", "no_helmet", "no-hardhat", "no_hardhat", "withouthelmet")

CSV_FIELDS = (
    "run_id",
    "image_id",
    "image_path",
    "experiment",
    "provider",
    "object_model",
    "ppe_model",
    "worker_count",
    "forklift_count",
    "helmet_violation",
    "forklift_person_risk",
    "overall_risk",
    "object_detection_count",
    "ppe_detection_count",
    "latency_ms",
    "object_latency_ms",
    "ppe_latency_ms",
    "error",
)


def collect_images(images_dir: Path, limit: Optional[int] = None) -> List[Path]:
    images = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        return images[: max(0, limit)]
    return images


def image_id_for(path: Path, images_dir: Path) -> str:
    rel = path.relative_to(images_dir)
    return rel.with_suffix("").as_posix().replace("/", "__")


def normalize_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def class_matches(class_name: str, aliases: Iterable[str]) -> bool:
    normalized = normalize_name(class_name)
    normalized_aliases = [normalize_name(alias) for alias in aliases]
    return any(alias == normalized or alias in normalized for alias in normalized_aliases)


def run_model(
    model: YOLO,
    source: Any,
    *,
    conf: float,
    imgsz: int,
    max_det: int,
    device: str,
) -> Tuple[List[object], int]:
    start = time.perf_counter()
    kwargs: Dict[str, Any] = {
        "conf": conf,
        "imgsz": imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    if device != "auto":
        kwargs["device"] = device
    results = model(source, **kwargs)
    latency_ms = int(round((time.perf_counter() - start) * 1000))
    return list(results), latency_ms


def extract_detections(
    results: List[object],
    model: YOLO,
    *,
    offset: Tuple[int, int] = (0, 0),
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    ox, oy = offset
    names = getattr(model, "names", {}) or {}
    for result_index, result in enumerate(results):
        for box_index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, cls_id))
            conf_score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            gx1 = int(round(float(x1))) + ox
            gy1 = int(round(float(y1))) + oy
            gx2 = int(round(float(x2))) + ox
            gy2 = int(round(float(y2))) + oy
            detections.append(
                {
                    "result_index": result_index,
                    "box_index": box_index,
                    "class_id": cls_id,
                    "class": class_name,
                    "confidence": round(conf_score, 6),
                    "bbox": [gx1, gy1, gx2, gy2],
                }
            )
    return detections


def is_worker_detection(detection: Dict[str, Any]) -> bool:
    return class_matches(str(detection.get("class", "")), WORKER_ALIASES)


def is_forklift_detection(detection: Dict[str, Any]) -> bool:
    return class_matches(str(detection.get("class", "")), FORKLIFT_ALIASES)


def crop_detection(image: Image.Image, detection: Dict[str, Any]) -> Tuple[Optional[Image.Image], Tuple[int, int]]:
    width, height = image.size
    x1, y1, x2, y2 = [int(value) for value in detection.get("bbox", [0, 0, 0, 0])]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)
    return image.crop((x1, y1, x2, y2)), (x1, y1)


def run_ppe_on_worker_crops(
    *,
    ppe_model: YOLO,
    image_path: Path,
    worker_detections: List[Dict[str, Any]],
    conf: float,
    imgsz: int,
    max_det: int,
    device: str,
) -> Tuple[List[Dict[str, Any]], int]:
    image = Image.open(image_path).convert("RGB")
    all_ppe: List[Dict[str, Any]] = []
    total_latency_ms = 0
    for worker_index, detection in enumerate(worker_detections):
        crop, offset = crop_detection(image, detection)
        if crop is None:
            continue
        results, latency_ms = run_model(
            ppe_model,
            crop,
            conf=conf,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
        )
        total_latency_ms += latency_ms
        ppe_detections = extract_detections(results, ppe_model, offset=offset)
        for ppe_detection in ppe_detections:
            ppe_detection["source"] = "worker_crop"
            ppe_detection["worker_index"] = worker_index
        all_ppe.extend(ppe_detections)
    return all_ppe, total_latency_ms


def resolve_helmet_violation(ppe_detections: List[Dict[str, Any]]) -> str:
    class_names = [str(item.get("class", "")) for item in ppe_detections]

    has_no_helmet = any(class_matches(name, HELMET_NEGATIVE_ALIASES) for name in class_names)
    has_helmet = any(class_matches(name, HELMET_POSITIVE_ALIASES) for name in class_names)

    return "true" if has_no_helmet else "false" if has_helmet else "unknown"


def bbox_center(detection: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(detection: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def estimate_forklift_person_risk(
    *,
    forklifts: List[Dict[str, Any]],
    workers: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    threshold: float,
) -> str:
    if not forklifts or not workers:
        return "false"
    width, height = image_size
    diagonal = math.sqrt(float(width * width + height * height))
    for forklift in forklifts:
        fx, fy = bbox_center(forklift)
        fw, fh = bbox_size(forklift)
        for worker in workers:
            wx, wy = bbox_center(worker)
            distance = math.sqrt((fx - wx) ** 2 + (fy - wy) ** 2) / max(diagonal, 1.0)
            if distance <= threshold:
                return "true"
            x1, y1, x2, y2 = [float(value) for value in forklift.get("bbox", [0, 0, 0, 0])]
            expanded = [
                x1 - fw * 0.35,
                y1 - fh * 0.35,
                x2 + fw * 0.35,
                y2 + fh * 0.35,
            ]
            if expanded[0] <= wx <= expanded[2] and expanded[1] <= wy <= expanded[3]:
                return "true"
    return "false"


def resolve_overall_risk(
    *,
    helmet_violation: str,
    forklift_person_risk: str,
) -> str:
    if forklift_person_risk == "true" or helmet_violation == "true":
        return "warning"
    return "safe"


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection and PPE detection for the CV baseline."
    )
    parser.add_argument("--images-dir", default=str(IMAGES_DIR), help="Directory containing benchmark images.")
    parser.add_argument("--object-model", default=str(DEFAULT_OBJECT_MODEL), help="YOLO object model path.")
    parser.add_argument("--ppe-model", default=str(DEFAULT_PPE_MODEL), help="YOLO PPE model path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--device", default="auto", help="Ultralytics device: auto, cpu, 0, etc.")
    parser.add_argument("--object-conf", type=float, default=0.25, help="Object detection confidence threshold.")
    parser.add_argument("--ppe-conf", type=float, default=0.25, help="PPE detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per model call.")
    parser.add_argument(
        "--ppe-mode",
        choices=("full-frame", "worker-crops", "both"),
        default="worker-crops",
        help="Where to run the PPE model.",
    )
    parser.add_argument(
        "--risk-distance-threshold",
        type=float,
        default=0.25,
        help="Normalized center-distance threshold for simple forklift-person risk heuristic.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id for output files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    object_model_path = Path(args.object_model).resolve()
    ppe_model_path = Path(args.ppe_model).resolve()

    if not object_model_path.exists():
        print(f"Object model not found: {object_model_path}")
        return 1
    if not ppe_model_path.exists():
        print(f"PPE model not found: {ppe_model_path}")
        return 1

    images = collect_images(images_dir, args.limit)
    if not images:
        print(f"No supported images found in: {images_dir}")
        return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_cv_yolo_ppe")
    output_dir = RESULTS_DIR / run_id
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run id: {run_id}")
    print(f"Images: {len(images)}")
    print(f"Object model: {object_model_path}")
    print(f"PPE model: {ppe_model_path}")
    print(f"Output: {output_dir}")

    object_model = YOLO(str(object_model_path))
    ppe_model = YOLO(str(ppe_model_path))

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    rows: List[Dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        image_id = image_id_for(image_path, images_dir)
        print(f"[{index}/{len(images)}] {image_id}")
        start = time.perf_counter()
        error = ""
        object_latency_ms = 0
        ppe_latency_ms = 0
        object_detections: List[Dict[str, Any]] = []
        ppe_detections: List[Dict[str, Any]] = []

        try:
            object_results, object_latency_ms = run_model(
                object_model,
                str(image_path),
                conf=args.object_conf,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=args.device,
            )
            object_detections = extract_detections(object_results, object_model)
            workers = [item for item in object_detections if is_worker_detection(item)]
            forklifts = [item for item in object_detections if is_forklift_detection(item)]

            if args.ppe_mode in ("full-frame", "both"):
                ppe_results, full_frame_ppe_latency = run_model(
                    ppe_model,
                    str(image_path),
                    conf=args.ppe_conf,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    device=args.device,
                )
                ppe_latency_ms += full_frame_ppe_latency
                for detection in extract_detections(ppe_results, ppe_model):
                    detection["source"] = "full_frame"
                    ppe_detections.append(detection)

            if args.ppe_mode in ("worker-crops", "both"):
                crop_ppe, crop_ppe_latency = run_ppe_on_worker_crops(
                    ppe_model=ppe_model,
                    image_path=image_path,
                    worker_detections=workers,
                    conf=args.ppe_conf,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    device=args.device,
                )
                ppe_latency_ms += crop_ppe_latency
                ppe_detections.extend(crop_ppe)

            with Image.open(image_path) as image:
                image_size = image.size
            helmet_violation = resolve_helmet_violation(ppe_detections)
            forklift_person_risk = estimate_forklift_person_risk(
                forklifts=forklifts,
                workers=workers,
                image_size=image_size,
                threshold=args.risk_distance_threshold,
            )
            overall_risk = resolve_overall_risk(
                helmet_violation=helmet_violation,
                forklift_person_risk=forklift_person_risk,
            )
            worker_count = len(workers)
            forklift_count = len(forklifts)

        except Exception as exc:  # noqa: BLE001 - batch boundary should continue
            error = str(exc)
            print(f"  error: {error}")
            workers = []
            forklifts = []
            helmet_violation = "unknown"
            forklift_person_risk = "unknown"
            overall_risk = "warning"
            worker_count = 0
            forklift_count = 0

        latency_ms = int(round((time.perf_counter() - start) * 1000))
        raw_payload = {
            "image_id": image_id,
            "image_path": str(image_path),
            "object_detections": object_detections,
            "ppe_detections": ppe_detections,
            "workers": workers,
            "forklifts": forklifts,
        }
        (raw_dir / f"{image_id}.json").write_text(
            json.dumps(raw_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "run_id": run_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "experiment": "cv_yolo_ppe",
            "provider": "local",
            "object_model": object_model_path.name,
            "ppe_model": ppe_model_path.name,
            "worker_count": worker_count,
            "forklift_count": forklift_count,
            "helmet_violation": helmet_violation,
            "forklift_person_risk": forklift_person_risk,
            "overall_risk": overall_risk,
            "object_detection_count": len(object_detections),
            "ppe_detection_count": len(ppe_detections),
            "latency_ms": latency_ms,
            "object_latency_ms": object_latency_ms,
            "ppe_latency_ms": ppe_latency_ms,
            "error": error,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        write_csv(csv_path, rows)
        print(
            "  "
            f"workers={worker_count} forklifts={forklift_count} "
            f"helmet_violation={helmet_violation} risk={overall_risk} "
            f"latency_ms={latency_ms}"
        )

    print(f"Done. CSV: {csv_path}")
    print(f"Done. JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
