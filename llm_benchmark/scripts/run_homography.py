"""Interactive homography-based distance benchmark runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from label_config import IMAGE_EXTENSIONS, PROJECT_ROOT


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "distance_images"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "labels" / "distance_labels.csv"
RESULTS_ROOT = PROJECT_ROOT / "results" / "homography_distance"

CSV_FIELDS = (
    "run_id",
    "image_id",
    "image_path",
    "experiment",
    "provider",
    "model",
    "prompt_version",
    "valid_json",
    "target_pair",
    "estimated_distance_m",
    "estimated_distance_cm",
    "distance_confidence",
    "reference_used",
    "reason",
    "latency_ms",
    "ref_width_m",
    "ref_height_m",
    "ref_points",
    "point_a",
    "point_b",
    "error",
)


@dataclass
class Calibration:
    points: List[Point]
    width_m: float
    height_m: float
    homography: np.ndarray


class MeasurementEngine:
    """Minimal image-pixel to metric-ground-plane projection helper."""

    horizon_w_threshold = 1e-5

    def __init__(self) -> None:
        self.calibration: Optional[Calibration] = None

    def set_reference(self, points: List[Point], width_m: float, height_m: float) -> None:
        points = self.validate_quad(points)
        width = self.validate_dimension(width_m, "width")
        height = self.validate_dimension(height_m, "height")
        src = np.array(points, dtype=np.float32)
        dst = np.array(
            [
                [0.0, 0.0],
                [width, 0.0],
                [width, height],
                [0.0, height],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src, dst)
        if not np.isfinite(homography).all():
            raise ValueError("Could not compute a finite homography.")
        self.calibration = Calibration(points, width, height, homography.astype(np.float64))

    def project_points(self, points: List[Point]) -> List[FloatPoint]:
        if self.calibration is None:
            raise ValueError("Reference has not been calibrated.")
        src = np.array(points, dtype=np.float64)
        src_h = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
        mapped = (self.calibration.homography @ src_h.T).T
        w = mapped[:, 2]
        if np.any(np.abs(w) <= self.horizon_w_threshold):
            raise ValueError("Selected point is too close to the projection horizon.")
        xy = mapped[:, :2] / w[:, np.newaxis]
        if not np.isfinite(xy).all():
            raise ValueError("Projected point is non-finite.")
        return [(float(x), float(y)) for x, y in xy]

    def point_distance(self, point_a: Point, point_b: Point) -> float:
        projected_a, projected_b = self.project_points([point_a, point_b])
        return float(np.hypot(projected_b[0] - projected_a[0], projected_b[1] - projected_a[1]))

    @staticmethod
    def validate_dimension(value: float, label: str) -> float:
        dimension = float(value)
        if not np.isfinite(dimension) or dimension <= 0:
            raise ValueError(f"Reference {label} must be positive and finite.")
        return dimension

    @staticmethod
    def validate_quad(points: List[Point]) -> List[Point]:
        if len(points) != 4:
            raise ValueError("Exactly four reference points are required.")
        pts = np.array(points, dtype=np.float64)
        if np.unique(pts, axis=0).shape[0] != 4:
            raise ValueError("Reference points must be distinct.")
        if not MeasurementEngine.is_convex_quad(pts):
            raise ValueError("Reference points must form a convex TL, TR, BR, BL quadrilateral.")
        return [(int(round(x)), int(round(y))) for x, y in pts]

    @staticmethod
    def is_convex_quad(points: np.ndarray) -> bool:
        signs = []
        for index in range(4):
            a = points[index]
            b = points[(index + 1) % 4]
            c = points[(index + 2) % 4]
            ab = b - a
            bc = c - b
            cross = float((ab[0] * bc[1]) - (ab[1] * bc[0]))
            if abs(cross) <= 1e-6:
                return False
            signs.append(cross > 0.0)
        return all(sign == signs[0] for sign in signs[1:])


class PointPicker:
    """Small OpenCV click collector."""

    def __init__(self, image: np.ndarray, image_id: str) -> None:
        self.original = image
        self.image_id = image_id
        self.points: List[Point] = []
        self.hover: Optional[Point] = None
        self.expected_count = 0
        self.window_name = f"Homography Distance: {image_id}"
        self.max_display_side = 1600
        self.scale = min(1.0, self.max_display_side / max(image.shape[:2]))
        self.display_width = max(900, int(round(image.shape[1] * self.scale)))
        self.display_height = max(700, int(round(image.shape[0] * self.scale)))

    def collect(self, count: int, title: str, labels: List[str]) -> Optional[List[Point]]:
        self.points = []
        self.hover = None
        self.expected_count = count
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.handle_mouse)
        print(title)
        print("  Left click points. Backspace/U undo. C clear. Enter/Space confirm. Q/Esc skip image.")

        while True:
            frame = self.draw_frame(count=count, labels=labels, title=title)
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                cv2.destroyWindow(self.window_name)
                return None
            if key in (8, ord("u")) and self.points:
                self.points.pop()
            if key == ord("c"):
                self.points.clear()
            if key in (13, 32):
                if len(self.points) == count:
                    return self.points.copy()
                print(f"Need {count - len(self.points)} more point(s).")

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        point = (int(round(x / self.scale)), int(round(y / self.scale)))
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = point
        elif event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.expected_count:
            self.points.append(point)

    def draw_frame(self, *, count: int, labels: List[str], title: str) -> np.ndarray:
        display = cv2.resize(
            self.original,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )
        frame = display.copy()
        scaled_points = [(int(x * self.scale), int(y * self.scale)) for x, y in self.points]

        for index in range(len(scaled_points) - 1):
            cv2.line(frame, scaled_points[index], scaled_points[index + 1], (0, 220, 255), 2)
        if count == 4 and len(scaled_points) == 4:
            cv2.line(frame, scaled_points[-1], scaled_points[0], (0, 220, 255), 2)
        if count == 2 and len(scaled_points) == 2:
            cv2.line(frame, scaled_points[0], scaled_points[1], (0, 255, 0), 2)
        if self.hover is not None and 0 < len(scaled_points) < count:
            hover_point = (int(self.hover[0] * self.scale), int(self.hover[1] * self.scale))
            draw_dashed_line(frame, scaled_points[-1], hover_point, (0, 255, 255), 2)

        for index, point in enumerate(scaled_points):
            label = labels[index] if index < len(labels) else str(index + 1)
            cv2.circle(frame, point, 7, (0, 255, 255), -1)
            cv2.putText(frame, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status = f"{title} | {len(self.points)}/{count}"
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 760), 42), (0, 0, 0), -1)
        cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame


def draw_dashed_line(
    frame: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    *,
    dash_length: int = 12,
    gap_length: int = 8,
) -> None:
    delta = np.array([end[0] - start[0], end[1] - start[1]], dtype=np.float64)
    length = float(np.hypot(delta[0], delta[1]))
    if length <= 1.0:
        return
    direction = delta / length
    cursor = 0.0
    while cursor < length:
        segment_start = np.array(start, dtype=np.float64) + direction * cursor
        segment_end = np.array(start, dtype=np.float64) + direction * min(cursor + dash_length, length)
        cv2.line(
            frame,
            tuple(np.round(segment_start).astype(int)),
            tuple(np.round(segment_end).astype(int)),
            color,
            thickness,
        )
        cursor += dash_length + gap_length


def collect_images(images_dir: Path, limit: Optional[int]) -> List[Path]:
    images = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[: max(0, limit)] if limit is not None else images


def image_id_for(path: Path, images_dir: Path) -> str:
    rel = path.relative_to(images_dir)
    return rel.with_suffix("").as_posix().replace("/", "__")


def load_labels(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: dict(row) for row in csv.DictReader(handle) if row.get("image_id")}


def prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Enter a positive number.")


def to_meters(value: float, units: str) -> float:
    if units == "cm":
        return value / 100.0
    return value


def format_distance(distance_m: float, units: str) -> str:
    if units == "cm":
        return f"{distance_m * 100.0:.1f} cm"
    return f"{distance_m:.3f} m"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive homography distance measurement.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Directory containing benchmark images.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Optional distance labels CSV for target names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--run-id", default=None, help="Optional run id.")
    parser.add_argument("--resume", action="store_true", help="Skip image_ids already present in results.csv.")
    parser.add_argument(
        "--units",
        choices=("cm", "m"),
        default="cm",
        help="Units used when entering reference dimensions. Results CSV still stores meters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    labels_path = Path(args.labels).resolve()
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return 1

    images = collect_images(images_dir, args.limit)
    if not images:
        print(f"No supported images found in: {images_dir}")
        return 1

    labels = load_labels(labels_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_homography_distance")
    output_dir = RESULTS_ROOT / run_id
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    rows = load_existing_rows(csv_path) if args.resume else []
    completed_ids = {row.get("image_id", "") for row in rows if row.get("image_id")}

    print(f"Run id: {run_id}")
    print(f"Images: {len(images)}")
    print(f"Input units: {args.units}")
    print(f"Output: {output_dir}")

    for index, image_path in enumerate(images, start=1):
        image_id = image_id_for(image_path, images_dir)
        if args.resume and image_id in completed_ids:
            print(f"[{index}/{len(images)}] {image_id}: skipped")
            continue

        print(f"[{index}/{len(images)}] {image_id}")
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append(build_error_row(run_id, image_id, image_path, "Could not load image."))
            write_csv(csv_path, rows)
            continue

        label = labels.get(image_id, {})
        target_pair = label.get("target_pair", "point A to point B")
        error = ""
        distance_m: Optional[float] = None
        ref_points: Optional[List[Point]] = None
        point_a: Optional[Point] = None
        point_b: Optional[Point] = None
        ref_width_m = ""
        ref_height_m = ""

        try:
            picker = PointPicker(image, image_id)
            ref_points = picker.collect(4, "Select reference corners: TL, TR, BR, BL", ["TL", "TR", "BR", "BL"])
            if ref_points is None:
                print("  skipped")
                continue

            ref_width_input = prompt_float(f"Reference width, edge TL-TR, in {args.units}: ")
            ref_height_input = prompt_float(f"Reference height, edge TR-BR, in {args.units}: ")
            ref_width = to_meters(ref_width_input, args.units)
            ref_height = to_meters(ref_height_input, args.units)
            ref_width_m = f"{ref_width:.3f}".rstrip("0").rstrip(".")
            ref_height_m = f"{ref_height:.3f}".rstrip("0").rstrip(".")

            engine = MeasurementEngine()
            engine.set_reference(ref_points, ref_width, ref_height)

            target_points = picker.collect(2, f"Select distance points for {target_pair}: A then B", ["A", "B"])
            cv2.destroyWindow(picker.window_name)
            if target_points is None:
                print("  skipped")
                continue
            point_a, point_b = target_points
            distance_m = engine.point_distance(point_a, point_b)

            raw_payload = {
                "image_id": image_id,
                "image_path": str(image_path),
                "target_pair": target_pair,
                "ref_points": ref_points,
                "ref_width_m": ref_width,
                "ref_height_m": ref_height,
                "homography": engine.calibration.homography.tolist() if engine.calibration else None,
                "point_a": point_a,
                "point_b": point_b,
                "estimated_distance_m": distance_m,
            }
            (raw_dir / f"{image_id}.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - keep batch workflow alive
            error = str(exc)
            print(f"  error: {error}")

        row = {
            "run_id": run_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "experiment": "homography_distance_estimation",
            "provider": "cv",
            "model": "manual_homography",
            "prompt_version": "",
            "valid_json": "",
            "target_pair": target_pair,
            "estimated_distance_m": "" if distance_m is None else f"{distance_m:.3f}",
            "estimated_distance_cm": "" if distance_m is None else f"{distance_m * 100.0:.1f}",
            "distance_confidence": "high" if not error and distance_m is not None else "",
            "reference_used": "manual four-point ground-plane reference",
            "reason": "Computed from manual reference homography.",
            "latency_ms": "",
            "ref_width_m": ref_width_m,
            "ref_height_m": ref_height_m,
            "ref_points": json.dumps(ref_points) if ref_points is not None else "",
            "point_a": json.dumps(point_a) if point_a is not None else "",
            "point_b": json.dumps(point_b) if point_b is not None else "",
            "error": error,
        }
        rows.append(row)
        write_csv(csv_path, rows)
        if distance_m is not None:
            print(
                f"  distance={format_distance(distance_m, args.units)} "
                f"distance_m={distance_m:.3f}"
            )

    cv2.destroyAllWindows()
    print(f"Done. CSV: {csv_path}")
    return 0


def build_error_row(run_id: str, image_id: str, image_path: Path, error: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "image_id": image_id,
        "image_path": str(image_path),
        "experiment": "homography_distance_estimation",
        "provider": "cv",
        "model": "manual_homography",
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
