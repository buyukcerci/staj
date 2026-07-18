"""Shared label definitions for manual benchmark annotation."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
CSV_PATH = LABELS_DIR / "manual_labels.csv"
JSONL_PATH = LABELS_DIR / "manual_labels.jsonl"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

BOOLEAN_LABELS = ("true", "false", "unknown")
RISK_LABELS = ("safe", "warning")
SCENE_TYPES = (
    "safe",
    "ppe_violation",
    "forklift_risk",
    "multiple_risks",
    "unclear",
    "hard_negative",
)

CSV_FIELDS = (
    "image_id",
    "image_path",
    "worker_count",
    "forklift_count",
    "helmet_violation",
    "forklift_person_risk",
    "overall_risk",
    "scene_type",
    "notes",
)
