"""Interactive image labeling tool for the CV/LLM benchmark dataset."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit("Tkinter is required to run the labeling UI.") from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Pillow is required. Install it with: pip install -r requirements.txt"
    ) from exc

from label_config import (
    BOOLEAN_LABELS,
    CSV_FIELDS,
    CSV_PATH,
    IMAGE_EXTENSIONS,
    IMAGES_DIR,
    JSONL_PATH,
    LABELS_DIR,
    RISK_LABELS,
    SCENE_TYPES,
)


MAX_IMAGE_WIDTH = 980
MAX_IMAGE_HEIGHT = 640


@dataclass(frozen=True)
class ImageItem:
    image_id: str
    path: Path


def collect_images(images_dir: Path) -> List[ImageItem]:
    """Return all supported images sorted by relative path."""
    images: List[ImageItem] = []
    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = path.relative_to(images_dir)
            image_id = rel.with_suffix("").as_posix().replace("/", "__")
            images.append(ImageItem(image_id=image_id, path=path))
    return images


def load_existing_labels(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load existing labels keyed by image_id."""
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("image_id", "")).strip(): dict(row)
            for row in reader
            if str(row.get("image_id", "")).strip()
        }


def write_outputs(labels: Dict[str, Dict[str, str]]) -> None:
    """Persist labels to CSV and JSONL."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ordered = [labels[key] for key in sorted(labels)]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in ordered:
            payload = {field: row.get(field, "") for field in CSV_FIELDS}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class LabelingApp:
    """Tkinter image labeling workflow."""

    def __init__(self, root: tk.Tk, images: List[ImageItem], labels: Dict[str, Dict[str, str]]) -> None:
        self.root = root
        self.images = images
        self.labels = labels
        self.index = self._first_unlabeled_index()
        self.photo: ImageTk.PhotoImage | None = None

        self.worker_count = tk.IntVar(value=0)
        self.forklift_count = tk.IntVar(value=0)
        self.helmet_violation = tk.StringVar(value="unknown")
        self.forklift_person_risk = tk.StringVar(value="unknown")
        self.overall_risk = tk.StringVar(value="safe")
        self.scene_type = tk.StringVar(value="unclear")

        self._build_ui()
        self._load_current_image()

    def _first_unlabeled_index(self) -> int:
        for idx, item in enumerate(self.images):
            if item.image_id not in self.labels:
                return idx
        return 0

    def _build_ui(self) -> None:
        self.root.title("CV-LLM Benchmark Labeler")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 700)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right.pack_propagate(False)

        self.status_label = ttk.Label(left, text="", font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor=tk.W, pady=(0, 8))

        self.image_label = ttk.Label(left, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.path_label = ttk.Label(left, text="", foreground="#5f6b76")
        self.path_label.pack(anchor=tk.W, pady=(8, 0))

        self._build_form(right)
        self._build_buttons(right)

        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<Left>", lambda _event: self.previous_image())

    def _build_form(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Manual Labels", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W)
        ttk.Label(parent, text="Use unknown/unclear when the image is ambiguous.").pack(anchor=tk.W, pady=(2, 14))

        self._number_field(parent, "Worker count", self.worker_count)
        self._number_field(parent, "Forklift count", self.forklift_count)
        self._choice_field(parent, "Helmet violation", self.helmet_violation, BOOLEAN_LABELS)
        self._choice_field(parent, "Forklift-person risk", self.forklift_person_risk, BOOLEAN_LABELS)
        self._choice_field(parent, "Overall risk", self.overall_risk, RISK_LABELS)
        self._choice_field(parent, "Scene type", self.scene_type, SCENE_TYPES)

        ttk.Label(parent, text="Notes").pack(anchor=tk.W, pady=(10, 3))
        self.notes_text = tk.Text(parent, height=8, wrap=tk.WORD)
        self.notes_text.pack(fill=tk.X)

    def _build_buttons(self, parent: ttk.Frame) -> None:
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(18, 0))

        ttk.Button(button_frame, text="Previous", command=self.previous_image).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="Save", command=self.save_current).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="Save & Next", command=self.save_and_next).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="Next", command=self.next_image).pack(fill=tk.X, pady=3)
        ttk.Button(button_frame, text="Skip unlabeled", command=self.skip_unlabeled).pack(fill=tk.X, pady=3)

        ttk.Label(
            parent,
            text="Shortcuts: Ctrl+S save, Left/Right navigate",
            foreground="#5f6b76",
        ).pack(anchor=tk.W, pady=(12, 0))

    @staticmethod
    def _number_field(parent: ttk.Frame, label: str, variable: tk.IntVar) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(8, 3))
        spin = ttk.Spinbox(parent, from_=0, to=99, textvariable=variable, width=8)
        spin.pack(anchor=tk.W)

    @staticmethod
    def _choice_field(parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...]) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W, pady=(8, 3))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill=tk.X)

    def _load_current_image(self) -> None:
        if not self.images:
            self.status_label.config(text="No images found.")
            self.path_label.config(text=f"Put images in: {IMAGES_DIR}")
            self.image_label.config(text="No images available.")
            return

        item = self.images[self.index]
        self.status_label.config(
            text=f"Image {self.index + 1}/{len(self.images)} - {item.image_id}"
        )
        self.path_label.config(text=str(item.path))
        self._display_image(item.path)
        self._load_form_values(item)

    def _display_image(self, path: Path) -> None:
        try:
            image = Image.open(path)
            image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo, text="")
        except Exception as exc:  # pragma: no cover - UI boundary
            self.photo = None
            self.image_label.config(image="", text=f"Failed to load image:\n{path}\n\n{exc}")

    def _load_form_values(self, item: ImageItem) -> None:
        row = self.labels.get(item.image_id, {})
        self.worker_count.set(self._parse_int(row.get("worker_count", "0")))
        self.forklift_count.set(self._parse_int(row.get("forklift_count", "0")))
        self.helmet_violation.set(row.get("helmet_violation") or "unknown")
        self.forklift_person_risk.set(row.get("forklift_person_risk") or "unknown")
        self.overall_risk.set(row.get("overall_risk") or "safe")
        self.scene_type.set(row.get("scene_type") or "unclear")
        self.notes_text.delete("1.0", tk.END)
        self.notes_text.insert("1.0", row.get("notes", ""))

    @staticmethod
    def _parse_int(raw: str) -> int:
        try:
            return max(0, int(str(raw).strip()))
        except ValueError:
            return 0

    def _current_row(self) -> Dict[str, str]:
        item = self.images[self.index]
        notes = self.notes_text.get("1.0", tk.END).strip()
        try:
            worker_count = max(0, int(self.worker_count.get()))
            forklift_count = max(0, int(self.forklift_count.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid counts", "Worker and forklift counts must be numbers.")
            raise

        return {
            "image_id": item.image_id,
            "image_path": str(item.path),
            "worker_count": str(worker_count),
            "forklift_count": str(forklift_count),
            "helmet_violation": self.helmet_violation.get(),
            "forklift_person_risk": self.forklift_person_risk.get(),
            "overall_risk": self.overall_risk.get(),
            "scene_type": self.scene_type.get(),
            "notes": notes,
        }

    def save_current(self) -> None:
        if not self.images:
            return
        try:
            row = self._current_row()
        except (tk.TclError, ValueError):
            return
        self.labels[row["image_id"]] = row
        write_outputs(self.labels)
        self.status_label.config(
            text=f"Saved {self.index + 1}/{len(self.images)} - {row['image_id']}"
        )

    def save_and_next(self) -> None:
        self.save_current()
        self.next_image()

    def previous_image(self) -> None:
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self._load_current_image()

    def next_image(self) -> None:
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self._load_current_image()

    def skip_unlabeled(self) -> None:
        if not self.images:
            return
        start = self.index
        for offset in range(1, len(self.images) + 1):
            candidate = (start + offset) % len(self.images)
            if self.images[candidate].image_id not in self.labels:
                self.index = candidate
                self._load_current_image()
                return
        messagebox.showinfo("Done", "All images have labels.")

    def on_close(self) -> None:
        if messagebox.askyesno("Exit", "Save current image before closing?"):
            self.save_current()
        self.root.destroy()


def main() -> int:
    images = collect_images(IMAGES_DIR)
    labels = load_existing_labels(CSV_PATH)

    root = tk.Tk()
    app = LabelingApp(root, images, labels)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
