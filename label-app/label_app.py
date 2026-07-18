import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


class DistanceDialog(tk.Toplevel):
    def __init__(self, parent, title, edge_names):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self.entries = {}

        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")

        ttk.Label(body, text="Enter distances for the drawn edges.").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        for row, edge_name in enumerate(edge_names, start=1):
            ttk.Label(body, text=edge_name).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(body, width=24)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.entries[edge_name] = entry

        buttons = ttk.Frame(body)
        buttons.grid(row=len(edge_names) + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(buttons, text="OK", command=self.ok).grid(row=0, column=1)

        self.bind("<Return>", lambda _event: self.ok())
        self.bind("<Escape>", lambda _event: self.cancel())

        first_entry = next(iter(self.entries.values()), None)
        if first_entry:
            first_entry.focus_set()

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.wait_visibility()
        self.geometry(f"+{parent.winfo_rootx() + 120}+{parent.winfo_rooty() + 120}")
        self.wait_window()

    def ok(self):
        self.result = {name: entry.get().strip() for name, entry in self.entries.items()}
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


class LabelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Distance Labeler")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.folder = None
        self.images = []
        self.image_index = 0
        self.labels = {}
        self.current_image = None
        self.current_photo = None
        self.current_image_size = (1, 1)
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.mode = tk.StringVar(value="line")
        self.status = tk.StringVar(value="Open a folder to start.")
        self.image_name = tk.StringVar(value="No folder selected")
        self.mouse_position = tk.StringVar(value="Mouse: -")

        self.drawing_points = []
        self.preview_item = None
        self.drag_start = None

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        ttk.Button(toolbar, text="Open Folder", command=self.open_folder).grid(row=0, column=0, padx=(0, 8))

        ttk.Label(toolbar, text="Mode:").grid(row=0, column=1, padx=(0, 4))
        for col, (text, value) in enumerate(
            [("Line", "line"), ("Rectangle", "rectangle"), ("Polygon", "polygon")],
            start=2,
        ):
            ttk.Radiobutton(toolbar, text=text, value=value, variable=self.mode, command=self.cancel_drawing).grid(
                row=0, column=col, padx=2
            )

        ttk.Button(toolbar, text="Undo Last", command=self.undo_last_label).grid(row=0, column=5, padx=(12, 4))
        ttk.Button(toolbar, text="Clear Image", command=self.clear_current_labels).grid(row=0, column=6, padx=4)
        ttk.Button(toolbar, text="Save + Next", command=self.save_and_next).grid(row=0, column=7, padx=(12, 0))

        ttk.Label(toolbar, textvariable=self.image_name, anchor="e").grid(row=0, column=8, sticky="e")

        self.canvas = tk.Canvas(self, background="#202124", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        footer = ttk.Frame(self, padding=(8, 4, 8, 8))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status).grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.mouse_position, anchor="e").grid(row=0, column=1, sticky="e")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)
        self.bind("<Return>", self.finish_polygon)
        self.bind("<Escape>", lambda _event: self.cancel_drawing())
        self.bind("<Configure>", self.on_resize)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        self.folder = Path(folder)
        self.images = sorted(
            path for path in self.folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            messagebox.showwarning("No images", "The selected folder does not contain supported image files.")
            return

        self.image_index = 0
        self.labels = self._load_labels()
        self.load_current_image()

    def _labels_path(self):
        return self.folder / "labels.json"

    def _load_labels(self):
        labels_path = self._labels_path()
        if not labels_path.exists():
            return {}

        try:
            with labels_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showwarning("Labels not loaded", f"Could not read labels.json:\n{exc}")
            return {}

        return data if isinstance(data, dict) else {}

    def save_labels(self):
        if not self.folder:
            return

        labels_path = self._labels_path()
        with labels_path.open("w", encoding="utf-8") as file:
            json.dump(self.labels, file, indent=2)

    def load_current_image(self):
        if not self.images:
            return

        self.cancel_drawing()
        image_path = self.images[self.image_index]
        try:
            self.current_image = Image.open(image_path).convert("RGB")
        except OSError as exc:
            messagebox.showerror("Image error", f"Could not open {image_path.name}:\n{exc}")
            self.next_image()
            return

        self.current_image_size = self.current_image.size
        self.image_name.set(f"{self.image_index + 1}/{len(self.images)}  {image_path.name}")
        self.status.set(self._help_text())
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        if not self.current_image:
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_width, image_height = self.current_image_size
        self.scale = min(canvas_width / image_width, canvas_height / image_height)
        display_width = max(1, int(image_width * self.scale))
        display_height = max(1, int(image_height * self.scale))
        self.offset_x = (canvas_width - display_width) // 2
        self.offset_y = (canvas_height - display_height) // 2

        resized = self.current_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.current_photo, tags=("image",))

        self.draw_saved_labels()
        self.draw_working_shape()

    def draw_saved_labels(self):
        image_name = self.images[self.image_index].name
        for label in self.labels.get(image_name, []):
            points = label.get("points", [])
            if len(points) < 2:
                continue

            canvas_points = [self.image_to_canvas(x, y) for x, y in points]
            flat_points = [coord for point in canvas_points for coord in point]
            shape_type = label.get("type")
            if shape_type == "line":
                self.canvas.create_line(*flat_points, fill="#00d1ff", width=3, tags=("label",))
            else:
                self.canvas.create_polygon(
                    *flat_points,
                    outline="#ffcc00",
                    fill="",
                    width=3,
                    tags=("label",),
                )

            for idx, (x, y) in enumerate(canvas_points, start=1):
                self.draw_vertex(x, y, str(idx), color="#ffffff")

            self.draw_distances(label, canvas_points)

    def draw_working_shape(self):
        if not self.drawing_points:
            return

        canvas_points = [self.image_to_canvas(x, y) for x, y in self.drawing_points]
        for idx, (x, y) in enumerate(canvas_points, start=1):
            self.draw_vertex(x, y, str(idx), color="#ff5252")

        if len(canvas_points) > 1:
            if self.mode.get() == "rectangle" and len(canvas_points) == 4:
                closed_points = canvas_points + [canvas_points[0]]
                flat_points = [coord for point in closed_points for coord in point]
                self.canvas.create_line(*flat_points, fill="#ff5252", width=2, dash=(5, 4), tags=("working",))
            else:
                flat_points = [coord for point in canvas_points for coord in point]
                self.canvas.create_line(*flat_points, fill="#ff5252", width=2, dash=(5, 4), tags=("working",))

    def draw_vertex(self, x, y, label, color="#ffffff"):
        radius = 4
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#111111")
        self.canvas.create_text(x + 9, y - 9, text=label, fill=color, anchor="w", font=("Segoe UI", 9, "bold"))

    def draw_distances(self, label, canvas_points):
        distances = label.get("distances") or {}
        for edge_name, distance in distances.items():
            if not distance:
                continue

            edge = self.edge_points_from_name(edge_name, canvas_points)
            if edge is None:
                continue

            (x1, y1), (x2, y2) = edge
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = f"{edge_name}: {distance}"
            text_item = self.canvas.create_text(
                mid_x,
                mid_y,
                text=text,
                fill="#111111",
                font=("Segoe UI", 10, "bold"),
                tags=("label",),
            )
            bbox = self.canvas.bbox(text_item)
            if bbox:
                pad_x = 5
                pad_y = 3
                background = self.canvas.create_rectangle(
                    bbox[0] - pad_x,
                    bbox[1] - pad_y,
                    bbox[2] + pad_x,
                    bbox[3] + pad_y,
                    fill="#fff3b0",
                    outline="#805f00",
                    tags=("label",),
                )
                self.canvas.tag_lower(background, text_item)

    def edge_points_from_name(self, edge_name, canvas_points):
        try:
            start_text, end_text = edge_name.split("-", 1)
            start_index = int(start_text) - 1
            end_index = int(end_text) - 1
        except ValueError:
            return None

        if not (0 <= start_index < len(canvas_points) and 0 <= end_index < len(canvas_points)):
            return None
        return canvas_points[start_index], canvas_points[end_index]

    def on_canvas_click(self, event):
        if not self.current_image or not self.point_inside_image(event.x, event.y):
            return

        point = self.canvas_to_image(event.x, event.y)
        active_mode = self.mode.get()

        if active_mode == "line":
            self.drawing_points.append(point)
            self.redraw()
            if len(self.drawing_points) == 2:
                self.add_shape("line", self.drawing_points[:])
                self.cancel_drawing()
            else:
                self.status.set("Click the second point for the line.")

        elif active_mode == "polygon":
            self.drawing_points.append(point)
            self.status.set("Click more vertices, then press Enter to finish the polygon.")
            self.redraw()

        elif active_mode == "rectangle":
            if self.drag_start is None:
                self.drag_start = point
                self.drawing_points = [point]
                self.status.set("Move to the opposite corner, then click to finish the rectangle.")
                self.redraw()
                return

            if self.add_rectangle_from_corners(self.drag_start, point):
                self.cancel_drawing()

    def on_canvas_drag(self, event):
        if self.mode.get() != "rectangle" or self.drag_start is None or not self.current_image:
            return

        current = self.clamp_canvas_event_to_image(event)
        if current is None:
            return

        self.remove_preview()
        self.drawing_points = self.rectangle_points(self.drag_start, current)
        self.redraw()

    def on_canvas_release(self, event):
        if self.mode.get() != "rectangle" or self.drag_start is None or not self.current_image:
            return

        end = self.clamp_canvas_event_to_image(event)
        if end is None:
            return

        if self.add_rectangle_from_corners(self.drag_start, end):
            self.cancel_drawing()

    def add_rectangle_from_corners(self, start, end):
        x1, y1 = start
        x2, y2 = end
        if math.hypot(x2 - x1, y2 - y1) < 3:
            return False

        points = self.rectangle_points(start, end)
        self.add_shape("rectangle", points)
        return True

    def on_canvas_motion(self, event):
        self.update_mouse_position(event.x, event.y)

        if self.mode.get() == "rectangle" and self.drag_start is not None and self.current_image:
            current = self.clamp_canvas_event_to_image(event)
            if current is not None:
                self.drawing_points = self.rectangle_points(self.drag_start, current)
                self.redraw()
            return

        if self.mode.get() != "polygon" or len(self.drawing_points) < 1 or not self.current_image:
            return

        current = self.clamp_canvas_event_to_image(event)
        self.redraw()
        if current is None:
            return

        points = self.drawing_points + [current]
        canvas_points = [self.image_to_canvas(x, y) for x, y in points]
        flat_points = [coord for point in canvas_points for coord in point]
        self.canvas.create_line(*flat_points, fill="#ff5252", width=2, dash=(5, 4), tags=("preview",))

    def on_canvas_leave(self, _event):
        self.mouse_position.set("Mouse: -")

    def finish_polygon(self, _event=None):
        if self.mode.get() != "polygon" or not self.current_image:
            return

        if len(self.drawing_points) < 3:
            self.status.set("A polygon needs at least three points.")
            return

        self.add_shape("polygon", self.drawing_points[:])
        self.cancel_drawing()

    def add_shape(self, shape_type, points):
        edge_names = self.edge_names(shape_type, len(points))
        dialog = DistanceDialog(self, f"{shape_type.title()} distances", edge_names)
        if dialog.result is None:
            return

        image_name = self.images[self.image_index].name
        label = {
            "type": shape_type,
            "points": [[round(x, 2), round(y, 2)] for x, y in points],
            "distances": dialog.result,
        }
        self.labels.setdefault(image_name, []).append(label)
        self.status.set(f"Added {shape_type}. Click Save + Next when this image is done.")
        self.redraw()

    def edge_names(self, shape_type, point_count):
        if shape_type == "line":
            return ["1-2"]
        if shape_type == "rectangle":
            return ["1-2", "2-3"]
        return [f"{idx}-{idx + 1}" for idx in range(1, point_count)] + [f"{point_count}-1"]

    def rectangle_points(self, start, end):
        x1, y1 = start
        x2, y2 = end
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    def save_and_next(self):
        if not self.images:
            return

        self.save_labels()
        if self.image_index >= len(self.images) - 1:
            self.status.set("Saved. This is the last image.")
            messagebox.showinfo("Done", "Labels saved. This is the last image.")
            return

        self.image_index += 1
        self.load_current_image()

    def next_image(self):
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
            self.load_current_image()

    def undo_last_label(self):
        if not self.images:
            return
        image_name = self.images[self.image_index].name
        if self.labels.get(image_name):
            self.labels[image_name].pop()
            if not self.labels[image_name]:
                self.labels.pop(image_name)
            self.status.set("Removed the last label from this image.")
            self.redraw()

    def clear_current_labels(self):
        if not self.images:
            return
        image_name = self.images[self.image_index].name
        if image_name in self.labels and messagebox.askyesno("Clear labels", "Clear all labels for this image?"):
            self.labels.pop(image_name)
            self.status.set("Cleared labels for this image.")
            self.redraw()

    def cancel_drawing(self):
        self.drawing_points = []
        self.drag_start = None
        self.remove_preview()
        self.status.set(self._help_text())
        if self.current_image:
            self.redraw()

    def remove_preview(self):
        if self.preview_item is not None:
            self.canvas.delete(self.preview_item)
            self.preview_item = None
        self.canvas.delete("preview")

    def on_resize(self, event):
        if event.widget == self and self.current_image:
            self.after_idle(self.redraw)

    def _help_text(self):
        active_mode = self.mode.get()
        if active_mode == "line":
            return "Line mode: click two points, then enter the 1-2 distance."
        if active_mode == "rectangle":
            return "Rectangle mode: drag the box, then enter distances 1-2 and 2-3."
        return "Polygon mode: click vertices, press Enter to finish, then enter edge distances."

    def point_inside_image(self, canvas_x, canvas_y):
        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        width, height = self.current_image_size
        return 0 <= image_x <= width and 0 <= image_y <= height

    def clamp_canvas_event_to_image(self, event):
        if not self.current_image:
            return None
        x, y = self.canvas_to_image(event.x, event.y)
        width, height = self.current_image_size
        return (min(max(x, 0), width), min(max(y, 0), height))

    def update_mouse_position(self, canvas_x, canvas_y):
        if not self.current_image:
            self.mouse_position.set("Mouse: -")
            return

        image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)
        width, height = self.current_image_size
        if 0 <= image_x <= width and 0 <= image_y <= height:
            self.mouse_position.set(f"Mouse: x={image_x:.1f}, y={image_y:.1f}")
        else:
            self.mouse_position.set("Mouse: outside image")

    def canvas_to_image(self, canvas_x, canvas_y):
        return ((canvas_x - self.offset_x) / self.scale, (canvas_y - self.offset_y) / self.scale)

    def image_to_canvas(self, image_x, image_y):
        return (image_x * self.scale + self.offset_x, image_y * self.scale + self.offset_y)


if __name__ == "__main__":
    app = LabelApp()
    app.mainloop()
