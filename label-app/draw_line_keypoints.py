import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_LABELS = Path("images") / "labels.json"
DEFAULT_RAW_DIR = Path("images") / "images" / "raw"
DEFAULT_OUTPUT_DIR = Path("images") / "images" / "with keypoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw saved line keypoints on raw images and save copies to an output folder."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to labels.json.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Folder containing raw images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where keypoint images will be written.",
    )
    parser.add_argument("--radius", type=int, default=10, help="Keypoint circle radius in pixels.")
    parser.add_argument("--point-color", default="#ff1f1f", help="Keypoint fill color.")
    parser.add_argument("--outline-color", default="#ffffff", help="Keypoint outline color.")
    parser.add_argument("--label-color", default="#ffffff", help="Point number text color.")
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Draw keypoints for every shape type, not only line labels.",
    )
    return parser.parse_args()


def load_labels(labels_path):
    with labels_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{labels_path} must contain a JSON object keyed by image filename.")
    return data


def load_font(size=28):
    for font_name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_keypoint(draw, x, y, index, radius, point_color, outline_color, label_color, font):
    left = x - radius
    top = y - radius
    right = x + radius
    bottom = y + radius
    draw.ellipse((left, top, right, bottom), fill=point_color, outline=outline_color, width=3)

    text = str(index)
    text_x = x + radius + 5
    text_y = y - radius - 3
    draw.text((text_x + 1, text_y + 1), text, fill="#000000", font=font)
    draw.text((text_x, text_y), text, fill=label_color, font=font)


def draw_image_keypoints(image_path, labels, output_path, args, font):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    drawn_count = 0

    for shape in labels:
        if not args.all_shapes and shape.get("type") != "line":
            continue

        points = shape.get("points") or []
        for point_index, point in enumerate(points, start=1):
            if len(point) != 2:
                continue
            x, y = float(point[0]), float(point[1])
            draw_keypoint(
                draw,
                x,
                y,
                point_index,
                args.radius,
                args.point_color,
                args.outline_color,
                args.label_color,
                font,
            )
            drawn_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return drawn_count


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    font = load_font()

    saved = 0
    for image_name, image_labels in labels.items():
        image_path = args.raw_dir / image_name
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        output_path = args.output_dir / image_name
        count = draw_image_keypoints(image_path, image_labels, output_path, args, font)
        print(f"Saved {output_path} ({count} keypoints)")
        saved += 1

    print(f"Done. Saved {saved} image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
