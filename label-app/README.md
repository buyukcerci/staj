# Tkinter Image Label App

Small desktop labeling tool for measuring lines, rectangles, and polygons on images.

## Setup

Install Python 3.10+ and then install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python label_app.py
```

## Workflow

1. Click `Open Folder` and select a folder containing images.
2. Choose a drawing mode:
   - `Line`: click two points, then enter distance `1-2`.
   - `Rectangle`: drag a rectangle, then enter distances `1-2` and `2-3`.
   - `Polygon`: click each vertex, press `Enter` to finish, then enter each edge distance.
3. Click `Save + Next` to write labels and load the next image.

Labels are saved to `labels.json` in the selected image folder.

## Output Format

```json
{
  "image-name.jpg": [
    {
      "type": "line",
      "points": [[123.4, 55.0], [200.0, 80.0]],
      "distances": {"1-2": "42"}
    }
  ]
}
```
